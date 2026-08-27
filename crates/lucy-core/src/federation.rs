//! # federation — Federated Learning Across Lucy Instances
//!
//! Meerdere Lucy instances leren parallel op verschillende data.
//! Periodiek mergen ze hun weights via FedAvg.

use lucy_state::*;
use serde::{Serialize, Deserialize};

/// Compact weight snapshot voor sync (alleen coupling weights)
#[derive(Serialize, Deserialize)]
pub struct WeightSnapshot {
    pub tick: u64,
    /// v3.1 compat-gate: merge is alleen betekenisvol tussen topologie-identieke velden
    /// (zelfde DIM en puntenaantal = dezelfde snapshot-lijn). Zie `is_compatible`.
    #[serde(default)]
    pub dim: usize,
    #[serde(default)]
    pub n_points: usize,
    pub weights: Vec<CouplingWeight>, // sparse: alleen niet-nul weights
}

/// v3.1: topologie/versie-gate voor federated merge. FedAvg is alleen zinvol tussen velden met
/// identieke topologie — anders wijzen (point_id, coupling_idx) naar verschillende concepten en
/// is het middelen stille corruptie. Nodes uit dezelfde GitHub-backup-lijn zijn compatibel.
pub fn is_compatible(field: &Field, remote: &WeightSnapshot) -> Result<(), String> {
    if remote.dim != DIM {
        return Err(format!("DIM mismatch: remote={} lokaal={}", remote.dim, DIM));
    }
    if remote.n_points != field.points.len() {
        return Err(format!(
            "n_points mismatch: remote={} lokaal={}",
            remote.n_points, field.points.len()
        ));
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct CouplingWeight {
    pub point_id: u32,
    pub coupling_idx: u16,
    /// edge-identity key: het doelpunt van deze coupling. Zonder deze check middelt
    /// merge_weights positioneel edges naar verschillende targets = stille corruptie.
    pub target: u32,
    pub weight: f32,
}

/// Exporteer alle coupling weights als compact snapshot
pub fn export_weights(field: &Field) -> WeightSnapshot {
    let mut weights = Vec::new();
    for &id in &field.active_set {
        let idx = id as usize;
        for (ci, c) in field.points[idx].couplings.iter().enumerate() {
            if c.weight.abs() > 0.001 { // skip near-zero weights
                weights.push(CouplingWeight {
                    point_id: id,
                    coupling_idx: ci as u16,
                    target: c.target,
                    weight: c.weight,
                });
            }
        }
    }
    WeightSnapshot { tick: field.tick, dim: DIM, n_points: field.points.len(), weights }
}

/// Merge remote weights met lokale weights via FedAvg.
/// alpha = 0.5 = gelijk gewicht, 0.1 = 10% remote invloed.
/// v3.1: compat-gated + finite-guards + clamp — weigert incompatibele topologie (geen corruptie)
/// en laat geen NaN/extreme remote-waarde het veld in (voorkomt tension-runaway na injectie).
pub fn merge_weights(field: &mut Field, remote: &WeightSnapshot, alpha: f32) -> usize {
    if let Err(e) = is_compatible(field, remote) {
        tracing::warn!(error = %e, "Federated merge GEWEIGERD — incompatibele topologie");
        return 0;
    }
    let a = alpha.clamp(0.0, 1.0);
    let mut merged = 0;
    for rw in &remote.weights {
        if !rw.weight.is_finite() { continue; }
        let pid = rw.point_id as usize;
        let ci = rw.coupling_idx as usize;
        if pid < field.points.len() && ci < field.points[pid].couplings.len() {
            let local_c = &field.points[pid].couplings[ci];
            // edge-identity check: alleen middelen als beide nodes op dit slot dezelfde
            // edge (zelfde target) hebben. Na autonome grow()/prune() kan positie ci naar
            // verschillende targets wijzen bij gelijk n_points — dan overslaan (fail-safe).
            if local_c.target != rw.target { continue; }
            let local = local_c.weight;
            let nw = (1.0 - a) * local + a * rw.weight;
            if nw.is_finite() {
                // clamp binnen realistisch gewicht-bereik (zoals update_weights elders)
                field.points[pid].couplings[ci].weight = nw.clamp(-2.5, 2.5);
                merged += 1;
            }
        }
    }
    tracing::info!(merged, remote_tick = remote.tick, alpha = a, "Federated merge complete");
    merged
}

/// Haal weights op van een remote replica en merge
pub async fn sync_from_replica(field: &mut Field, replica_url: &str) -> Result<usize, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Client error: {}", e))?;

    let url = format!("{}/weights", replica_url);
    let resp = client.get(&url)
        .send()
        .await
        .map_err(|e| format!("Fetch weights failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Replica returned {}", resp.status()));
    }

    let bytes = resp.bytes().await.map_err(|e| format!("Read failed: {}", e))?;
    let snapshot: WeightSnapshot = bincode::deserialize(&bytes)
        .map_err(|e| format!("Deserialize failed: {}", e))?;

    Ok(merge_weights(field, &snapshot, 0.3)) // 30% remote invloed
}

/// Push lokale weights naar een replica
pub async fn push_to_replica(field: &Field, replica_url: &str) -> Result<(), String> {
    let snapshot = export_weights(field);
    let bytes = bincode::serialize(&snapshot)
        .map_err(|e| format!("Serialize failed: {}", e))?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Client error: {}", e))?;

    let url = format!("{}/sync", replica_url);
    let resp = client.post(&url)
        .header("Content-Type", "application/octet-stream")
        .body(bytes)
        .send()
        .await
        .map_err(|e| format!("Push failed: {}", e))?;

    if resp.status().is_success() {
        tracing::info!(url = %replica_url, "Weights pushed to replica");
        Ok(())
    } else {
        Err(format!("Replica returned {}", resp.status()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(dim: usize, n_points: usize, weights: Vec<CouplingWeight>) -> WeightSnapshot {
        WeightSnapshot { tick: 1, dim, n_points, weights }
    }

    #[test]
    fn compat_rejects_dim_mismatch() {
        let field = Field::new();
        assert!(is_compatible(&field, &snap(DIM + 1, field.points.len(), vec![])).is_err());
    }

    #[test]
    fn compat_rejects_npoints_mismatch() {
        let field = Field::new();
        assert!(is_compatible(&field, &snap(DIM, field.points.len() + 7, vec![])).is_err());
    }

    #[test]
    fn compat_accepts_identical_topology() {
        let field = Field::new();
        assert!(is_compatible(&field, &snap(DIM, field.points.len(), vec![])).is_ok());
    }

    #[test]
    fn merge_refuses_incompatible_topology() {
        let mut field = Field::new();
        let n = field.points.len();
        let s = snap(DIM, n + 1, vec![CouplingWeight { point_id: 0, coupling_idx: 0, target: 0, weight: 1.0 }]);
        // incompatibele topologie → 0 merges, geen corruptie
        assert_eq!(merge_weights(&mut field, &s, 0.3), 0);
    }

    #[test]
    fn merge_skips_nonfinite_without_panic() {
        let mut field = Field::new();
        let n = field.points.len();
        let s = snap(DIM, n, vec![CouplingWeight { point_id: 0, coupling_idx: 0, target: 0, weight: f32::NAN }]);
        // compatibele topologie maar NaN-gewicht → overgeslagen, geen panic
        assert_eq!(merge_weights(&mut field, &s, 0.3), 0);
    }
}
