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
    pub weights: Vec<CouplingWeight>, // sparse: alleen niet-nul weights
}

#[derive(Serialize, Deserialize)]
pub struct CouplingWeight {
    pub point_id: u32,
    pub coupling_idx: u16,
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
                    weight: c.weight,
                });
            }
        }
    }
    WeightSnapshot { tick: field.tick, weights }
}

/// Merge remote weights met lokale weights via FedAvg
/// alpha = 0.5 = gelijk gewicht, 0.1 = 10% remote invloed
pub fn merge_weights(field: &mut Field, remote: &WeightSnapshot, alpha: f32) -> usize {
    let mut merged = 0;
    for rw in &remote.weights {
        let pid = rw.point_id as usize;
        let ci = rw.coupling_idx as usize;
        if pid < field.points.len() && ci < field.points[pid].couplings.len() {
            let local = field.points[pid].couplings[ci].weight;
            // FedAvg: gewogen gemiddelde
            field.points[pid].couplings[ci].weight = (1.0 - alpha) * local + alpha * rw.weight;
            merged += 1;
        }
    }
    tracing::info!(merged, remote_tick = remote.tick, alpha, "Federated merge complete");
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
