//! # gpu_client — Remote GPU bridge
//!
//! Lucy stuurt velddata naar de Vast.ai GPU service.
//! Communiceert via bincode over HTTP voor minimale overhead.

use lucy_state::*;
use serde::{Serialize, Deserialize};

/// Snapshot formaat — identiek aan lucy-gpu's FieldSnapshot
#[derive(Serialize, Deserialize)]
struct FieldSnapshot {
    phi: Vec<f32>,
    prediction: Vec<f32>,
    memory: Vec<f32>,
    tension: Vec<f32>,
    tension_ema: Vec<f32>,
    energy: Vec<f32>,
    eta: Vec<f32>,
    curiosity: Vec<f32>,
    satisfaction: Vec<f32>,
    frustration: Vec<f32>,
    coupling_offsets: Vec<u32>,
    coupling_targets: Vec<u32>,
    coupling_weights: Vec<f32>,
    coupling_momentum: Vec<f32>,
    n_points: u32,
    lambda: f32,
    beta: f32,
    alpha: f32,
    rho: f32,
    mu: f32,
    kappa: f32,
    omega: f32,
    ema_decay: f32,
    growth_threshold: f32,
}

pub struct GpuClient {
    base_url: String,
    client: reqwest::Client,
    uploaded: bool,
}

impl GpuClient {
    pub fn new(gpu_url: &str) -> Self {
        Self {
            base_url: gpu_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
            uploaded: false,
        }
    }

    /// Check of de GPU service bereikbaar is
    pub async fn is_available(&self) -> bool {
        self.client.get(format!("{}/status", self.base_url))
            .send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Upload volledige veldstaat naar GPU
    pub async fn upload(&mut self, field: &Field) -> Result<(), String> {
        let snap = self.field_to_snapshot(field);
        let bytes = bincode::serialize(&snap).map_err(|e| format!("serialize: {e}"))?;

        let resp = self.client.post(format!("{}/upload", self.base_url))
            .header("content-type", "application/octet-stream")
            .body(bytes)
            .send().await
            .map_err(|e| format!("upload: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("upload failed: {}", resp.status()));
        }

        self.uploaded = true;
        tracing::info!("Field uploaded to remote GPU");
        Ok(())
    }

    /// Run N ticks op de GPU
    pub async fn tick(&self, n: u32) -> Result<(), String> {
        if !self.uploaded { return Err("niet geüpload".into()); }

        let resp = self.client.post(format!("{}/tick", self.base_url))
            .json(&serde_json::json!({"n_ticks": n}))
            .send().await
            .map_err(|e| format!("tick: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("tick failed: {}", resp.status()));
        }
        Ok(())
    }

    /// Run weight update op GPU
    pub async fn update_weights(&self) -> Result<(), String> {
        if !self.uploaded { return Err("niet geüpload".into()); }

        self.client.post(format!("{}/weights", self.base_url))
            .send().await
            .map_err(|e| format!("weights: {e}"))?;
        Ok(())
    }

    /// Upload clamps naar GPU
    pub async fn set_clamps(&self, field: &Field) -> Result<(), String> {
        if !self.uploaded { return Err("niet geüpload".into()); }

        let n = field.points.len();
        let mut mask = vec![0u32; n];
        let mut values = vec![0.0f32; n * DIM];

        for (&id, clamp) in &field.clamps {
            let idx = id as usize;
            if idx < n {
                mask[idx] = 1;
                values[idx * DIM..(idx + 1) * DIM].copy_from_slice(clamp);
            }
        }

        #[derive(Serialize)]
        struct ClampData { mask: Vec<u32>, values: Vec<f32> }

        let bytes = bincode::serialize(&ClampData { mask, values })
            .map_err(|e| format!("serialize: {e}"))?;

        self.client.post(format!("{}/clamps", self.base_url))
            .header("content-type", "application/octet-stream")
            .body(bytes)
            .send().await
            .map_err(|e| format!("clamps: {e}"))?;
        Ok(())
    }

    /// Sync hyperparameters naar GPU
    pub async fn sync_params(&self, field: &Field) -> Result<(), String> {
        if !self.uploaded { return Err("niet geüpload".into()); }

        self.client.post(format!("{}/params", self.base_url))
            .json(&serde_json::json!({
                "lambda": field.lambda,
                "beta": field.beta,
                "alpha": field.alpha,
                "rho": field.rho,
                "mu": field.mu,
                "kappa": field.kappa,
                "omega": field.omega,
                "ema_decay": field.ema_decay,
                "growth_threshold": field.growth_threshold,
            }))
            .send().await
            .map_err(|e| format!("params: {e}"))?;
        Ok(())
    }

    /// Download veldstaat van GPU terug naar Field
    pub async fn download(&self, field: &mut Field) -> Result<(), String> {
        if !self.uploaded { return Err("niet geüpload".into()); }

        let resp = self.client.get(format!("{}/download", self.base_url))
            .send().await
            .map_err(|e| format!("download: {e}"))?;

        let bytes = resp.bytes().await.map_err(|e| format!("read: {e}"))?;
        let snap: FieldSnapshot = bincode::deserialize(&bytes)
            .map_err(|e| format!("deserialize: {e}"))?;

        let n = snap.n_points as usize;
        for i in 0..n.min(field.points.len()) {
            field.points[i].phi.copy_from_slice(&snap.phi[i * DIM..(i + 1) * DIM]);
            field.points[i].tension = snap.tension[i];
            field.points[i].tension_ema = snap.tension_ema[i];
            field.points[i].curiosity = snap.curiosity[i];
            field.points[i].satisfaction = snap.satisfaction[i];
            field.points[i].frustration = snap.frustration[i];

            // Sync coupling weights terug
            if !snap.coupling_weights.is_empty() {
                // Weights worden via CSR teruggeschreven
                // We moeten de offset info van het lokale veld gebruiken
                let mut offset = 0usize;
                for j in 0..i {
                    offset += field.points[j].couplings.len();
                }
                for (ci, c) in field.points[i].couplings.iter_mut().enumerate() {
                    let idx = offset + ci;
                    if idx < snap.coupling_weights.len() {
                        c.weight = snap.coupling_weights[idx];
                        c.momentum = snap.coupling_momentum[idx];
                    }
                }
            }
        }

        Ok(())
    }

    fn field_to_snapshot(&self, field: &Field) -> FieldSnapshot {
        let n = field.points.len();
        let mut phi = Vec::with_capacity(n * DIM);
        let mut prediction = Vec::with_capacity(n * DIM);
        let mut memory = Vec::with_capacity(n * DIM);
        let mut tension = Vec::with_capacity(n);
        let mut tension_ema = Vec::with_capacity(n);
        let mut energy = Vec::with_capacity(n);
        let mut eta = Vec::with_capacity(n);
        let mut curiosity = Vec::with_capacity(n);
        let mut satisfaction = Vec::with_capacity(n);
        let mut frustration = Vec::with_capacity(n);

        let mut offsets = Vec::with_capacity(n + 1);
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        let mut momentum = Vec::new();

        let mut offset = 0u32;
        for p in &field.points {
            phi.extend_from_slice(&p.phi);
            prediction.extend_from_slice(&p.prediction);
            memory.extend_from_slice(&p.memory);
            tension.push(p.tension);
            tension_ema.push(p.tension_ema);
            energy.push(p.energy);
            eta.push(p.eta);
            curiosity.push(p.curiosity);
            satisfaction.push(p.satisfaction);
            frustration.push(p.frustration);

            offsets.push(offset);
            for c in &p.couplings {
                targets.push(c.target);
                weights.push(c.weight);
                momentum.push(c.momentum);
            }
            offset += p.couplings.len() as u32;
        }
        offsets.push(offset);

        if targets.is_empty() {
            targets.push(0);
            weights.push(0.0);
            momentum.push(0.0);
        }

        FieldSnapshot {
            phi, prediction, memory, tension, tension_ema,
            energy, eta, curiosity, satisfaction, frustration,
            coupling_offsets: offsets, coupling_targets: targets,
            coupling_weights: weights, coupling_momentum: momentum,
            n_points: n as u32,
            lambda: field.lambda, beta: field.beta, alpha: field.alpha,
            rho: field.rho, mu: field.mu, kappa: field.kappa,
            omega: field.omega, ema_decay: field.ema_decay,
            growth_threshold: field.growth_threshold,
        }
    }
}
