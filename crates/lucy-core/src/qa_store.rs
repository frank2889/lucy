use lucy_state::DIM;
use serde::{Deserialize, Serialize};
use std::path::Path;

const MAX_PAIRS: usize = 500;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QaPair {
    pub question: String,
    pub answer: String,
    pub fingerprint: [f32; DIM],
    pub hit_count: u32,
    pub created_tick: u64,
    pub field_point_id: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QaStore {
    pub pairs: Vec<QaPair>,
}

impl QaStore {
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Zoek de beste match voor een fingerprint
    /// Retourneert (index, similarity) als boven threshold
    pub fn find_nearest(&self, fp: &[f32; DIM], threshold: f32) -> Option<(usize, f32)> {
        let mut best_idx = 0;
        let mut best_sim = f32::MIN;

        for (i, pair) in self.pairs.iter().enumerate() {
            let sim = super::proxy::similarity(fp, &pair.fingerprint);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        if best_sim >= threshold && !self.pairs.is_empty() {
            Some((best_idx, best_sim))
        } else {
            None
        }
    }

    /// Voeg een nieuw Q&A paar toe
    pub fn insert(&mut self, question: String, answer: String, fingerprint: [f32; DIM], tick: u64) {
        // Als vol, verwijder het minst gebruikte paar
        if self.pairs.len() >= MAX_PAIRS {
            let min_idx = self.pairs.iter()
                .enumerate()
                .min_by_key(|(_, p)| p.hit_count)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.pairs.remove(min_idx);
        }

        self.pairs.push(QaPair {
            question,
            answer,
            fingerprint,
            hit_count: 0,
            created_tick: tick,
            field_point_id: None,
        });
    }

    /// Registreer een cache hit
    pub fn record_hit(&mut self, idx: usize) {
        if let Some(pair) = self.pairs.get_mut(idx) {
            pair.hit_count += 1;
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        tracing::info!(pairs = self.pairs.len(), "QA store saved");
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let store: Self = bincode::deserialize(&bytes)?;
        tracing::info!(pairs = store.pairs.len(), "QA store loaded");
        Ok(store)
    }
}
