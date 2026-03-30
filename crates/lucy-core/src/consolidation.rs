//! # consolidation — Geheugenconsolidatie via Prioritized Replay (v0.9)
//!
//! Geïnspireerd op hippocampale replay tijdens slaap bij zoogdieren.
//! Lucy slaat trainingsvoorbeelden op en herhaalt ze periodiek door
//! het veld, waardoor kennis wordt versterkt en catastrophic forgetting
//! wordt voorkomen.
//!
//! v0.9: Prioritized replay — moeilijke voorbeelden worden vaker herhaald.
//! Buffer vergroot naar 2000 voorbeelden.

use lucy_state::*;
use crate::compute;

/// Maximaal aantal opgeslagen replay-voorbeelden.
const MAX_REPLAY_BUFFER: usize = 2000;

/// Replay buffer — Lucy herhaalt eerder geleerde patronen.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReplayBuffer {
    /// Opgeslagen (input, target) embedding sequenties
    pub examples: Vec<ReplayExample>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReplayExample {
    pub input: Vec<Vec<f32>>,  // Sequentie van DIM-vectors
    pub target: Vec<Vec<f32>>, // Sequentie van DIM-vectors
    /// Prioriteit: hoe moeilijk dit voorbeeld is (hoge EP delta = moeilijk = meer replay)
    #[serde(default)]
    pub priority: f32,
}

impl ReplayBuffer {
    pub fn new() -> Self {
        Self { examples: Vec::new() }
    }

    /// Sla een training voorbeeld op voor replay
    pub fn store(&mut self, input: Vec<[f32; DIM]>, target: Vec<[f32; DIM]>) {
        if self.examples.len() >= MAX_REPLAY_BUFFER {
            // Verwijder het voorbeeld met de laagste prioriteit
            if let Some(min_idx) = self.examples.iter().enumerate()
                .min_by(|a, b| a.1.priority.partial_cmp(&b.1.priority).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                self.examples.remove(min_idx);
            }
        }
        self.examples.push(ReplayExample {
            input: input.into_iter().map(|v| v.to_vec()).collect(),
            target: target.into_iter().map(|v| v.to_vec()).collect(),
            priority: 1.0, // nieuw voorbeeld begint met hoge prioriteit
        });
    }

    /// Replay een voorbeeld — prioritized sampling (moeilijke voorbeelden vaker)
    pub fn replay_one(&mut self, field: &mut Field) -> Option<f32> {
        if self.examples.is_empty() { return None; }

        use rand::Rng;
        let mut rng = rand::rng();

        // Prioritized sampling: kies proportioneel aan prioriteit
        let total_priority: f32 = self.examples.iter().map(|e| e.priority.max(0.1)).sum();
        let mut threshold = rng.random_range(0.0..total_priority);
        let mut idx = 0;
        for (i, ex) in self.examples.iter().enumerate() {
            threshold -= ex.priority.max(0.1);
            if threshold <= 0.0 {
                idx = i;
                break;
            }
        }

        let example = &self.examples[idx];

        // Converteer Vec<Vec<f32>> terug naar Vec<[f32; DIM]>
        let input: Vec<[f32; DIM]> = example.input.iter().map(|v| {
            let mut arr = [0.0f32; DIM];
            for (d, val) in v.iter().enumerate().take(DIM) {
                arr[d] = *val;
            }
            arr
        }).collect();

        let target: Vec<[f32; DIM]> = example.target.iter().map(|v| {
            let mut arr = [0.0f32; DIM];
            for (d, val) in v.iter().enumerate().take(DIM) {
                arr[d] = *val;
            }
            arr
        }).collect();

        // v0.9: contrastief EP voor replay (sterker leersignaal)
        let delta = compute::contrastive_ep(
            field,
            &input,
            &target,
            5,   // ticks per token
            50,  // settle ticks
            0.2, // mild EP beta voor consolidatie
        );

        // Update prioriteit: hoge delta = moeilijk = hogere prioriteit
        self.examples[idx].priority = delta.min(10.0);
        // Decay alle prioriteiten langzaam (voorkomt dat oude items altijd hoog blijven)
        for ex in &mut self.examples {
            ex.priority *= 0.999;
        }

        Some(delta)
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Atomic save: schrijf naar .tmp, dan rename
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let tmp = path.with_extension("tmp");
        let bytes = bincode::serialize(self)?;
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Ok(bincode::deserialize(&bytes)?)
    }
}
