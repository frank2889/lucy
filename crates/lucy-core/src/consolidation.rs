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

// Dynamic: buffer size is now derived from vocab size (was: hardcoded 2000).
// Default fallback when no vocab context available.
const DEFAULT_REPLAY_BUFFER: usize = 2000;

/// Replay buffer — Lucy herhaalt eerder geleerde patronen.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReplayBuffer {
    /// Opgeslagen (input, target) embedding sequenties
    pub examples: Vec<ReplayExample>,
    /// Dynamic buffer capacity — derived from vocab size or field params
    #[serde(default = "default_buffer_cap")]
    pub buffer_capacity: usize,
}

fn default_buffer_cap() -> usize { DEFAULT_REPLAY_BUFFER }

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
        Self {
            examples: Vec::new(),
            buffer_capacity: DEFAULT_REPLAY_BUFFER,
        }
    }

    /// Create with dynamic capacity derived from vocab size.
    /// Rule: buffer_capacity = vocab_size.clamp(500, 10_000)
    pub fn with_capacity(vocab_size: usize) -> Self {
        Self {
            examples: Vec::new(),
            // Dynamic: scale replay buffer with vocabulary richness
            buffer_capacity: vocab_size.clamp(500, 10_000),
        }
    }

    /// Sla een training voorbeeld op voor replay
    pub fn store(&mut self, input: Vec<[f32; DIM]>, target: Vec<[f32; DIM]>) {
        if self.examples.len() >= self.buffer_capacity {
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

        // Dynamic: priority floor uses field.error_tracking_rate (was: hardcoded 0.1)
        let priority_floor = field.error_tracking_rate;

        // Prioritized sampling: kies proportioneel aan prioriteit
        let total_priority: f32 = self.examples.iter().map(|e| e.priority.max(priority_floor)).sum();
        let mut threshold = rng.random_range(0.0..total_priority);
        let mut idx = 0;
        for (i, ex) in self.examples.iter().enumerate() {
            threshold -= ex.priority.max(priority_floor);
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
        // Dynamic: settle convergence uses field.settle_epsilon (was: hardcoded params 5, 50, 0.2)
        let delta = compute::contrastive_ep(
            field,
            &input,
            &target,
            5,   // ticks per token (structural)
            50,  // settle ticks (structural)
            0.2, // mild EP beta voor consolidatie
        );

        // Dynamic: priority clamp uses field.clamp_range (was: hardcoded 10.0)
        self.examples[idx].priority = delta.min(field.clamp_range);
        // Dynamic: priority decay = 1.0 - 1.0/buffer_size (was: hardcoded 0.999)
        // Larger buffers decay slower, smaller buffers forget faster
        let priority_decay = 1.0 - 1.0 / self.buffer_capacity as f32;
        for ex in &mut self.examples {
            ex.priority *= priority_decay;
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
        // CRC32 checksum + backup rotation
        let checksum = crc32fast::hash(&bytes);
        let _ = std::fs::write(path.with_extension("crc32"), checksum.to_le_bytes());
        lucy_io::fs::rotate_backups(path, 5);
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let (bytes, _src) = lucy_io::fs::load_with_fallback(path).ok_or("no primary or backup found")?;
        Ok(bincode::deserialize(&bytes)?)
    }
}
