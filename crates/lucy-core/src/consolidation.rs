//! # consolidation — Geheugenconsolidatie via Replay
//!
//! Geïnspireerd op hippocampale replay tijdens slaap bij zoogdieren.
//! Lucy slaat trainingsvoorbeelden op en herhaalt ze periodiek door
//! het veld, waardoor kennis wordt versterkt en catastrophic forgetting
//! wordt voorkomen.
//!
//! ## Hoe het werkt
//!
//! 1. Tijdens training (via `/feed` of autonome exploratie) worden
//!    (input_embeddings, target_embeddings) paren opgeslagen
//! 2. Elke 500 ticks wordt een willekeurig voorbeeld herhaald
//! 3. Herhaling gebruikt equilibrium propagation met mildere β (0.2 vs 0.3)
//!    zodat oude kennis wordt versterkt zonder nieuwe kennis te overschrijven
//!
//! ## Buffer Management
//!
//! - FIFO buffer van max 500 voorbeelden
//! - Oudste voorbeelden worden eerst verwijderd
//! - Embeddings worden opgeslagen als `Vec<Vec<f32>>` (i.p.v. `[f32; DIM]`)
//!   voor flexibele serialisatie met bincode
//!
//! ## Biologische Analogie
//!
//! Net als bij slaap in het menselijk brein worden ervaringen van de dag
//! herhaald en geconsolideerd in het langetermijngeheugen. Dit is essentieel
//! voor generalisatie: patronen die meerdere keren worden gezien krijgen
//! sterkere koppelingen in het veld.

use lucy_state::*;
use crate::compute;

/// Maximaal aantal opgeslagen replay-voorbeelden.
/// 500 × ~512 bytes ≈ 256KB geheugen.
const MAX_REPLAY_BUFFER: usize = 500;

/// Replay buffer — Lucy herhaalt eerder geleerde patronen.
/// Vergelijkbaar met hippocampale replay tijdens slaap.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReplayBuffer {
    /// Opgeslagen (input, target) embedding sequenties
    pub examples: Vec<ReplayExample>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReplayExample {
    pub input: Vec<Vec<f32>>,  // Sequentie van DIM-vectors
    pub target: Vec<Vec<f32>>, // Sequentie van DIM-vectors
}

impl ReplayBuffer {
    pub fn new() -> Self {
        Self { examples: Vec::new() }
    }

    /// Sla een training voorbeeld op voor replay
    pub fn store(&mut self, input: Vec<[f32; DIM]>, target: Vec<[f32; DIM]>) {
        if self.examples.len() >= MAX_REPLAY_BUFFER {
            self.examples.remove(0); // FIFO
        }
        self.examples.push(ReplayExample {
            input: input.into_iter().map(|v| v.to_vec()).collect(),
            target: target.into_iter().map(|v| v.to_vec()).collect(),
        });
    }

    /// Replay een random voorbeeld door het veld
    pub fn replay_one(&self, field: &mut Field) -> Option<f32> {
        if self.examples.is_empty() { return None; }

        use rand::Rng;
        let mut rng = rand::rng();
        let idx = rng.random_range(0..self.examples.len());
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

        let delta = compute::equilibrium_propagation(
            field,
            &input,
            &target,
            5,   // ticks per token
            50,  // settle ticks
            0.2, // mild EP beta voor consolidatie
        );

        Some(delta)
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Ok(bincode::deserialize(&bytes)?)
    }
}
