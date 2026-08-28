pub mod decoder;
pub mod trainer;

pub use decoder::{PhiDecoder, D_PHI, D_HIDDEN, VOCAB_SIZE};
pub use trainer::LmTrainer;

/// Status van Lucy's decoder/LM component.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LmStatus {
    pub training_steps: u64,
    pub loss: f32,
    pub corpus_mb: f32,
    pub ready: bool,
}

impl Default for LmStatus {
    fn default() -> Self {
        Self {
            training_steps: 0,
            loss: f32::INFINITY,
            corpus_mb: 0.0,
            ready: false,
        }
    }
}
