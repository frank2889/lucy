use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub const D_PHI: usize = 256;
pub const D_HIDDEN: usize = 512;
pub const VOCAB_SIZE: usize = 26_000;

/// Maps Lucy's veld phi-vectors (256-dim) → vocabulary logits (26K tokens).
pub struct PhiDecoder {
    fc1: candle_nn::Linear,
    norm: candle_nn::LayerNorm,
    fc2: candle_nn::Linear,
}

impl PhiDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(D_PHI, D_HIDDEN, vb.pp("fc1"))?,
            norm: candle_nn::layer_norm(D_HIDDEN, 1e-5, vb.pp("norm"))?,
            fc2: candle_nn::linear(D_HIDDEN, VOCAB_SIZE, vb.pp("fc2"))?,
        })
    }

    /// phi: [seq, D_PHI] → logits: [seq, VOCAB_SIZE]
    pub fn forward(&self, phi: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(phi)?;
        let h = h.gelu()?;
        let h = self.norm.forward(&h)?;
        self.fc2.forward(&h)
    }

    /// Genereer N tokens autoregressief gegeven output-regio phi's.
    pub fn generate(
        &self,
        output_phis: &[[f32; D_PHI]],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        let device = Device::Cpu;
        let mut tokens = Vec::new();
        let mut phi_history = output_phis.to_vec();

        for step in 0..max_tokens {
            // Laatste phi vector
            let latest_phi = phi_history.last().ok_or_else(|| {
                candle_core::Error::Msg("empty phi history".to_string())
            })?;

            // Convert to tensor [1, 256]
            let phi_tensor = Tensor::new(&[*latest_phi], &device)?.reshape((1, D_PHI))?;

            // Forward pass
            let logits = self.forward(&phi_tensor)?;
            let logits = logits.squeeze(0)?; // [VOCAB_SIZE]

            // Temperature scaling
            let logits = if (temperature - 1.0).abs() > 1e-6 {
                let temp_t = Tensor::new(&[temperature], &device)?;
                logits.broadcast_div(&temp_t)?
            } else {
                logits
            };

            // Top-p sampling
            let token = sample_nucleus(&logits, 0.9)?;
            tokens.push(token);

            // Simple mock: append small noise to phi for next iteration
            let mut next_phi = *latest_phi;
            for i in 0..D_PHI {
                next_phi[i] += ((step as f32) * 0.001) - 0.0005;
            }
            phi_history.push(next_phi);

            if phi_history.len() > 512 {
                phi_history.remove(0);
            }
        }

        Ok(tokens)
    }
}

/// Top-p (nucleus) sampling: keep top tokens that account for cumulative probability p.
fn sample_nucleus(logits: &Tensor, top_p: f32) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;

    // Softmax
    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits_vec.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

    // Sort by probability (descending)
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Cumulative probability
    let mut cumsum = 0.0;
    let mut valid_indices = vec![];
    for &idx in &indices {
        cumsum += probs[idx];
        valid_indices.push(idx);
        if cumsum >= top_p {
            break;
        }
    }

    // Sample uniformly from valid_indices
    if valid_indices.is_empty() {
        return Ok(indices[0] as u32);
    }
    let sampled_idx = valid_indices[(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as usize)
        % valid_indices.len()];
    Ok(sampled_idx as u32)
}
