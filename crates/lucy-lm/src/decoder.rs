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

    /// Decodeer een GEGEVEN reeks veld-phi's naar tokens (greedy/argmax). Deterministisch.
    /// Eerlijk model: het VELD levert de phi's, de decoder benoemt elke phi. De oude
    /// `generate` mockte de volgende phi met ruis (verzon dus zijn eigen "gedachten") —
    /// dat is verwijderd; echte autoregressie hoort door het veld gedreven te worden.
    pub fn decode_greedy(&self, phis: &[[f32; D_PHI]]) -> Result<Vec<u32>> {
        if phis.is_empty() { return Ok(vec![]); }
        let seq = phis.len();
        let flat: Vec<f32> = phis.iter().flat_map(|p| p.iter().copied()).collect();
        let x = Tensor::from_vec(flat, (seq, D_PHI), &Device::Cpu)?;
        let logits = self.forward(&x)?; // [seq, VOCAB_SIZE]
        let ids = logits.argmax(candle_core::D::Minus1)?; // [seq]
        Ok(ids.to_vec1::<u32>()?)
    }

    /// Decodeer een gegeven reeks phi's met temperature + nucleus (top-p) sampling.
    /// Deterministisch gegeven `seed` (reproduceerbaar en testbaar).
    pub fn decode_sampled(
        &self,
        phis: &[[f32; D_PHI]],
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(phis.len());
        let mut s = seed;
        for phi in phis {
            let x = Tensor::from_vec(phi.to_vec(), (1, D_PHI), &Device::Cpu)?;
            let mut logits = self.forward(&x)?.squeeze(0)?; // [VOCAB_SIZE]
            if temperature > 1e-6 && (temperature - 1.0).abs() > 1e-6 {
                let t = Tensor::new(&[temperature], &Device::Cpu)?;
                logits = logits.broadcast_div(&t)?;
            }
            // LCG-advance zodat elk token een eigen (deterministische) trekking krijgt
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            out.push(sample_nucleus(&logits, top_p, s)?);
        }
        Ok(out)
    }
}

/// Top-p (nucleus) sampling: houd de kleinste kop-set tokens die samen kans `top_p` dekken,
/// en trek daaruit PROPORTIONEEL aan waarschijnlijkheid (de oude versie trok uniform binnen de
/// nucleus — dat is fout). Deterministisch gegeven `rand_u64`.
fn sample_nucleus(logits: &Tensor, top_p: f32, rand_u64: u64) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;
    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits_vec.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut nucleus: Vec<usize> = Vec::new();
    for &idx in &indices {
        cumsum += probs[idx];
        nucleus.push(idx);
        if cumsum >= top_p { break; }
    }
    if nucleus.is_empty() { return Ok(indices[0] as u32); }

    // Proportioneel trekken binnen de nucleus
    let mass: f32 = nucleus.iter().map(|&i| probs[i]).sum();
    let r = (rand_u64 as f64 / u64::MAX as f64) as f32 * mass;
    let mut acc = 0.0;
    for &idx in &nucleus {
        acc += probs[idx];
        if acc >= r { return Ok(idx as u32); }
    }
    Ok(*nucleus.last().unwrap() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use candle_core::DType;

    fn build() -> PhiDecoder {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        PhiDecoder::new(vb).unwrap()
    }

    #[test]
    fn forward_vorm_klopt() {
        let d = build();
        let x = Tensor::from_vec(vec![0.1f32; D_PHI], (1, D_PHI), &Device::Cpu).unwrap();
        let logits = d.forward(&x).unwrap();
        assert_eq!(logits.dims(), &[1, VOCAB_SIZE]);
    }

    #[test]
    fn decode_greedy_geldig_en_deterministisch() {
        let d = build();
        let phis = vec![[0.2f32; D_PHI], [-0.3f32; D_PHI]];
        let ids = d.decode_greedy(&phis).unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.iter().all(|&t| (t as usize) < VOCAB_SIZE));
        assert_eq!(ids, d.decode_greedy(&phis).unwrap());
    }
}
