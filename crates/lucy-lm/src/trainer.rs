//! # trainer — Traint de PhiDecoder (veld-phi -> taal) via cross-entropy.
//!
//! Zelf-getraind, geen externe modellen: het VELD denkt, de decoder leert het te
//! verwoorden. Dit is het fundament van Lucy's eigen taal-mond (Visie A).
//!
//! Status: NOG NIET gewired in lucy-core en NOG NIET getraind op echte data. Dat is
//! een volgende fase, gated op compute/host (candle-training is zwaar; de huidige
//! 1 GB-host kan het niet dragen). Hier staat het correcte, geteste fundament klaar.

use crate::decoder::{PhiDecoder, D_PHI};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{Optimizer, SGD};
use candle_nn::{VarBuilder, VarMap};

/// Houdt het model, zijn trainbare variabelen en een optimizer bij elkaar.
pub struct LmTrainer {
    pub model: PhiDecoder,
    varmap: VarMap,
    opt: SGD,
    device: Device,
    pub steps: u64,
}

impl LmTrainer {
    pub fn new(learning_rate: f64) -> Result<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PhiDecoder::new(vb)?;
        let opt = SGD::new(varmap.all_vars(), learning_rate)?;
        Ok(Self { model, varmap, opt, device, steps: 0 })
    }

    /// Eén trainingsstap: leer de decoder om per veld-phi het bijbehorende token te
    /// voorspellen. `phis`: [seq][D_PHI], `targets`: [seq] token-ids. Returnt de
    /// (mean) cross-entropy loss. Puur zelf-supervised op Lucy's eigen data.
    pub fn train_step(&mut self, phis: &[[f32; D_PHI]], targets: &[u32]) -> Result<f32> {
        assert_eq!(phis.len(), targets.len(), "phis en targets moeten even lang zijn");
        let seq = phis.len();
        let flat: Vec<f32> = phis.iter().flat_map(|p| p.iter().copied()).collect();
        let x = Tensor::from_vec(flat, (seq, D_PHI), &self.device)?;
        let logits = self.model.forward(&x)?; // [seq, VOCAB]
        let tgt = Tensor::from_vec(targets.to_vec(), (seq,), &self.device)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &tgt)?;
        self.opt.backward_step(&loss)?;
        self.steps += 1;
        loss.to_scalar::<f32>()
    }

    /// Bewaar de gewichten (safetensors) — persistentie van de taal-mond.
    pub fn save(&self, path: &str) -> Result<()> { self.varmap.save(path) }
    /// Laad gewichten terug in de bestaande variabelen.
    pub fn load(&mut self, path: &str) -> Result<()> { self.varmap.load(path) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traint_en_loss_daalt() -> Result<()> {
        // Bewijs dat het component echt LEERT: één vast (phi -> token 7)-paar,
        // loss moet dalen en greedy-decode moet na training het target geven.
        let mut tr = LmTrainer::new(0.5)?;
        let mut phi = [0.0f32; D_PHI];
        phi[0] = 1.0;
        let phis = vec![phi];
        let targets = vec![7u32];
        let first = tr.train_step(&phis, &targets)?;
        let mut last = first;
        for _ in 0..120 {
            last = tr.train_step(&phis, &targets)?;
        }
        assert!(last < first, "loss moet dalen: {} -> {}", first, last);
        assert_eq!(tr.steps, 121);
        assert_eq!(tr.model.decode_greedy(&phis)?[0], 7, "na training greedy == target");
        Ok(())
    }
}
