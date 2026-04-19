//! # field_handle — Lock-Free Concurrency voor Lucy's Field
//!
//! v2.0: ArcSwap voor zero-lock snapshot reads. Geen RwLock meer.

use lucy_state::*;
use std::sync::Arc;
use tokio::sync::{mpsc, watch, oneshot};
use arc_swap::ArcSwap;

/// Commands van API endpoints naar de tick loop
pub enum FieldCommand {
    Train { text: String, respond: oneshot::Sender<TrainResult> },
    TrainPair { input: String, target: String, respond: oneshot::Sender<TrainResult> },
    FeedCode { filename: String, code: String, respond: oneshot::Sender<TrainResult> },
    Save { respond: oneshot::Sender<bool> },
    /// v2.6.8 P2.2: Ablation-window toggle voor concept-priors
    SetConceptPriors { enabled: bool, until_delta_ticks: u64 },
    /// v2.7.0: Veldgroei — voeg n nieuwe verborgen punten toe
    GrowHidden { n: usize, respond: oneshot::Sender<usize> },
    /// v3.0: Self-fix loop — detect bottlenecks, generate + test patches, deploy
    RunSelfFix { respond: oneshot::Sender<String> },
}

pub struct TrainResult {
    pub tokens_trained: usize,
    pub new_vocab_size: usize,
}

/// Live status — zero lock via watch channel
#[derive(Clone, Debug)]
pub struct FieldStatus {
    pub tick: u64,
    pub phase: String,
    pub avg_tension: f32,
    pub n_active: usize,
    pub n_points: usize,
    pub dopamine: f32,
    pub norepinephrine: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
}

impl Default for FieldStatus {
    fn default() -> Self {
        Self {
            tick: 0, phase: "Idle".to_string(), avg_tension: 0.0,
            n_active: 0, n_points: 0,
            dopamine: 0.0, norepinephrine: 0.0, serotonin: 0.0, acetylcholine: 0.0,
        }
    }
}

/// v2.0: ArcSwap-based handle — ZERO lock reads
#[derive(Clone)]
pub struct FieldHandle {
    /// ArcSwap snapshot — reads zijn INSTANT, nooit geblokkeerd
    pub snapshot: Arc<ArcSwap<Field>>,

    /// Command channel naar tick loop
    pub commands: mpsc::Sender<FieldCommand>,

    /// Live status via watch channel
    pub status_rx: watch::Receiver<FieldStatus>,
}

/// Exploration resultaat — van async task naar tick loop
pub struct ExploreResult {
    pub url: String,
    pub body: Result<Vec<u8>, String>,
    pub status: u16,
}

/// Weight delta — van training thread naar tick loop
pub struct WeightDelta {
    pub point_id: usize,
    pub coupling_idx: usize,
    pub delta: f32,
}

/// Maak status snapshot vanuit een Field reference
pub fn field_to_status(field: &Field) -> FieldStatus {
    let n = field.active_set.len() as f32;
    let avg_tension = if n > 0.0 {
        field.active_set.iter()
            .map(|&id| field.points[id as usize].tension)
            .sum::<f32>() / n
    } else { 0.0 };

    FieldStatus {
        tick: field.tick,
        phase: format!("{:?}", field.phase),
        avg_tension,
        n_active: field.active_set.len(),
        n_points: field.points.len(),
        dopamine: field.dopamine,
        norepinephrine: field.norepinephrine,
        serotonin: field.serotonin,
        acetylcholine: field.acetylcholine,
    }
}
