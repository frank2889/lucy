//! # schema_v2 — Frozen Field+FieldPoint layout at DIM=256 (v2.9.0-pre-upgrade)
//!
//! Snapshot du Field-struct & FieldPoint comme ils existent juste avant DIM upgrade vers 512.
//! Utilisé pour lire des snapshots v2.8.x (DIM=256) et les migrer vers v2.9.0+ (DIM=512).
//!
//! **NE PAS MODIFIER** — ces structs doivent correspondre exactement à Field/FieldPoint à la
//! version avant v2.9.0. Toute différence = charge corrompue et perte de données.
//!
//! La migration vers current Field avec DIM=512 se fait via `From<FieldV2> for Field`
//! dans lib.rs (voir ci-dessous).

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use smallvec::SmallVec;
use std::collections::HashMap;

use crate::{
    Economy, FieldPhase, Goal, AdaptiveParams, Coupling,
    TemporalSnapshot, CausalEvent, GoalOutcome, ConceptAudit, SelfEpAudit, SelfMetrics,
    default_phase, default_chaos_target, default_dim,
    default_emo_decay, default_error_tracking_rate, default_fe_tracking_rate,
    default_kl_scale, default_clamp_range, default_settle_epsilon,
    default_noise_amplitude, default_think_interval, default_reflect_interval,
    default_max_goals, default_learn_interval, default_explore_interval,
    default_consolidation_interval, default_snapshot_interval, default_growth_interval,
    default_sampling_temperature, default_sampling_top_p, default_repetition_penalty,
    default_teacher_interval, default_true, default_precision, default_control_belief,
};

const DIM_V2: usize = 256;

/// FieldPoint locked at DIM_V2=256 for v2.8.x snapshots
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldPointV2 {
    #[serde(with = "BigArray")]
    pub phi: [f32; DIM_V2],
    #[serde(with = "BigArray")]
    pub prediction: [f32; DIM_V2],
    #[serde(with = "BigArray")]
    pub memory: [f32; DIM_V2],
    pub couplings: SmallVec<[Coupling; 16]>,
    pub eta: f32,
    pub energy: f32,
    pub tension: f32,
    pub tension_ema: f32,
    pub curiosity: f32,
    pub satisfaction: f32,
    pub frustration: f32,
    #[serde(default)]
    pub stuck_ticks: u16,
    pub flags: u8,
    #[serde(default)]
    pub depth: u8,
    #[serde(default)]
    pub rnd_score: f32,
    #[serde(default = "default_precision")]
    pub error_var: f32,
    #[serde(default = "default_precision")]
    pub precision: f32,
    #[serde(default)]
    pub kwta_active: bool,
    #[serde(default)]
    pub phi_integrated: f32,
    #[serde(default)]
    pub oscillation_phase: f32,
    #[serde(default)]
    pub free_energy: f32,
    #[serde(default)]
    pub kl_divergence: f32,
    #[serde(default)]
    pub oscillation_phase_theta: f32,
    #[serde(default = "default_control_belief")]
    pub control_belief: f32,
    #[serde(default)]
    pub reward_quantiles: [f32; 5],
    #[serde(default)]
    pub causal_power: f32,
    #[serde(default)]
    pub expected_free_energy: f32,
    #[serde(default)]
    pub branching_ratio: f32,
    #[serde(default)]
    pub free_energy_rate: f32,
    #[serde(default)]
    pub blanket_surprise: f32,
    #[serde(default)]
    pub irreducible_free_energy: f32,
}

/// Pre-v2.9.0 Field layout — frozen at DIM=256. Migration target pour 256→512 upgrade.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldV2 {
    pub points: Vec<FieldPointV2>,
    pub active_set: Vec<u32>,
    pub tick: u64,
    pub total_energy: f32,
    pub economy: Economy,

    #[serde(default = "default_phase")]
    pub phase: FieldPhase,

    #[serde(skip)]
    pub clamps: HashMap<u32, [f32; DIM_V2]>,

    #[serde(default)]
    pub input_ids: Vec<u32>,
    #[serde(default)]
    pub output_ids: Vec<u32>,

    #[serde(skip)]
    pub last_thought: Option<[f32; DIM_V2]>,
    #[serde(default)]
    pub reflection_count: u64,

    #[serde(default)]
    pub dopamine: f32,
    #[serde(default)]
    pub norepinephrine: f32,
    #[serde(default)]
    pub serotonin: f32,
    #[serde(default)]
    pub acetylcholine: f32,

    #[serde(default)]
    pub spectral_radius: f32,
    #[serde(default = "default_chaos_target")]
    pub chaos_target: f32,

    #[serde(skip)]
    pub current_intention: Option<[f32; DIM_V2]>,
    #[serde(default)]
    pub intention_strength: f32,

    #[serde(skip)]
    pub rnd_frozen: Option<Vec<f32>>,
    #[serde(skip)]
    pub rnd_predictor: Option<Vec<f32>>,

    pub kappa: f32,
    pub omega: f32,
    pub lambda: f32,
    pub beta: f32,
    pub alpha: f32,
    pub rho: f32,
    pub mu: f32,
    pub growth_threshold: f32,
    pub ema_decay: f32,

    #[serde(default = "default_emo_decay")]
    pub emo_decay: f32,
    #[serde(default = "default_error_tracking_rate")]
    pub error_tracking_rate: f32,
    #[serde(default = "default_fe_tracking_rate")]
    pub fe_tracking_rate: f32,
    #[serde(default = "default_kl_scale")]
    pub kl_scale: f32,
    #[serde(default = "default_clamp_range")]
    pub clamp_range: f32,
    #[serde(default = "default_settle_epsilon")]
    pub settle_epsilon: f32,
    #[serde(default = "default_noise_amplitude")]
    pub noise_amplitude: f32,
    #[serde(default = "default_think_interval")]
    pub think_interval: u64,
    #[serde(default = "default_reflect_interval")]
    pub reflect_interval: u64,
    #[serde(default = "default_max_goals")]
    pub max_goals: usize,
    #[serde(default = "default_learn_interval")]
    pub learn_interval: u64,
    #[serde(default = "default_explore_interval")]
    pub explore_interval: u64,
    #[serde(default = "default_consolidation_interval")]
    pub consolidation_interval: u64,
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval: u64,
    #[serde(default = "default_growth_interval")]
    pub growth_interval: u64,
    #[serde(default = "default_sampling_temperature")]
    pub sampling_temperature: f32,
    #[serde(default = "default_sampling_top_p")]
    pub sampling_top_p: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    #[serde(default)]
    pub goals: Vec<Goal>,
    #[serde(default)]
    pub last_think_tick: u64,
    #[serde(default)]
    pub last_reflect_tick: u64,

    #[serde(default = "default_teacher_interval")]
    pub teacher_interval: u64,
    #[serde(default)]
    pub teacher_success_streak: u32,

    #[serde(default)]
    pub last_boredom_tension: f32,
    #[serde(default)]
    pub last_boredom_curiosity: f32,
    #[serde(default)]
    pub boredom_streak: u32,
    #[serde(default)]
    pub is_sluggish_flag: bool,

    #[serde(default)]
    pub attn_weights: Option<Vec<f32>>,

    #[serde(skip)]
    pub attn_context: Vec<[f32; DIM_V2]>,

    #[serde(skip)]
    pub working_memory: Vec<[f32; DIM_V2]>,
    #[serde(skip)]
    pub wm_gates: Vec<f32>,
    #[serde(skip)]
    pub wm_ages: Vec<u32>,

    #[serde(default)]
    pub concept_centroids: Vec<Vec<f32>>,

    #[serde(skip)]
    pub concept_assignments: Vec<u16>,

    #[serde(default)]
    pub global_oscillation_phase: f32,
    #[serde(default)]
    pub oscillation_coherence: f32,
    #[serde(default)]
    pub integrated_info_avg: f32,
    #[serde(default)]
    pub power_law_exponent: f32,
    #[serde(default)]
    pub is_critical: bool,
    #[serde(default)]
    pub avg_free_energy: f32,

    #[serde(default)]
    pub theta_phase: f32,
    #[serde(default)]
    pub theta_gamma_coupling: f32,
    #[serde(default)]
    pub branching_ratio_avg: f32,
    #[serde(default)]
    pub ignition_active: bool,
    #[serde(default)]
    pub ignition_strength: f32,

    #[serde(default)]
    pub last_winner_id: u32,
    #[serde(default)]
    pub winner_refractory_until: u64,

    #[serde(default)]
    pub avg_expected_free_energy: f32,

    #[serde(default)]
    pub adaptive: AdaptiveParams,

    #[serde(default)]
    pub temporal_history: Vec<TemporalSnapshot>,
    #[serde(default)]
    pub causal_log: Vec<CausalEvent>,
    #[serde(default)]
    pub goal_history: Vec<GoalOutcome>,

    #[serde(default)]
    pub pages_read: u64,
    #[serde(default)]
    pub tokens_processed: u64,

    #[serde(default)]
    pub snapshot_version: u32,
    #[serde(default = "default_dim")]
    pub field_dim: usize,
    #[serde(default)]
    pub n_input: usize,
    #[serde(default)]
    pub n_hidden: usize,
    #[serde(default)]
    pub n_output: usize,

    #[serde(skip, default = "default_true")]
    pub concept_priors_enabled: bool,
    #[serde(skip)]
    pub concept_priors_disabled_until_tick: u64,
    #[serde(skip)]
    pub concept_audit_log: Vec<ConceptAudit>,
    #[serde(skip)]
    pub self_ep_audit: Vec<SelfEpAudit>,
    #[serde(skip)]
    pub self_metrics: SelfMetrics,
    #[serde(skip)]
    pub provenance_counts: [u64; 6],
}
