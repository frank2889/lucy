//! # schema_v0 — Legacy v2.6.5-era Field layout
//!
//! Exacte kopie van Field-struct zoals die draaide in productie onder commit
//! `b89ab6e` (v2.6.5 + dopamine/ignition NaN-guard hotfix, pre-v2.6.6). Dat was
//! de binary die vanaf ±Apr 11 in de Docker image zat en pas op 2026-04-15 is
//! opgemerkt (we dachten jarenlang dat v2.6.6-v2.6.9 gedeployed waren — ze
//! waren alleen gecommitteerd).
//!
//! Verschil met [`FieldV1`] (v2.6.7-era):
//!   * Geen `last_boredom_tension`, `last_boredom_curiosity`, `boredom_streak`,
//!     `is_sluggish_flag` (v2.6.6 introduceerde die tussen `teacher_success_streak`
//!     en `attn_weights`).
//!   * Geen `concept_centroids` / `concept_assignments` (v2.6.7 depth-3
//!     concept-clustering — kwam na `wm_ages`).
//!
//! Omdat bincode positie-gebaseerd is, gedragen `#[serde(default)]` extra velden
//! zich als harde layout-verschillen (niet als "ontbrekende bytes worden default").
//! Een v2.6.5-snapshot via FieldV1 lezen faalt daarom op de eerste bool die op de
//! verkeerde byte-offset terechtkomt (`is_sluggish_flag` leest `0x40`).
//!
//! **NIET WIJZIGEN** — deze struct moet bit-precies matchen met v2.6.5 Field.
//! De migratie naar current `Field` gaat via `From<FieldV0> for FieldV1` en
//! vervolgens `From<FieldV1> for Field` in lib.rs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    DIM, FieldPoint, Economy, FieldPhase, Goal, AdaptiveParams,
    TemporalSnapshot, CausalEvent, GoalOutcome,
    default_phase, default_chaos_target, default_dim,
    default_emo_decay, default_error_tracking_rate, default_fe_tracking_rate,
    default_kl_scale, default_clamp_range, default_settle_epsilon,
    default_noise_amplitude, default_think_interval, default_reflect_interval,
    default_max_goals, default_learn_interval, default_explore_interval,
    default_consolidation_interval, default_snapshot_interval, default_growth_interval,
    default_sampling_temperature, default_sampling_top_p, default_repetition_penalty,
    default_teacher_interval,
};

/// Legacy v2.6.5 Field layout — **niet wijzigen**. Gebruik alleen voor backward-compat load.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldV0 {
    pub points: Vec<FieldPoint>,
    pub active_set: Vec<u32>,
    pub tick: u64,
    pub total_energy: f32,
    pub economy: Economy,

    #[serde(default = "default_phase")]
    pub phase: FieldPhase,

    #[serde(skip)]
    pub clamps: HashMap<u32, [f32; DIM]>,

    #[serde(default)]
    pub input_ids: Vec<u32>,
    #[serde(default)]
    pub output_ids: Vec<u32>,

    #[serde(skip)]
    pub last_thought: Option<[f32; DIM]>,
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
    pub current_intention: Option<[f32; DIM]>,
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

    // v2.6.6 velden (last_boredom_*, boredom_streak, is_sluggish_flag) ontbreken HIER.
    // v2.6.7 concept_centroids ontbreekt eveneens (komt pas na wm_ages).

    #[serde(default)]
    pub attn_weights: Option<Vec<f32>>,

    #[serde(skip)]
    pub attn_context: Vec<[f32; DIM]>,

    #[serde(skip)]
    pub working_memory: Vec<[f32; DIM]>,
    #[serde(skip)]
    pub wm_gates: Vec<f32>,
    #[serde(skip)]
    pub wm_ages: Vec<u32>,

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
}
