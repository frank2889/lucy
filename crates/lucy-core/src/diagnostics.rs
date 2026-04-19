//! # Diagnostics — Lucy's Self-Aware Bottleneck Detector
//!
//! Reads live telemetry, identifies anomalies, suggests self-modification patches.

use lucy_state::Field;
use serde::{Deserialize, Serialize};

/// Detected bottleneck — root cause + suggested fix
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BottleneckDiagnosis {
    pub anomaly: String,           // e.g., "dopamine_stuck"
    pub root_cause: String,        // e.g., "learning disabled, tension not dropping"
    pub suggested_patch: CodePatchSuggestion,
    pub severity: f32,             // 0.0-1.0 (1.0 = critical)
    pub confidence: f32,           // 0.0-1.0 in diagnosis accuracy
}

/// Suggested code patch to fix bottleneck
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodePatchSuggestion {
    pub file: String,              // e.g., "crates/lucy-core/src/compute.rs"
    pub old_pattern: String,
    pub new_pattern: String,
    pub reason: String,
    pub estimated_impact: String,  // e.g., "dopamine: 0.006 → 0.5"
}

/// Live telemetry snapshot (subset of /telemetry)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub tick: u64,
    pub dopamine: f32,
    pub frustration: f32,
    pub satisfaction: f32,
    pub curiosity: f32,
    pub free_energy: f32,
    pub tension: f32,
    pub ignition_active: bool,
    pub ignition_strength: f32,
    pub goal_progress: f32,
    pub n_goals: usize,
    pub reflect_interval: u64,
}

pub struct BottleneckDetector;

impl BottleneckDetector {
    /// Analyze field + telemetry for bottlenecks
    pub fn diagnose(field: &Field, telemetry: &TelemetrySnapshot) -> Vec<BottleneckDiagnosis> {
        let mut diagnoses = vec![];

        // Dopamine stuck: reward signal flatlined
        if telemetry.dopamine < 0.1 && telemetry.tick > 1000 {
            diagnoses.push(BottleneckDiagnosis {
                anomaly: "dopamine_stuck".to_string(),
                root_cause: "Reward signal flatlined → learning disabled. Tension not dropping.".to_string(),
                suggested_patch: CodePatchSuggestion {
                    file: "crates/lucy-core/src/compute.rs".to_string(),
                    old_pattern: "field.dopamine = nan_safe((field.dopamine + deltas[0]).clamp(0.0, 2.0));".to_string(),
                    new_pattern: "let base_reward = 0.1; field.dopamine = nan_safe((field.dopamine + deltas[0] + base_reward).clamp(0.0, 2.0));".to_string(),
                    reason: "Add baseline dopamine reward to prevent learning collapse".to_string(),
                    estimated_impact: "dopamine: ~0.006 → 0.1+ (restores reward signal)".to_string(),
                },
                severity: 0.95,
                confidence: 0.85,
            });
        }

        // Free energy exploding: numerical instability
        if telemetry.free_energy > 1e10 {
            diagnoses.push(BottleneckDiagnosis {
                anomaly: "free_energy_exploding".to_string(),
                root_cause: "KL divergence + blanket_surprise blowup. Clamp too high.".to_string(),
                suggested_patch: CodePatchSuggestion {
                    file: "crates/lucy-core/src/compute.rs".to_string(),
                    old_pattern: "raw_fe.clamp(0.0, f32::MAX / 1e6)".to_string(),
                    new_pattern: "raw_fe.clamp(0.0, 1e6)".to_string(),
                    reason: "Lower free energy ceiling: 3.4e31 → 1e6 prevents wind-up".to_string(),
                    estimated_impact: "free_energy: 2.76e31 → <1e6 (stable)".to_string(),
                },
                severity: 0.9,
                confidence: 0.9,
            });
        }

        // Goal progress stuck: age-based calculation too slow
        if telemetry.goal_progress < 0.1 && telemetry.goal_progress > 0.0 && telemetry.tick > telemetry.reflect_interval * 2 {
            diagnoses.push(BottleneckDiagnosis {
                anomaly: "goal_progress_stuck".to_string(),
                root_cause: "Age-based progress: age/(3×reflect_interval). Coefficient too conservative.".to_string(),
                suggested_patch: CodePatchSuggestion {
                    file: "crates/lucy-core/src/thinker.rs".to_string(),
                    old_pattern: "let age_progress = (age as f32 / (field.reflect_interval * 3) as f32).min(0.8);".to_string(),
                    new_pattern: "let age_progress = (age as f32 / (field.reflect_interval * 2) as f32).min(0.8);".to_string(),
                    reason: "Reduce age coefficient: 3× → 2× speeds up progress increment".to_string(),
                    estimated_impact: "goal_progress: 4% → 6-8% per reflect cycle (faster completion)".to_string(),
                },
                severity: 0.7,
                confidence: 0.8,
            });
        }

        // Frustration spike: insufficient goal diversity
        if telemetry.frustration > 0.9 && telemetry.satisfaction < 0.2 {
            diagnoses.push(BottleneckDiagnosis {
                anomaly: "frustration_spike".to_string(),
                root_cause: "Only 1 goal active. No diversity → stuck in single strategy.".to_string(),
                suggested_patch: CodePatchSuggestion {
                    file: "crates/lucy-core/src/thinker.rs".to_string(),
                    old_pattern: "field.goals.iter().find(|g| g.progress < 1.0)".to_string(),
                    new_pattern: "field.goals.iter().filter(|g| g.progress < 1.0).take(3).next()".to_string(),
                    reason: "Allow up to 3 parallel active goals instead of 1".to_string(),
                    estimated_impact: "frustration: 0.935 → 0.5 (parallelism diversity)".to_string(),
                },
                severity: 0.75,
                confidence: 0.7,
            });
        }

        diagnoses
    }

    /// Severity assessment
    pub fn triage(diagnoses: &[BottleneckDiagnosis]) -> String {
        let max_severity = diagnoses.iter().map(|d| d.severity).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0.0);

        match max_severity {
            s if s >= 0.9 => "CRITICAL".to_string(),
            s if s >= 0.7 => "HIGH".to_string(),
            s if s >= 0.5 => "MEDIUM".to_string(),
            _ => "LOW".to_string(),
        }
    }
}
