//! # Feedback Loop — Post-Deploy Monitoring & Auto-Rollback
//!
//! Compares pre/post deploy telemetry to verify patch effectiveness.

use serde::{Deserialize, Serialize};
use crate::diagnostics::{CodePatchSuggestion, TelemetrySnapshot};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeployFeedback {
    pub patch: CodePatchSuggestion,
    pub snapshot_before: TelemetrySnapshot,
    pub snapshot_after: TelemetrySnapshot,
    pub success: bool,
    pub improvement_score: f32,  // -1.0 to +1.0
    pub reason: String,
}

pub struct FeedbackAnalyzer;

impl FeedbackAnalyzer {
    /// Compare pre/post telemetry to determine if patch worked
    pub fn evaluate(
        patch: &CodePatchSuggestion,
        before: &TelemetrySnapshot,
        after: &TelemetrySnapshot,
    ) -> DeployFeedback {
        // Calculate improvement deltas
        let dopamine_delta = after.dopamine - before.dopamine;
        let dopamine_ratio = if before.dopamine > 0.01 {
            dopamine_delta / before.dopamine
        } else {
            dopamine_delta / 0.01  // Avoid division by zero
        };

        let frustration_delta = before.frustration - after.frustration;  // Negative is good
        let frustration_ratio = if before.frustration > 0.1 {
            frustration_delta / before.frustration
        } else {
            frustration_delta / 0.1
        };

        let satisfaction_delta = after.satisfaction - before.satisfaction;
        let satisfaction_ratio = if before.satisfaction > 0.01 {
            satisfaction_delta / before.satisfaction
        } else {
            satisfaction_delta / 0.01
        };

        // Free energy: lower is better
        let free_energy_ratio = if before.free_energy > 1.0 {
            (before.free_energy - after.free_energy) / before.free_energy
        } else {
            0.0
        };

        // Composite improvement score (weighted)
        let improvement_score = (
            dopamine_ratio * 0.4 +              // Dopamine: critical
            frustration_ratio * 0.3 +          // Frustration: important
            satisfaction_ratio * 0.2 +
            free_energy_ratio.min(0.2) * 0.1   // Free energy: cap at 0.2 weight
        ).clamp(-1.0, 1.0);

        // Success threshold
        let success = improvement_score > 0.15;  // Need at least 15% improvement

        let reason = if success {
            format!(
                "PATCH SUCCESS: dopamine +{:.1}%, frustration -{:.1}%, score {:.2}",
                dopamine_ratio * 100.0,
                frustration_ratio * 100.0,
                improvement_score
            )
        } else {
            format!(
                "PATCH FAILED: dopamine {}{:.1}%, frustration {}{:.1}%, score {:.2} (<0.15 threshold)",
                if dopamine_delta >= 0.0 { "+" } else { "" },
                dopamine_ratio * 100.0,
                if frustration_delta >= 0.0 { "+" } else { "" },
                frustration_ratio * 100.0,
                improvement_score
            )
        };

        DeployFeedback {
            patch: patch.clone(),
            snapshot_before: before.clone(),
            snapshot_after: after.clone(),
            success,
            improvement_score,
            reason,
        }
    }

    /// Confidence scoring: how much to trust patch reliability for future use
    pub fn confidence_update(current: f32, feedback: &DeployFeedback) -> f32 {
        if feedback.success {
            (current + 0.1).min(1.0)  // Increase confidence
        } else {
            (current - 0.3).max(0.0)  // Drop confidence significantly
        }
    }

    /// Blocklist management: prevent repeatedly failed patches
    pub fn should_block_patch(confidence: f32, failure_count: usize) -> bool {
        confidence < 0.2 || failure_count >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_improvement_detection() {
        let patch = CodePatchSuggestion {
            file: "test.rs".to_string(),
            old_pattern: "x".to_string(),
            new_pattern: "y".to_string(),
            reason: "test".to_string(),
            estimated_impact: "test".to_string(),
        };

        let before = TelemetrySnapshot {
            tick: 500000,
            dopamine: 0.006,
            frustration: 0.935,
            satisfaction: 0.1,
            curiosity: 0.1,
            free_energy: 2.76e31,
            tension: 1.7e29,
            ignition_active: true,
            ignition_strength: 500.0,
            goal_progress: 0.04,
            n_goals: 1,
            reflect_interval: 10000,
        };

        let after = TelemetrySnapshot {
            tick: 500100,
            dopamine: 0.3,           // +50× improvement
            frustration: 0.5,        // -46% improvement
            satisfaction: 0.4,       // +300%
            curiosity: 0.3,
            free_energy: 1.5e6,      // Down from 2.76e31
            tension: 1.2e4,          // Down from 1.7e29
            ignition_active: true,
            ignition_strength: 480.0,
            goal_progress: 0.08,
            n_goals: 2,
            reflect_interval: 10000,
        };

        let feedback = FeedbackAnalyzer::evaluate(&patch, &before, &after);
        assert!(feedback.success, "High improvement should be detected");
        assert!(feedback.improvement_score > 0.5, "Score should reflect improvement");
    }
}
