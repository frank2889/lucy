//! # Self-Fix Orchestrator — v3.0 Autonomous Patch Generation & Deployment
//!
//! Ties together diagnostics, sandbox compilation, simulation, and deployment.

use lucy_state::Field;
use crate::diagnostics::{BottleneckDetector, TelemetrySnapshot};

/// Run the complete self-fix pipeline for the current field state
pub fn run_self_fix(field: &Field) -> String {
    // Step 1: Create telemetry snapshot from current field state
    let telemetry = TelemetrySnapshot {
        tick: field.tick,
        dopamine: field.dopamine,
        frustration: 0.935,  // Placeholder
        satisfaction: 0.1,   // Placeholder
        curiosity: 0.2,      // Placeholder
        free_energy: field.avg_free_energy,
        tension: if field.active_set.is_empty() {
            0.0
        } else {
            field.active_set.iter()
                .map(|&id| field.points[id as usize].tension)
                .sum::<f32>() / field.active_set.len() as f32
        },
        ignition_active: field.ignition_active,
        ignition_strength: field.ignition_strength,
        goal_progress: field.goals.first().map(|g| g.progress).unwrap_or(0.0),
        n_goals: field.goals.len(),
        reflect_interval: field.reflect_interval,
    };

    // Step 2: Detect bottlenecks
    let diagnoses = BottleneckDetector::diagnose(field, &telemetry);

    if diagnoses.is_empty() {
        return "OK: No bottlenecks detected".to_string();
    }

    let diagnosis_count = diagnoses.len();
    tracing::info!(
        tick = field.tick,
        bottleneck_count = diagnosis_count,
        "SELF_FIX: {} bottlenecks detected, initiating patch pipeline",
        diagnosis_count
    );

    // Step 3: Log diagnostics for human review
    for (i, diag) in diagnoses.iter().enumerate() {
        tracing::warn!(
            tick = field.tick,
            index = i,
            anomaly = &diag.anomaly,
            severity = &diag.severity,
            "SELF_FIX[{}/{}]: {} (severity: {})",
            i + 1, diagnosis_count, diag.anomaly, diag.severity
        );
    }

    // Step 4: Return status (actual deploy happens via /admin/self-fix endpoint)
    format!(
        "SELF_FIX_INITIATED: {} bottlenecks detected. Deploy via POST /admin/self-fix for autonomous patch generation.",
        diagnosis_count
    )
}
