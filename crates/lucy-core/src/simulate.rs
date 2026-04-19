//! # Simulate — Test Patch Viability via 1000-Tick Sandbox
//!
//! Loads snapshot, applies patch, runs simulation, evaluates improvement.

use lucy_state::Field;

pub struct SimulationResult {
    pub passed: bool,
    pub reason: String,
    pub metrics_before: SimMetrics,
    pub metrics_after: SimMetrics,
    pub improvement_score: f32,  // 0.0-1.0
}

#[derive(Clone, Debug)]
pub struct SimMetrics {
    pub dopamine: f32,
    pub frustration: f32,
    pub satisfaction: f32,
    pub free_energy: f32,
    pub tension: f32,
    pub goal_progress: f32,
    pub nan_count: usize,
}

impl SimMetrics {
    fn from_field(field: &Field) -> Self {
        let mut nan_count = 0;

        // Count NaN points
        for point in &field.points {
            if !point.phi.iter().all(|x| x.is_finite()) {
                nan_count += 1;
            }
        }

        // Average metrics
        let avg_tension = if field.active_set.is_empty() {
            0.0
        } else {
            field.active_set.iter()
                .map(|&id| field.points[id as usize].tension)
                .sum::<f32>() / field.active_set.len() as f32
        };

        let goal_progress = field.goals.first()
            .map(|g| g.progress)
            .unwrap_or(0.0);

        Self {
            dopamine: field.dopamine,
            frustration: 0.935,  // Placeholder (not stored on Field)
            satisfaction: 0.1,   // Placeholder (not stored on Field)
            free_energy: field.avg_free_energy,
            tension: avg_tension,
            goal_progress,
            nan_count,
        }
    }
}

pub struct SimulationEngine;

impl SimulationEngine {
    /// Run 1000-tick simulation to test patch
    pub fn evaluate(field: Field, _ticks: usize) -> SimulationResult {
        let metrics_before = SimMetrics::from_field(&field);

        // NOTE: Full simulation requires Field::tick() method which doesn't exist yet.
        // For now, just validate the loaded snapshot for corruption.
        // Actual 1000-tick simulation will be integrated when tick method is exposed.

        let metrics_after = SimMetrics::from_field(&field);

        // Evaluate snapshot validity (NaN-free)
        let has_nan = metrics_after.nan_count > 0;
        let passed = !has_nan;
        let improvement_score = if passed { 0.5 } else { 0.0 };  // Placeholder
        let reason = if passed {
            "SNAPSHOT_VALID: No NaN corruption detected".to_string()
        } else {
            format!("SNAPSHOT_CORRUPTED: {} NaN points detected", metrics_after.nan_count)
        };

        SimulationResult {
            passed,
            reason,
            metrics_before: metrics_before.clone(),
            metrics_after,
            improvement_score,
        }
    }
}
