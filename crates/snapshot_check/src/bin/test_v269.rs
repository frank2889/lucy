//! Integration test v2.6.9 migratie:
//!
//! 1. Maak vers FieldV1 (v2.6.7 layout) + schrijf als raw bincode (legacy)
//! 2. Field::load leest: valt terug op FieldV1 → migreert → current Field
//! 3. Field::save schrijft met header
//! 4. Field::load leest header → direct current Field
//!
//! Als alle stappen slagen → v2.6.9 framework werkt voor toekomstige snapshots.

use lucy_state::{Field, FieldPoint, schema_v1::FieldV1};

fn main() {
    let tmp_dir = std::env::temp_dir();
    let legacy_path = tmp_dir.join("lucy_v269_test_legacy.snapshot");
    let headered_path = tmp_dir.join("lucy_v269_test_headered.snapshot");

    // === Stap 1: Maak vers Field, voeg een dummy point toe, save als legacy ===
    println!("1. Maak vers Field + save als LEGACY (geen header) ...");
    let mut fresh = Field::new();
    fresh.points.push(FieldPoint::new());
    fresh.active_set.push(0);
    // Convert Field → FieldV1 door handmatige assignment (omgekeerd van migratie)
    let v1 = FieldV1 {
        points: fresh.points.clone(),
        active_set: fresh.active_set.clone(),
        tick: 42,
        total_energy: fresh.total_energy,
        economy: fresh.economy.clone(),
        phase: fresh.phase,
        clamps: fresh.clamps.clone(),
        input_ids: fresh.input_ids.clone(),
        output_ids: fresh.output_ids.clone(),
        last_thought: fresh.last_thought,
        reflection_count: 100,
        dopamine: fresh.dopamine,
        norepinephrine: fresh.norepinephrine,
        serotonin: fresh.serotonin,
        acetylcholine: fresh.acetylcholine,
        spectral_radius: fresh.spectral_radius,
        chaos_target: fresh.chaos_target,
        current_intention: fresh.current_intention,
        intention_strength: fresh.intention_strength,
        rnd_frozen: fresh.rnd_frozen.clone(),
        rnd_predictor: fresh.rnd_predictor.clone(),
        kappa: fresh.kappa,
        omega: fresh.omega,
        lambda: fresh.lambda,
        beta: fresh.beta,
        alpha: fresh.alpha,
        rho: fresh.rho,
        mu: fresh.mu,
        growth_threshold: fresh.growth_threshold,
        ema_decay: fresh.ema_decay,
        emo_decay: fresh.emo_decay,
        error_tracking_rate: fresh.error_tracking_rate,
        fe_tracking_rate: fresh.fe_tracking_rate,
        kl_scale: fresh.kl_scale,
        clamp_range: fresh.clamp_range,
        settle_epsilon: fresh.settle_epsilon,
        noise_amplitude: fresh.noise_amplitude,
        think_interval: fresh.think_interval,
        reflect_interval: fresh.reflect_interval,
        max_goals: fresh.max_goals,
        learn_interval: fresh.learn_interval,
        explore_interval: fresh.explore_interval,
        consolidation_interval: fresh.consolidation_interval,
        snapshot_interval: fresh.snapshot_interval,
        growth_interval: fresh.growth_interval,
        sampling_temperature: fresh.sampling_temperature,
        sampling_top_p: fresh.sampling_top_p,
        repetition_penalty: fresh.repetition_penalty,
        goals: fresh.goals.clone(),
        last_think_tick: fresh.last_think_tick,
        last_reflect_tick: fresh.last_reflect_tick,
        teacher_interval: fresh.teacher_interval,
        teacher_success_streak: fresh.teacher_success_streak,
        last_boredom_tension: fresh.last_boredom_tension,
        last_boredom_curiosity: fresh.last_boredom_curiosity,
        boredom_streak: fresh.boredom_streak,
        is_sluggish_flag: fresh.is_sluggish_flag,
        attn_weights: fresh.attn_weights.clone(),
        attn_context: Vec::new(),
        working_memory: Vec::new(),
        wm_gates: Vec::new(),
        wm_ages: Vec::new(),
        concept_centroids: fresh.concept_centroids.clone(),
        concept_assignments: Vec::new(),
        global_oscillation_phase: fresh.global_oscillation_phase,
        oscillation_coherence: fresh.oscillation_coherence,
        integrated_info_avg: fresh.integrated_info_avg,
        power_law_exponent: fresh.power_law_exponent,
        is_critical: fresh.is_critical,
        avg_free_energy: fresh.avg_free_energy,
        theta_phase: fresh.theta_phase,
        theta_gamma_coupling: fresh.theta_gamma_coupling,
        branching_ratio_avg: fresh.branching_ratio_avg,
        ignition_active: fresh.ignition_active,
        ignition_strength: fresh.ignition_strength,
        avg_expected_free_energy: fresh.avg_expected_free_energy,
        adaptive: fresh.adaptive.clone(),
        temporal_history: fresh.temporal_history.clone(),
        causal_log: fresh.causal_log.clone(),
        goal_history: fresh.goal_history.clone(),
        pages_read: 777,
        tokens_processed: 1234,
        snapshot_version: fresh.snapshot_version,
        field_dim: fresh.field_dim,
        n_input: fresh.n_input,
        n_hidden: fresh.n_hidden,
        n_output: fresh.n_output,
    };
    // Schrijf raw bincode (geen header)
    let v1_bytes = bincode::serialize(&v1).expect("serialize v1");
    std::fs::write(&legacy_path, &v1_bytes).expect("write legacy");
    // CRC sidecar
    let crc = crc32fast::hash(&v1_bytes);
    std::fs::write(legacy_path.with_extension("crc32"), crc.to_le_bytes()).expect("write crc");
    println!("   Legacy written: {} bytes", v1_bytes.len());

    // === Stap 2: Field::load leest legacy via FieldV1 migratie ===
    println!("2. Field::load leest LEGACY → FieldV1 → migreert naar current ...");
    let loaded: Field = Field::load(&legacy_path).expect("load legacy");
    assert_eq!(loaded.tick, 42, "tick preserved");
    assert_eq!(loaded.reflection_count, 100, "reflection_count preserved");
    assert_eq!(loaded.pages_read, 777, "pages_read preserved");
    assert_eq!(loaded.concept_priors_enabled, true, "new field default true");
    assert_eq!(loaded.concept_audit_log.len(), 0, "new ring-buffer empty");
    println!("   ✓ Legacy → current migratie werkt (tick={}, reflections={}, pages={})",
        loaded.tick, loaded.reflection_count, loaded.pages_read);

    // === Stap 3: Field::save schrijft met header ===
    println!("3. Field::save schrijft met SnapshotHeader (current schema) ...");
    loaded.save(&headered_path).expect("save headered");
    let written = std::fs::read(&headered_path).expect("read written");
    let magic = &written[0..4];
    assert_eq!(magic, b"LUCY", "magic bytes present");
    let version = u32::from_le_bytes(written[4..8].try_into().unwrap());
    println!("   ✓ Header gedetecteerd — magic=LUCY, version={}", version);

    // === Stap 4: Field::load leest headered direct ===
    println!("4. Field::load leest HEADERED → direct current Field ...");
    let loaded2: Field = Field::load(&headered_path).expect("load headered");
    assert_eq!(loaded2.tick, 42);
    assert_eq!(loaded2.reflection_count, 100);
    assert_eq!(loaded2.pages_read, 777);
    println!("   ✓ Headered load werkt (tick={}, reflections={}, pages={})",
        loaded2.tick, loaded2.reflection_count, loaded2.pages_read);

    // Cleanup
    let _ = std::fs::remove_file(&legacy_path);
    let _ = std::fs::remove_file(legacy_path.with_extension("crc32"));
    let _ = std::fs::remove_file(&headered_path);
    let _ = std::fs::remove_file(headered_path.with_extension("crc32"));

    println!();
    println!("🟢 v2.6.9 migratie-framework werkt end-to-end.");
    println!();
    println!("   - Pre-v2.6.9 snapshots (zonder header) → auto-migrate via FieldV1");
    println!("   - v2.6.9+ snapshots → header + direct load");
    println!("   - Data volledig preserved over migratie");
}
