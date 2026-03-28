//! # compute — Lucy's Velddynamica en Leeralgoritme
//!
//! Dit is de wiskundige kern van Lucy. Alle berekeningen die het veld
//! "laten denken" en "laten leren" staan hier.
//!
//! ## Velddynamica (per tick)
//!
//! Elke tick berekent voor elk actief punt:
//!
//! 1. **Φ update**: `Φ_{t+1} = Φ_t + λ·[-∇spanning] + β·input - α·sign(Φ)`
//!    - λ = veld-update schaalfactor (hoe snel het veld beweegt)
//!    - ∇spanning = gewogen verschil met buren (lokale gradient)
//!    - β = input injectie sterkte
//!    - α = sparse penalty (L1 regularisatie, houdt het veld schoon)
//!
//! 2. **M update**: `M_{t+1} = ρ·M_t + (1-ρ)·ε_t` (exponentieel geheugen)
//!
//! 3. **Spanning**: `τ = ||ε||² + κ·compute_kosten - ω·empowerment`
//!    - ε = Φ - P (prediction error — het leersignaal)
//!    - κ = kosten per koppeling (motiveert efficiëntie)
//!    - ω = empowerment beloning (motiveert sterke verbindingen)
//!
//! ## Equilibrium Propagation (het leeralgoritme)
//!
//! Lucy leert NIET via backprop maar via EP (Scellier & Bengio, 2017):
//!
//! 1. **Vrije fase**: input geklemd, veld settlet vrij naar laag-energie toestand
//! 2. **Geklemde fase**: input + output geklemd, veld settlet naar target
//! 3. **Gewicht update**: `ΔW = η·β·(Φ_clamped·Φ_neighbor_clamped - Φ_free·Φ_neighbor_free)`
//!
//! Dit is biologisch plausibel: elk punt gebruikt alleen lokale informatie.
//! Geen globale foutpropagatie, geen berekening van gradiënten over het hele netwerk.
//!
//! ## Settle Loop
//!
//! Bij een query of training: het veld wordt herhaaldelijk geüpdatet tot
//! de gemiddelde spanning onder een drempel zakt (convergentie) of het
//! maximum aantal ticks bereikt is. Dit "settlen" IS het nadenken.
//!
//! ## Clamping
//!
//! Punten kunnen "geklemd" worden: hun Φ wordt geforceerd op een externe waarde.
//! Dit is hoe input (tekst-embeddings) en targets (gewenste output) worden geïnjecteerd.

use lucy_state::*;

/// Bereken prediction error: ε = Φ - P
#[inline]
pub fn prediction_error(point: &FieldPoint) -> [f32; DIM] {
    let mut eps = [0.0f32; DIM];
    for d in 0..DIM {
        eps[d] = point.phi[d] - point.prediction[d];
    }
    eps
}

/// Bereken ||ε||²
#[inline]
pub fn squared_norm(v: &[f32; DIM]) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..DIM {
        sum += v[d] * v[d];
    }
    sum
}

/// Update Φ voor een actief punt:
/// Φ_{t+1} = Φ_t + λ·[-lokale_spanning_gradient] + β·I - α·sign(Φ)
/// ALS het punt geklemd is → forceer phi = clamp waarde
pub fn update_phi(field: &mut Field, id: u32, input: Option<&[f32; DIM]>) {
    // Check clamp EERST — geklemde punten worden geforceerd
    if let Some(clamp) = field.clamps.get(&id) {
        let point = &mut field.points[id as usize];
        point.prediction = point.phi;
        for d in 0..DIM.min(clamp.len()) {
            point.phi[d] = clamp[d];
        }
        return;
    }

    let lambda = field.lambda;
    let beta = field.beta;
    let alpha = field.alpha;

    // Bereken lokale spanning gradient vanuit buren
    let mut grad = [0.0f32; DIM];
    let point = &field.points[id as usize];
    let n_couplings = point.couplings.len();

    if n_couplings > 0 {
        let couplings: Vec<(u32, f32)> = point
            .couplings
            .iter()
            .map(|c| (c.target, c.weight))
            .collect();

        let phi_self = point.phi;

        for (target, weight) in &couplings {
            if let Some(neighbor) = field.points.get(*target as usize) {
                for d in 0..DIM {
                    let diff = phi_self[d] - neighbor.phi[d];
                    grad[d] += weight * diff;
                }
            }
        }

        let inv_n = 1.0 / n_couplings as f32;
        for d in 0..DIM {
            grad[d] *= inv_n;
        }
    }

    // Update Φ
    let point = &mut field.points[id as usize];
    for d in 0..DIM {
        point.prediction[d] = point.phi[d];
        point.phi[d] -= lambda * grad[d];

        if let Some(inp) = input {
            point.phi[d] += beta * inp[d];
        }

        // Sparse penalty: L1
        point.phi[d] -= alpha * point.phi[d].signum();

        point.phi[d] = point.phi[d].clamp(-10.0, 10.0);
    }
}

/// Update W (koppelingen) via prediction error gradient
pub fn update_weights(field: &mut Field, id: u32) {
    let mu = field.mu;
    let point = &field.points[id as usize];
    let eps = prediction_error(point);
    let eta = point.eta;
    let phi_self = point.phi;

    let targets: Vec<u32> = point.couplings.iter().map(|c| c.target).collect();
    let neighbor_phis: Vec<[f32; DIM]> = targets
        .iter()
        .filter_map(|&t| field.points.get(t as usize).map(|p| p.phi))
        .collect();

    let point = &mut field.points[id as usize];

    for (i, coupling) in point.couplings.iter_mut().enumerate() {
        if i >= neighbor_phis.len() { break; }

        let mut dw = 0.0f32;
        for d in 0..DIM {
            let diff = phi_self[d] - neighbor_phis[i][d];
            dw += eps[d] * diff;
        }
        dw /= DIM as f32;

        coupling.momentum = mu * coupling.momentum + (1.0 - mu) * dw;
        coupling.weight -= eta * dw + mu * coupling.momentum;
        coupling.weight = coupling.weight.clamp(-5.0, 5.0);
    }
}

/// Update M (geheugen): M_{t+1} = ρ·M_t + (1-ρ)·ε_t
pub fn update_memory(field: &mut Field, id: u32) {
    let rho = field.rho;
    let point = &field.points[id as usize];
    let eps = prediction_error(point);

    let point = &mut field.points[id as usize];
    for d in 0..DIM {
        point.memory[d] = rho * point.memory[d] + (1.0 - rho) * eps[d];
    }
}

/// Bereken spanning: ||ε||² + κ·compute - ω·empowerment
pub fn compute_tension(field: &mut Field, id: u32) {
    let kappa = field.kappa;
    let omega = field.omega;
    let ema_decay = field.ema_decay;

    let point = &field.points[id as usize];
    let eps = prediction_error(point);
    let pred_error = squared_norm(&eps);
    let compute_cost = point.couplings.len() as f32 * 0.01;
    let empowerment: f32 = point.couplings.iter().map(|c| c.weight.abs()).sum();
    let tension = pred_error + kappa * compute_cost - omega * empowerment;

    let point = &mut field.points[id as usize];
    point.tension = tension.max(0.0);
    point.tension_ema = ema_decay * point.tension_ema + (1.0 - ema_decay) * point.tension;
}

// === NIEUW: Settle loop + token injection + equilibrium propagation ===

/// Run één volledige tick op alle actieve punten
pub fn tick_all(field: &mut Field) {
    let active: Vec<u32> = field.active_set.clone();
    for &id in &active {
        update_phi(field, id, None);
        update_memory(field, id);
        compute_tension(field, id);
    }
}

/// Settle het veld tot convergentie of max ticks
/// Retourneert: (ticks gebruikt, gemiddelde spanning)
pub fn settle_until_converged(field: &mut Field, max_ticks: usize, epsilon: f32) -> (usize, f32) {
    for t in 0..max_ticks {
        tick_all(field);

        // Check convergentie
        let n = field.active_set.len() as f32;
        if n == 0.0 { return (t, 0.0); }

        let total_tension: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].tension)
            .sum();

        let avg = total_tension / n;
        if avg < epsilon {
            return (t + 1, avg);
        }
    }

    let n = field.active_set.len() as f32;
    let avg = if n > 0.0 {
        field.active_set.iter()
            .map(|&id| field.points[id as usize].tension)
            .sum::<f32>() / n
    } else { 0.0 };

    (max_ticks, avg)
}

/// Injecteer een sequentie van tokens in de input-regio
/// Elke token wordt geklemd voor `ticks_per_token` ticks
pub fn inject_token_sequence(
    field: &mut Field,
    embeddings: &[[f32; DIM]],
    ticks_per_token: usize,
) {
    let input_ids = field.input_ids.clone();
    if input_ids.is_empty() { return; }

    for embedding in embeddings {
        // Klem alle input-punten op deze embedding
        for &id in &input_ids {
            field.clamps.insert(id, embedding.to_vec());
        }

        // Run ticks met deze clamp
        for _ in 0..ticks_per_token {
            tick_all(field);
        }
    }

    // Verwijder input clamps na injectie
    for &id in &input_ids {
        field.clamps.remove(&id);
    }
}

/// Equilibrium Propagation — het echte leeralgoritme
///
/// Twee fasen:
/// 1. Vrije fase: alleen input geklemd, veld settlet vrij
/// 2. Geklemde fase: input + output geklemd, veld settlet naar target
///
/// W update: W += eta * beta * (Φ_clamped·Φ_neighbor_clamped - Φ_free·Φ_neighbor_free)
///
/// Dit is biologisch plausibel: alleen lokale informatie, geen backprop
pub fn equilibrium_propagation(
    field: &mut Field,
    input_embeddings: &[[f32; DIM]],
    target_embeddings: &[[f32; DIM]],
    ticks_per_token: usize,
    settle_ticks: usize,
    ep_beta: f32,
) -> f32 {
    // === VRIJE FASE ===
    field.phase = FieldPhase::Processing;

    // Injecteer input
    inject_token_sequence(field, input_embeddings, ticks_per_token);

    // Settle vrij
    settle_until_converged(field, settle_ticks, 0.001);

    // Sla vrije-fase toestanden op
    let free_states: Vec<[f32; DIM]> = field.points.iter()
        .map(|p| p.phi)
        .collect();

    // === GEKLEMDE FASE ===
    field.phase = FieldPhase::Learning;

    // Klem output-punten op target
    let output_ids = field.output_ids.clone();
    if !output_ids.is_empty() && !target_embeddings.is_empty() {
        // Gebruik de gemiddelde target embedding als clamp voor alle output punten
        let mut avg_target = [0.0f32; DIM];
        for emb in target_embeddings {
            for d in 0..DIM { avg_target[d] += emb[d]; }
        }
        let n = target_embeddings.len() as f32;
        for d in 0..DIM { avg_target[d] /= n; }

        for &id in &output_ids {
            field.clamps.insert(id, avg_target.to_vec());
        }
    }

    // Re-injecteer input (geklemd)
    let input_ids = field.input_ids.clone();
    if !input_embeddings.is_empty() {
        let last_emb = input_embeddings.last().unwrap();
        for &id in &input_ids {
            field.clamps.insert(id, last_emb.to_vec());
        }
    }

    // Settle geklemd
    settle_until_converged(field, settle_ticks, 0.001);

    // === WEIGHT UPDATE ===
    let mut total_delta = 0.0f32;

    let active: Vec<u32> = field.active_set.clone();
    for &id in &active {
        let idx = id as usize;
        let phi_clamped = field.points[idx].phi;
        let phi_free = free_states[idx];

        let targets: Vec<(u32, usize)> = field.points[idx].couplings
            .iter()
            .enumerate()
            .map(|(ci, c)| (c.target, ci))
            .collect();

        for (target, ci) in targets {
            let t = target as usize;
            if t >= field.points.len() { continue; }

            let neighbor_clamped = field.points[t].phi;
            let neighbor_free = free_states[t];

            // EP weight update: W += beta * (clamped_correlation - free_correlation)
            let mut dw = 0.0f32;
            for d in 0..DIM {
                let corr_clamped = phi_clamped[d] * neighbor_clamped[d];
                let corr_free = phi_free[d] * neighbor_free[d];
                dw += corr_clamped - corr_free;
            }
            dw *= ep_beta / DIM as f32;

            let eta = field.points[idx].eta;
            field.points[idx].couplings[ci].weight += eta * dw;
            field.points[idx].couplings[ci].weight =
                field.points[idx].couplings[ci].weight.clamp(-5.0, 5.0);

            total_delta += dw.abs();
        }
    }

    // Verwijder alle clamps
    field.clamps.clear();
    field.phase = FieldPhase::Idle;

    total_delta
}

/// Voer Lucy's eigen veldstaat terug als sensorische input (zelf-model)
pub fn self_state_input(field: &Field) -> [f32; DIM] {
    let mut input = [0.0f32; DIM];
    let n = field.active_set.len() as f32;

    if n > 0.0 {
        let avg_tension: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].tension)
            .sum::<f32>() / n;

        let max_tension: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].tension)
            .fold(0.0f32, |a, b| a.max(b));

        let avg_energy: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].energy)
            .sum::<f32>() / n;

        input[0] = avg_tension;
        input[1] = avg_energy;
        input[2] = field.active_set.len() as f32 / field.points.len().max(1) as f32;
        input[3] = (field.tick as f32 / 1000.0).sin();
        input[4] = field.total_energy / field.points.len().max(1) as f32;
        input[5] = max_tension;
        input[6] = field.economy.balance_cents as f32 / 1000.0;

        for d in 7..DIM {
            input[d] = field.active_set.iter()
                .map(|&id| field.points[id as usize].phi[d])
                .sum::<f32>() / n;
        }
    }

    input
}

/// Injecteer exploratie-ruis als spanning te laag is
pub fn maybe_perturb(field: &mut Field) {
    use rand::Rng;
    let mut rng = rand::rng();

    let avg_tension: f32 = if !field.active_set.is_empty() {
        field.active_set.iter()
            .map(|&id| field.points[id as usize].tension)
            .sum::<f32>() / field.active_set.len() as f32
    } else { return; };

    if avg_tension < 0.001 {
        let n_perturb = (field.active_set.len() / 10).max(1);
        for _ in 0..n_perturb {
            let idx = rng.random_range(0..field.active_set.len());
            let id = field.active_set[idx] as usize;
            for d in 0..DIM {
                field.points[id].phi[d] += rng.random_range(-0.05..0.05);
            }
        }
    }
}
