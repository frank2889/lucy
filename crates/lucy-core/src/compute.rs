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

/// Bereken ||ε||² — de kern van spanning
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
pub fn update_phi(field: &mut Field, id: u32, input: Option<&[f32; DIM]>) {
    let lambda = field.lambda;
    let beta = field.beta;
    let alpha = field.alpha;

    // Bereken lokale spanning gradient vanuit buren
    let mut grad = [0.0f32; DIM];
    let point = &field.points[id as usize];
    let n_couplings = point.couplings.len();

    if n_couplings > 0 {
        // Verzamel buur-informatie
        let couplings: Vec<(u32, f32)> = point
            .couplings
            .iter()
            .map(|c| (c.target, c.weight))
            .collect();

        let phi_self = point.phi;

        for (target, weight) in &couplings {
            if let Some(neighbor) = field.points.get(*target as usize) {
                for d in 0..DIM {
                    // Gradient van lokale spanning: verschil met buur, gewogen door W
                    let diff = phi_self[d] - neighbor.phi[d];
                    grad[d] += weight * diff;
                }
            }
        }

        // Normaliseer
        let inv_n = 1.0 / n_couplings as f32;
        for d in 0..DIM {
            grad[d] *= inv_n;
        }
    }

    // Update Φ
    let point = &mut field.points[id as usize];
    for d in 0..DIM {
        // Sla huidige Φ op als voorspelling voor volgende tick
        point.prediction[d] = point.phi[d];

        // Φ_{t+1} = Φ_t - λ·grad + β·input - α·sign(Φ)
        point.phi[d] -= lambda * grad[d];

        if let Some(inp) = input {
            point.phi[d] += beta * inp[d];
        }

        // Sparse penalty: L1
        point.phi[d] -= alpha * point.phi[d].signum();

        // Clamp om numerieke explosie te voorkomen
        point.phi[d] = point.phi[d].clamp(-10.0, 10.0);
    }
}

/// Update W (koppelingen) voor een actief punt:
/// W_{t+1} = W_t - η·∂||ε||²/∂W + μ·momentum
pub fn update_weights(field: &mut Field, id: u32) {
    let mu = field.mu;
    let point = &field.points[id as usize];
    let eps = prediction_error(point);
    let eta = point.eta;
    let phi_self = point.phi;

    // Verzamel buur phi's
    let targets: Vec<u32> = point.couplings.iter().map(|c| c.target).collect();
    let neighbor_phis: Vec<[f32; DIM]> = targets
        .iter()
        .filter_map(|&t| field.points.get(t as usize).map(|p| p.phi))
        .collect();

    let point = &mut field.points[id as usize];

    for (i, coupling) in point.couplings.iter_mut().enumerate() {
        if i >= neighbor_phis.len() {
            break;
        }

        // ∂||ε||²/∂W ≈ ε · (Φ_self - Φ_neighbor) — lokale gradient
        let mut dw = 0.0f32;
        for d in 0..DIM {
            let diff = phi_self[d] - neighbor_phis[i][d];
            dw += eps[d] * diff;
        }
        dw /= DIM as f32;

        // Update momentum: M_{t+1} = μ·M_t + (1-μ)·gradient
        coupling.momentum = mu * coupling.momentum + (1.0 - mu) * dw;

        // Update gewicht: W_{t+1} = W_t - η·gradient + μ·momentum
        coupling.weight -= eta * dw + mu * coupling.momentum;

        // Clamp gewichten
        coupling.weight = coupling.weight.clamp(-5.0, 5.0);
    }
}

/// Update M (geheugen) voor een actief punt:
/// M_{t+1} = ρ·M_t + (1-ρ)·ε_t — onthoud verrassingen
pub fn update_memory(field: &mut Field, id: u32) {
    let rho = field.rho;
    let point = &field.points[id as usize];
    let eps = prediction_error(point);

    let point = &mut field.points[id as usize];
    for d in 0..DIM {
        point.memory[d] = rho * point.memory[d] + (1.0 - rho) * eps[d];
    }
}

/// Bereken totale spanning: ||ε||² + κ·compute - ω·empowerment
pub fn compute_tension(field: &mut Field, id: u32) {
    let kappa = field.kappa;
    let omega = field.omega;
    let ema_decay = field.ema_decay;

    let point = &field.points[id as usize];
    let eps = prediction_error(point);
    let pred_error = squared_norm(&eps);

    // Compute kosten: proportioneel aan aantal koppelingen (meer werk = meer kosten)
    let compute_cost = point.couplings.len() as f32 * 0.01;

    // Empowerment: hoe veel invloed heeft dit punt? (simpele proxy: som van |W|)
    let empowerment: f32 = point.couplings.iter().map(|c| c.weight.abs()).sum();

    let tension = pred_error + kappa * compute_cost - omega * empowerment;

    let point = &mut field.points[id as usize];
    point.tension = tension.max(0.0);
    // EMA voor vastloop-detectie
    point.tension_ema = ema_decay * point.tension_ema + (1.0 - ema_decay) * point.tension;
}

/// Voer Lucy's eigen veldstaat terug als sensorische input (zelf-model)
pub fn self_state_input(field: &Field) -> [f32; DIM] {
    let mut input = [0.0f32; DIM];
    let n = field.active_set.len() as f32;

    if n > 0.0 {
        // Gemiddelde spanning
        let avg_tension: f32 = field
            .active_set
            .iter()
            .map(|&id| field.points[id as usize].tension)
            .sum::<f32>()
            / n;

        // Max spanning (niet alleen gemiddelde — pieken zijn informatief)
        let max_tension: f32 = field
            .active_set
            .iter()
            .map(|&id| field.points[id as usize].tension)
            .fold(0.0f32, |a, b| a.max(b));

        // Gemiddelde energie
        let avg_energy: f32 = field
            .active_set
            .iter()
            .map(|&id| field.points[id as usize].energy)
            .sum::<f32>()
            / n;

        // Spanning-variantie (hoe ongelijk is spanning verdeeld?)
        let tension_var: f32 = field
            .active_set
            .iter()
            .map(|&id| {
                let d = field.points[id as usize].tension - avg_tension;
                d * d
            })
            .sum::<f32>()
            / n;

        // Vul de input vector met zelf-informatie
        input[0] = avg_tension;
        input[1] = avg_energy;
        input[2] = field.active_set.len() as f32 / field.points.len().max(1) as f32;
        input[3] = (field.tick as f32 / 1000.0).sin(); // Periodieke tijd — creëert ritme
        input[4] = field.total_energy / field.points.len().max(1) as f32;
        input[5] = max_tension;
        input[6] = tension_var;
        input[7] = field.economy.balance_cents as f32 / 1000.0; // Financiële staat als gevoel
        input[8] = if field.economy.is_profitable() { 1.0 } else { -1.0 };

        // Overige dimensies: gemiddelde phi over actieve punten (variërend per dimensie)
        for d in 9..DIM {
            input[d] = field
                .active_set
                .iter()
                .map(|&id| field.points[id as usize].phi[d % DIM])
                .sum::<f32>()
                / n;
        }
    }

    input
}

/// Injecteer exploratie-ruis als spanning te laag is (voorkom dood evenwicht)
pub fn maybe_perturb(field: &mut Field) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let avg_tension: f32 = if !field.active_set.is_empty() {
        field
            .active_set
            .iter()
            .map(|&id| field.points[id as usize].tension)
            .sum::<f32>()
            / field.active_set.len() as f32
    } else {
        return;
    };

    // Als spanning te laag → injecteer perturbatie in ~10% van actieve punten
    if avg_tension < 0.001 {
        let n_perturb = (field.active_set.len() / 10).max(1);
        for _ in 0..n_perturb {
            let idx = rng.gen_range(0..field.active_set.len());
            let id = field.active_set[idx] as usize;
            for d in 0..DIM {
                field.points[id].phi[d] += rng.gen_range(-0.05..0.05);
            }
        }
    }
}
