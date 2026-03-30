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
use smallvec::SmallVec;
use rayon::prelude::*;
use std::collections::HashMap;
use crate::encoding::Vocabulary;

/// Wrapper voor raw pointers zodat ze via rayon par_iter gestuurd kunnen worden.
/// SAFETY: alleen gebruiken wanneer elke thread naar een uniek index schrijft.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}

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
        point.phi = *clamp;
        return;
    }

    let lambda = field.lambda;
    let beta = field.beta;
    let alpha = field.alpha;
    let id_usize = id as usize;

    // Bereken lokale spanning gradient vanuit buren — GEEN Vec allocatie
    let mut grad = [0.0f32; DIM];
    let point = &field.points[id as usize];
    let n_couplings = point.couplings.len();

    if n_couplings > 0 {
        // Kopieer coupling data naar stack-buffer (SmallVec past op stack voor K≤16)
        let coupling_data: SmallVec<[(u32, f32); 16]> = point
            .couplings
            .iter()
            .map(|c| (c.target, c.weight))
            .collect();

        let phi_self = point.phi;

        for &(target, weight) in &coupling_data {
            if let Some(neighbor) = field.points.get(target as usize) {
                for d in 0..DIM {
                    grad[d] += weight * (phi_self[d] - neighbor.phi[d]);
                }
            }
        }

        let inv_n = 1.0 / n_couplings as f32;
        for d in 0..DIM {
            grad[d] *= inv_n;
        }
    }

    // Attention context (v0.9) — kopieer om borrow conflict te vermijden
    let attn_ctx = if id_usize < field.attn_context.len() {
        Some(field.attn_context[id_usize])
    } else {
        None
    };

    // Update Φ
    let point = &mut field.points[id_usize];
    for d in 0..DIM {
        point.prediction[d] = point.phi[d];
        point.phi[d] -= lambda * grad[d];

        if let Some(inp) = input {
            point.phi[d] += beta * inp[d];
        }

        // Attention context injectie (v0.9)
        if let Some(ref ctx) = attn_ctx {
            point.phi[d] += ATTN_GAMMA * ctx[d];
        }

        // Sparse penalty: L1
        point.phi[d] -= alpha * point.phi[d].signum();
    }

    // Layer norm verplaatst naar apply_layer_norm() — wordt elke 10 ticks aangeroepen
    // i.p.v. elke tick. Phi drifted niet snel genoeg voor per-tick normalisatie.
}

/// Layer normalization voor alle actieve punten — mean=0, var=1
/// v0.9+: Welford single-pass, elke 10 ticks, PARALLEL over actieve punten.
/// Elke punt schrijft alleen naar eigen phi — geen cross-point dependencies.
pub fn apply_layer_norm(field: &mut Field) {
    // v1.0: Parallel — berekent nieuwe phi's in bulk, dan toepassen
    // Elke punt is onafhankelijk, dus we berekenen parallel en schrijven terug.
    let new_phis: Vec<(usize, [f32; DIM])> = field.active_set
        .par_iter()
        .map(|&id| {
            let idx = id as usize;
            let phi = field.points[idx].phi;

            // Welford's online algorithm: mean + variance in 1 pass
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;
            for d in 0..DIM {
                let delta = phi[d] - mean;
                mean += delta / (d + 1) as f32;
                let delta2 = phi[d] - mean;
                m2 += delta * delta2;
            }
            let std = (m2 / DIM as f32 + 1e-5).sqrt();
            let inv_std = 1.0 / std;

            let mut normed = [0.0f32; DIM];
            for d in 0..DIM {
                normed[d] = ((phi[d] - mean) * inv_std).clamp(-10.0, 10.0);
            }
            (idx, normed)
        })
        .collect();

    // Schrijf terug (sequentieel maar snel — alleen copy)
    for (idx, normed) in new_phis {
        field.points[idx].phi = normed;
    }
}

/// Update W (koppelingen) via prediction error gradient — GEEN Vec allocatie
pub fn update_weights(field: &mut Field, id: u32) {
    let mu = field.mu;
    let point = &field.points[id as usize];
    let eps = prediction_error(point);
    let eta = point.eta;
    let phi_self = point.phi;

    // Kopieer buur-phi's naar stack-buffer (SmallVec past op stack voor K≤16)
    let neighbor_data: SmallVec<[[f32; DIM]; 16]> = point
        .couplings
        .iter()
        .filter_map(|c| field.points.get(c.target as usize).map(|p| p.phi))
        .collect();

    let point = &mut field.points[id as usize];

    for (i, coupling) in point.couplings.iter_mut().enumerate() {
        if i >= neighbor_data.len() { break; }

        let mut dw = 0.0f32;
        for d in 0..DIM {
            dw += eps[d] * (phi_self[d] - neighbor_data[i][d]);
        }
        dw /= DIM as f32;

        coupling.momentum = mu * coupling.momentum + (1.0 - mu) * dw;
        // EWC penalty: bescherm belangrijke weights tegen catastrophic forgetting
        let ewc_pen = ewc_penalty(coupling, 0.1);
        // v1.0: Broadcast couplings krijgen 2× learning rate (dragen bewust signaal)
        let lr = if coupling.broadcast { eta * 2.0 } else { eta };
        coupling.weight -= lr * dw + mu * coupling.momentum + lr * ewc_pen;
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

/// Bereken spanning + emotionele signalen
pub fn compute_tension(field: &mut Field, id: u32) {
    let kappa = field.kappa;
    let omega = field.omega;
    let ema_decay = field.ema_decay;
    let growth_threshold = field.growth_threshold;

    let point = &field.points[id as usize];
    let eps = prediction_error(point);
    let pred_error = squared_norm(&eps);
    let compute_cost = point.couplings.len() as f32 * 0.01;
    let empowerment: f32 = point.couplings.iter().map(|c| c.weight.abs()).sum();
    let old_tension_ema = point.tension_ema;

    let point = &mut field.points[id as usize];

    // Precision weighting (Friston): track error-variantie, weeg tension ermee
    let error_diff = pred_error - point.tension_ema;
    point.error_var = 0.99 * point.error_var + 0.01 * error_diff * error_diff;
    point.precision = 1.0 / (point.error_var + 0.01);
    // === v2.0: Free Energy Principle (Friston) ===
    // F = KL(q||p) + surprise
    // KL = complexiteit van het interne model (hoe ver wijkt phi af van een simpel model?)
    // Surprise = prediction error (al berekend)
    let phi_var: f32 = point.phi.iter().map(|x| x * x).sum::<f32>() / DIM as f32;
    let kl_proxy = 0.1 * phi_var; // Complexiteitspenalty — simpeler model = beter
    let free_energy = kl_proxy + point.precision * pred_error;
    point.kl_divergence = 0.99 * point.kl_divergence + 0.01 * kl_proxy;
    point.free_energy = 0.95 * point.free_energy + 0.05 * free_energy;

    // Tension = Free Energy + kosten - beloning
    // Lucy minimaliseert F door haar model te verbeteren (leren) OF acties te nemen (exploratie)
    point.tension = (free_energy + kappa * compute_cost - omega * empowerment).max(0.0);
    point.tension_ema = ema_decay * old_tension_ema + (1.0 - ema_decay) * point.tension;

    // Emotionele signalen (exponentieel afvallend)
    let emo_decay = 0.95;

    // Nieuwsgierigheid: precision-gewogen verrassing
    // v1.3: NaN-safe — .max(0.0) voorkomt sqrt van negatief getal
    let surprise = (point.precision * pred_error.max(0.0).sqrt() - old_tension_ema.max(0.0).sqrt()).max(0.0);
    let surprise = if surprise.is_nan() || surprise.is_infinite() { 0.0 } else { surprise };
    point.curiosity = emo_decay * point.curiosity + (1.0 - emo_decay) * surprise;
    if point.curiosity.is_nan() || point.curiosity.is_infinite() { point.curiosity = 0.0; }

    // Tevredenheid: spanning daalt → leren werkt
    let improvement = (old_tension_ema - point.tension).max(0.0);
    point.satisfaction = emo_decay * point.satisfaction + (1.0 - emo_decay) * improvement;

    // Frustratie: spanning hoog EN stabiel → vastgelopen
    let flatness = 1.0 / (1.0 + (point.tension - point.tension_ema).abs() * 10.0);
    let stuck = (point.tension - growth_threshold).max(0.0) * flatness;
    point.frustration = emo_decay * point.frustration + (1.0 - emo_decay) * stuck;
}

/// kWTA (k-Winner-Take-All): alleen top-K% punten blijven actief
/// Dit forceert scherpe representaties — het verschil tussen soep en gedachten
pub fn apply_kwta(field: &mut Field) {
    let n = field.active_set.len();
    if n < 20 { return; } // te weinig punten voor kWTA

    // Top 10% mag actief blijven (hidden punten — input/output altijd actief)
    let k = (n / 10).max(10);

    // Sorteer op phi-magnitude (hoe sterk "denkt" dit punt?)
    let mut scores: Vec<(usize, f32)> = field.active_set.iter()
        .map(|&id| {
            let p = &field.points[id as usize];
            let mag: f32 = p.phi.iter().map(|x| x * x).sum();
            (id as usize, mag)
        })
        .collect();

    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k = actief, rest = stil (maar input/output altijd actief)
    for (rank, &(idx, _)) in scores.iter().enumerate() {
        let p = &mut field.points[idx];
        if p.is_input() || p.is_output() {
            p.kwta_active = true;
        } else {
            p.kwta_active = rank < k;
        }
        // Stille punten: dempen phi richting nul (niet hard nul — zachte inhibitie)
        if !p.kwta_active {
            for d in 0..DIM {
                p.phi[d] *= 0.9; // zachte decay, niet hard afkappen
            }
        }
    }
}

// === Settle loop + token injection + equilibrium propagation ===

/// Run één volledige tick op alle actieve punten — geen clone
pub fn tick_all(field: &mut Field) {
    if field.active_set.len() >= 128 {
        tick_all_parallel(field);
    } else {
        for i in 0..field.active_set.len() {
            let id = field.active_set[i];
            update_phi(field, id, None);
            update_memory(field, id);
            compute_tension(field, id);
        }
    }
}

/// Parallelle tick: Jacobi-stijl (lees vorige staat, schrijf nieuwe staat)
/// Correcte fysica: alle punten updaten simultaan op basis van vorige tick
fn tick_all_parallel(field: &mut Field) {
    let lambda = field.lambda;
    let _beta = field.beta; // beschikbaar voor input injection in toekomst
    let alpha = field.alpha;
    let rho = field.rho;
    let kappa = field.kappa;
    let omega = field.omega;
    let ema_decay = field.ema_decay;
    let growth_threshold = field.growth_threshold;

    // Lees alle punt-data immutable (snapshot van vorige tick)
    let points = &field.points;
    let active_set = &field.active_set;

    // Parallel: bereken nieuwe phi + memory + tension per punt
    struct PointUpdate {
        id: usize,
        phi: [f32; DIM],
        prediction: [f32; DIM],
        memory: [f32; DIM],
        tension: f32,
        tension_ema: f32,
        error_var: f32,
        precision: f32,
        curiosity: f32,
        satisfaction: f32,
        frustration: f32,
    }

    let updates: Vec<PointUpdate> = active_set.par_iter().filter_map(|&id| {
        let idx = id as usize;
        let point = points.get(idx)?;

        // --- phi update ---
        if let Some(clamp) = field.clamps.get(&id) {
            // Geklemde punten: forceer phi
            let mut upd = PointUpdate {
                id: idx,
                phi: *clamp,
                prediction: point.phi,
                memory: point.memory,
                tension: point.tension,
                tension_ema: point.tension_ema,
                error_var: point.error_var,
                precision: point.precision,
                curiosity: point.curiosity,
                satisfaction: point.satisfaction,
                frustration: point.frustration,
            };
            // Memory update
            let eps = {
                let mut e = [0.0f32; DIM];
                for d in 0..DIM { e[d] = upd.phi[d] - upd.prediction[d]; }
                e
            };
            for d in 0..DIM {
                upd.memory[d] = rho * point.memory[d] + (1.0 - rho) * eps[d];
            }
            return Some(upd);
        }

        // Lokale spanning gradient vanuit buren
        let mut grad = [0.0f32; DIM];
        let n_couplings = point.couplings.len();
        if n_couplings > 0 {
            for c in &point.couplings {
                if let Some(neighbor) = points.get(c.target as usize) {
                    for d in 0..DIM {
                        grad[d] += c.weight * (point.phi[d] - neighbor.phi[d]);
                    }
                }
            }
            let inv_n = 1.0 / n_couplings as f32;
            for d in 0..DIM { grad[d] *= inv_n; }
        }

        let mut new_phi = [0.0f32; DIM];
        let mut new_pred = [0.0f32; DIM];
        for d in 0..DIM {
            new_pred[d] = point.phi[d];
            new_phi[d] = point.phi[d] - lambda * grad[d] - alpha * point.phi[d].signum();
            new_phi[d] = new_phi[d].clamp(-10.0, 10.0);
        }

        // --- memory update ---
        let eps = {
            let mut e = [0.0f32; DIM];
            for d in 0..DIM { e[d] = new_phi[d] - new_pred[d]; }
            e
        };
        let mut new_memory = [0.0f32; DIM];
        for d in 0..DIM {
            new_memory[d] = rho * point.memory[d] + (1.0 - rho) * eps[d];
        }

        // --- tension + precision weighting + emoties ---
        let pred_error = squared_norm(&eps);
        let compute_cost = n_couplings as f32 * 0.01;
        let empowerment: f32 = point.couplings.iter().map(|c| c.weight.abs()).sum();
        let error_diff = pred_error - point.tension_ema;
        let error_var = 0.99 * point.error_var + 0.01 * error_diff * error_diff;
        let precision = 1.0 / (error_var + 0.01);
        let tension = (precision * pred_error + kappa * compute_cost - omega * empowerment).max(0.0);
        let tension_ema = ema_decay * point.tension_ema + (1.0 - ema_decay) * tension;

        let emo_decay = 0.95;
        let surprise = (precision * pred_error.max(0.0).sqrt() - point.tension_ema.max(0.0).sqrt()).max(0.0);
        let surprise = if surprise.is_nan() || surprise.is_infinite() { 0.0 } else { surprise };
        let curiosity = emo_decay * point.curiosity + (1.0 - emo_decay) * surprise;
        let curiosity = if curiosity.is_nan() { 0.0 } else { curiosity };
        let improvement = (point.tension_ema - tension).max(0.0);
        let satisfaction = emo_decay * point.satisfaction + (1.0 - emo_decay) * improvement;
        let flatness = 1.0 / (1.0 + (tension - point.tension_ema).abs() * 10.0);
        let stuck = (tension - growth_threshold).max(0.0) * flatness;
        let frustration = emo_decay * point.frustration + (1.0 - emo_decay) * stuck;

        Some(PointUpdate {
            id: idx,
            phi: new_phi,
            prediction: new_pred,
            memory: new_memory,
            tension,
            tension_ema,
            error_var,
            precision,
            curiosity,
            satisfaction,
            frustration,
        })
    }).collect();

    // Apply: sequentieel schrijven (snel, geen contention)
    for upd in updates {
        let p = &mut field.points[upd.id];
        p.phi = upd.phi;
        p.prediction = upd.prediction;
        p.memory = upd.memory;
        p.tension = upd.tension;
        p.tension_ema = upd.tension_ema;
        p.error_var = upd.error_var;
        p.precision = upd.precision;
        p.curiosity = upd.curiosity;
        p.satisfaction = upd.satisfaction;
        p.frustration = upd.frustration;
    }
}

/// Attention-gewogen tick: punten met hoge spanning krijgen extra updates
/// v1.0: Cache tensions in 1 pass, skip converged punten efficiënt
pub fn attention_tick(field: &mut Field) {
    let n = field.active_set.len();
    if n == 0 { return; }

    // v1.0: Cache tensions — voorkomt herberekening na update_phi/compute_tension
    let tensions: SmallVec<[f32; 2048]> = field.active_set.iter()
        .map(|&id| field.points[id as usize].tension)
        .collect();

    let avg_tension: f32 = tensions.iter().sum::<f32>() / n as f32;
    let high_threshold = avg_tension * 2.0;
    let low_threshold = avg_tension * 0.1;

    for (i, &tension) in tensions.iter().enumerate() {
        if tension < low_threshold { continue; }

        let id = field.active_set[i];
        update_phi(field, id, None);
        update_memory(field, id);
        compute_tension(field, id);

        // Hoge spanning → extra updates (2× meer aandacht)
        if tension > high_threshold {
            update_phi(field, id, None);
            compute_tension(field, id);
        }
    }
}

/// v1.2: Lichte settle voor queries — alleen IO punten + buren (niet alle 8704)
/// ~10× sneller dan full settle. Gebruikt voor /ask endpoint.
pub fn settle_light(field: &mut Field, max_ticks: usize, epsilon: f32) -> (usize, f32) {
    // Bouw subset: input + output + hun directe buren
    let mut subset: Vec<u32> = Vec::with_capacity(2048);
    let mut seen = std::collections::HashSet::new();

    for &id in field.input_ids.iter().chain(field.output_ids.iter()) {
        if seen.insert(id) { subset.push(id); }
        // Voeg directe buren toe
        let neighbors: SmallVec<[u32; 16]> = field.points[id as usize].couplings
            .iter().map(|c| c.target).collect();
        for &n in &neighbors {
            if seen.insert(n) { subset.push(n); }
        }
    }

    for t in 0..max_ticks {
        for &id in &subset {
            update_phi(field, id, None);
            update_memory(field, id);
            compute_tension(field, id);
        }

        if (t + 1) % 5 == 0 || t + 1 == max_ticks {
            let total: f32 = subset.iter()
                .map(|&id| field.points[id as usize].tension)
                .sum();
            let avg = total / subset.len().max(1) as f32;
            if avg < epsilon { return (t + 1, avg); }
        }
    }

    let avg = subset.iter()
        .map(|&id| field.points[id as usize].tension)
        .sum::<f32>() / subset.len().max(1) as f32;
    (max_ticks, avg)
}

/// Full settle — alle actieve punten. Gebruikt voor training (EP).
/// Retourneert: (ticks gebruikt, gemiddelde spanning)
pub fn settle_until_converged(field: &mut Field, max_ticks: usize, epsilon: f32) -> (usize, f32) {
    let attn_interval = 10; // v0.9+: elke 10 ticks (was 5) — halveert attention kosten
    let norm_interval = 10; // v0.9+: layer norm elke 10 ticks (was elke tick)
    let check_interval = 10;

    // Clone attention weights EENMAAL voor de hele settle loop
    let attn_weights_buf = field.attn_weights.clone();

    for t in 0..max_ticks {
        attention_tick(field);

        // v0.9+: Layer norm elke 10 ticks (was elke tick in update_phi)
        if (t + 1) % norm_interval == 0 {
            apply_layer_norm(field);
        }

        // v0.9+: Attention elke 10 ticks — weights referentie, GEEN clone
        if (t + 1) % attn_interval == 0 {
            if let Some(ref w) = attn_weights_buf {
                attention_gate_with_weights(field, w);
            }
        }

        // Check convergentie elke 10 ticks
        if (t + 1) % check_interval != 0 && t + 1 != max_ticks { continue; }

        let n = field.active_set.len() as f32;
        if n == 0.0 { return (t + 1, 0.0); }

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

/// Roteer een embedding met `shift` posities — elk input-punt ziet een uniek perspectief
#[inline]
pub fn rotate_embedding(emb: &[f32; DIM], shift: usize) -> [f32; DIM] {
    let mut rotated = [0.0f32; DIM];
    for d in 0..DIM {
        rotated[d] = emb[(d + shift) % DIM];
    }
    rotated
}

/// v1.2: Lichte injectie — alleen IO punten + buren updaten (voor queries)
pub fn inject_token_sequence_light(
    field: &mut Field,
    embeddings: &[[f32; DIM]],
    ticks_per_token: usize,
) {
    let input_ids = field.input_ids.clone();
    if input_ids.is_empty() { return; }

    // Bouw IO subset
    let mut subset: Vec<u32> = Vec::with_capacity(1024);
    let mut seen = std::collections::HashSet::new();
    for &id in input_ids.iter().chain(field.output_ids.iter()) {
        if seen.insert(id) { subset.push(id); }
        let neighbors: SmallVec<[u32; 16]> = field.points[id as usize].couplings
            .iter().map(|c| c.target).collect();
        for &n in &neighbors {
            if seen.insert(n) { subset.push(n); }
        }
    }

    for embedding in embeddings {
        for (i, &id) in input_ids.iter().enumerate() {
            field.clamps.insert(id, rotate_embedding(embedding, i));
        }
        for _ in 0..ticks_per_token {
            for &id in &subset {
                update_phi(field, id, None);
                compute_tension(field, id);
            }
        }
    }

    for &id in &input_ids {
        field.clamps.remove(&id);
    }
}

/// Full injectie — alle punten (voor training)
pub fn inject_token_sequence(
    field: &mut Field,
    embeddings: &[[f32; DIM]],
    ticks_per_token: usize,
) {
    let input_ids = field.input_ids.clone();
    if input_ids.is_empty() { return; }

    for embedding in embeddings {
        // Elk input-punt krijgt een uniek perspectief (geroteerde embedding)
        for (i, &id) in input_ids.iter().enumerate() {
            field.clamps.insert(id, rotate_embedding(embedding, i));
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

    // v0.9+: Attention gate vóór injectie — context-aware verwerking
    if field.attn_weights.is_some() {
        attention_gate(field);
    }

    // Injecteer input
    inject_token_sequence(field, input_embeddings, ticks_per_token);

    // Settle vrij
    settle_until_converged(field, settle_ticks, 0.001);

    // Sla vrije-fase toestanden op — ALLEEN actieve punten + buren (was: ALLE punten)
    // Bespaart ~80% geheugen (actief ~1000 + buren ~5000 vs totaal 8704+)
    let mut free_states: HashMap<usize, [f32; DIM]> = HashMap::with_capacity(field.active_set.len() * 2);
    for &id in &field.active_set {
        let idx = id as usize;
        free_states.insert(idx, field.points[idx].phi);
        // Ook buren opslaan (nodig voor neighbor_free in weight update)
        for c in &field.points[idx].couplings {
            let t = c.target as usize;
            free_states.entry(t).or_insert(field.points[t].phi);
        }
    }

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

        for (i, &id) in output_ids.iter().enumerate() {
            field.clamps.insert(id, rotate_embedding(&avg_target, i));
        }
    }

    // v0.9+: Re-injecteer VOLLEDIGE input context (niet alleen laatste token)
    // Gemiddelde van alle input embeddings → meer context in de geklemde fase
    let input_ids = field.input_ids.clone();
    if !input_embeddings.is_empty() {
        let mut avg_input = [0.0f32; DIM];
        for emb in input_embeddings {
            for d in 0..DIM { avg_input[d] += emb[d]; }
        }
        let n = input_embeddings.len() as f32;
        for d in 0..DIM { avg_input[d] /= n; }
        for (i, &id) in input_ids.iter().enumerate() {
            field.clamps.insert(id, rotate_embedding(&avg_input, i));
        }
    }

    // Settle geklemd
    settle_until_converged(field, settle_ticks, 0.001);

    // === WEIGHT UPDATE ===
    let mut total_delta = 0.0f32;

    for i in 0..field.active_set.len() {
        let id = field.active_set[i];
        let idx = id as usize;
        let phi_clamped = field.points[idx].phi;
        let phi_free = match free_states.get(&idx) {
            Some(s) => *s,
            None => continue, // punt niet opgeslagen → skip
        };

        let targets: SmallVec<[(u32, usize); 16]> = field.points[idx].couplings
            .iter()
            .enumerate()
            .map(|(ci, c)| (c.target, ci))
            .collect();

        for (target, ci) in targets {
            let t = target as usize;
            if t >= field.points.len() { continue; }

            let neighbor_clamped = field.points[t].phi;
            let neighbor_free = match free_states.get(&t) {
                Some(s) => *s,
                None => continue, // buur niet opgeslagen → skip
            };

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

/// Genereer hard negatives voor contrastief leren.
/// Met vocab: zoek 2e-meest-gelijkende token (bijna-juist maar fout = sterk leersignaal).
/// Zonder vocab: fallback naar inversie + ruis.
fn generate_hard_negatives(target_embeddings: &[[f32; DIM]], vocab: Option<&Vocabulary>) -> Vec<[f32; DIM]> {
    if let Some(v) = vocab {
        let vocab_size = v.vocab_size();
        if vocab_size > 10 {
            // Hard negatives: voor elke target embedding, vind het 2e-beste token
            return target_embeddings.iter().map(|target| {
                let mut best_id = 0u32;
                let mut best_sim = f32::NEG_INFINITY;
                let mut second_id = 0u32;
                let mut second_sim = f32::NEG_INFINITY;

                // Stap door vocab, zoek top-2
                for id in 3..vocab_size as u32 { // skip <pad>, <unk>, <eos>
                    let emb = v.embed(id);
                    let mut dot = 0.0f32;
                    for d in 0..DIM { dot += target[d] * emb[d]; }

                    if dot > best_sim {
                        second_id = best_id;
                        second_sim = best_sim;
                        best_id = id;
                        best_sim = dot;
                    } else if dot > second_sim {
                        second_id = id;
                        second_sim = dot;
                    }
                }

                // Return embedding van 2e-beste token
                *v.embed(second_id)
            }).collect();
        }
    }

    // Fallback: inversie + ruis (originele methode)
    use rand::Rng;
    let mut rng = rand::rng();
    target_embeddings.iter().map(|emb| {
        let mut neg = [0.0f32; DIM];
        for d in 0..DIM {
            neg[d] = -emb[d] * 0.5 + rng.random_range(-0.3..0.3f32);
        }
        neg
    }).collect()
}

/// v0.9+: Contrastieve EP — leer van positief EN negatief voorbeeld
/// Hard negatives: zoek 2e-beste token in vocab (bijna-juist maar fout).
/// Veel sterker leersignaal dan random ruis. Fallback naar inversie+ruis als geen vocab.
pub fn contrastive_ep(
    field: &mut Field,
    input_embeddings: &[[f32; DIM]],
    target_embeddings: &[[f32; DIM]],
    ticks_per_token: usize,
    settle_ticks: usize,
    ep_beta: f32,
) -> f32 {
    contrastive_ep_with_vocab(field, input_embeddings, target_embeddings, ticks_per_token, settle_ticks, ep_beta, None)
}

/// Contrastieve EP met optioneel vocabulary voor hard negatives
pub fn contrastive_ep_with_vocab(
    field: &mut Field,
    input_embeddings: &[[f32; DIM]],
    target_embeddings: &[[f32; DIM]],
    ticks_per_token: usize,
    settle_ticks: usize,
    ep_beta: f32,
    vocab: Option<&Vocabulary>,
) -> f32 {
    // Stap 1: normale EP (positief voorbeeld)
    let positive_delta = equilibrium_propagation(
        field, input_embeddings, target_embeddings, ticks_per_token, settle_ticks, ep_beta,
    );

    // v0.9+: Hard negatives — zoek 2e-beste token per target embedding
    let neg_target = generate_hard_negatives(target_embeddings, vocab);

    // Stap 3: geklemde fase met negatief voorbeeld → duw vandaan
    field.phase = FieldPhase::Processing;
    inject_token_sequence(field, input_embeddings, ticks_per_token);
    settle_until_converged(field, settle_ticks / 2, 0.005); // korter settlen voor negatief

    // Sparse free states — alleen actieve punten + buren
    let mut free_states: HashMap<usize, [f32; DIM]> = HashMap::with_capacity(field.active_set.len() * 2);
    for &id in &field.active_set {
        let idx = id as usize;
        free_states.insert(idx, field.points[idx].phi);
        for c in &field.points[idx].couplings {
            let t = c.target as usize;
            free_states.entry(t).or_insert(field.points[t].phi);
        }
    }

    field.phase = FieldPhase::Learning;
    let output_ids = field.output_ids.clone();
    if !output_ids.is_empty() && !neg_target.is_empty() {
        let mut avg_neg = [0.0f32; DIM];
        for emb in &neg_target { for d in 0..DIM { avg_neg[d] += emb[d]; } }
        let n = neg_target.len() as f32;
        for d in 0..DIM { avg_neg[d] /= n; }
        for (i, &id) in output_ids.iter().enumerate() {
            field.clamps.insert(id, rotate_embedding(&avg_neg, i));
        }
    }

    settle_until_converged(field, settle_ticks / 2, 0.005);

    // Negatieve weight update: DUW vandaan (−margin × correlatie)
    let margin = 0.3f32;
    for i in 0..field.active_set.len() {
        let id = field.active_set[i];
        let idx = id as usize;
        let phi_neg = field.points[idx].phi;
        let phi_free = match free_states.get(&idx) { Some(s) => *s, None => continue };

        let targets: SmallVec<[(u32, usize); 16]> = field.points[idx].couplings
            .iter().enumerate().map(|(ci, c)| (c.target, ci)).collect();

        for (target, ci) in targets {
            let t = target as usize;
            if t >= field.points.len() { continue; }

            let neighbor_neg = field.points[t].phi;
            let neighbor_free = match free_states.get(&t) { Some(s) => *s, None => continue };

            let mut dw = 0.0f32;
            for d in 0..DIM {
                let corr_neg = phi_neg[d] * neighbor_neg[d];
                let corr_free = phi_free[d] * neighbor_free[d];
                dw += corr_neg - corr_free;
            }
            dw *= ep_beta * margin / DIM as f32;

            let eta = field.points[idx].eta;
            // NEGATIEF: duw vandaan
            field.points[idx].couplings[ci].weight -= eta * dw;
            field.points[idx].couplings[ci].weight =
                field.points[idx].couplings[ci].weight.clamp(-5.0, 5.0);
        }
    }

    field.clamps.clear();
    field.phase = FieldPhase::Idle;

    positive_delta
}

/// Lucy's zelf-model: het veld voelt zijn eigen toestand (primitief zelfbewustzijn)
/// Bevat spanning, energie, emoties, reflectie-teller, en veld-gemiddelden
/// v1.0: Neem een temporele snapshot — Lucy voelt haar eigen trends.
/// Roep elke 100 ticks aan vanuit main.rs.
pub fn record_temporal_snapshot(field: &mut Field) {
    let n = field.active_set.len() as f32;
    let avg_tension = if n > 0.0 {
        field.active_set.iter().map(|&id| field.points[id as usize].tension).sum::<f32>() / n
    } else { 0.0 };

    let goal_progress = if field.goals.is_empty() { 0.0 } else {
        field.goals.iter().map(|g| g.progress).sum::<f32>() / field.goals.len() as f32
    };

    field.temporal_history.push(TemporalSnapshot {
        tick: field.tick,
        avg_tension,
        dopamine: field.dopamine,
        serotonin: field.serotonin,
        n_active: field.active_set.len() as u32,
        active_goal_progress: goal_progress,
    });

    // Ring buffer: max 64 entries
    if field.temporal_history.len() > 64 {
        field.temporal_history.remove(0);
    }
}

/// v1.0: Bereken trends — lineaire regressie over laatste 16 snapshots.
/// Retourneert (tension_trend, dopamine_trend): negatief = dalend, positief = stijgend.
pub fn compute_trends(field: &Field) -> (f32, f32) {
    let history = &field.temporal_history;
    if history.len() < 4 { return (0.0, 0.0); }

    let recent = &history[history.len().saturating_sub(16)..];
    let n = recent.len() as f32;

    // Simpele lineaire regressie: slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
    let mean_x = (n - 1.0) / 2.0;
    let mean_tension: f32 = recent.iter().map(|s| s.avg_tension).sum::<f32>() / n;
    let mean_dopamine: f32 = recent.iter().map(|s| s.dopamine).sum::<f32>() / n;

    let mut sx2 = 0.0f32;
    let mut sxy_tension = 0.0f32;
    let mut sxy_dopamine = 0.0f32;

    for (i, s) in recent.iter().enumerate() {
        let dx = i as f32 - mean_x;
        sx2 += dx * dx;
        sxy_tension += dx * (s.avg_tension - mean_tension);
        sxy_dopamine += dx * (s.dopamine - mean_dopamine);
    }

    if sx2 < 1e-8 { return (0.0, 0.0); }

    (sxy_tension / sx2, sxy_dopamine / sx2)
}

/// v1.0: Log een causaal event — "ik deed X, spanning was Y"
pub fn log_causal_event(field: &mut Field, action: &str) {
    let n = field.active_set.len() as f32;
    let tension = if n > 0.0 {
        field.active_set.iter().map(|&id| field.points[id as usize].tension).sum::<f32>() / n
    } else { 0.0 };

    field.causal_log.push(CausalEvent {
        tick: field.tick,
        action: action.to_string(),
        tension_before: tension,
        tension_after: 0.0, // wordt 100 ticks later ingevuld
        dopamine_delta: 0.0,
    });

    // Ring buffer: max 256 entries
    if field.causal_log.len() > 256 {
        field.causal_log.remove(0);
    }
}

/// v1.0: Vul causale events achteraf in — meet effect 100 ticks later
pub fn update_causal_events(field: &mut Field) {
    let tick = field.tick;
    let n = field.active_set.len() as f32;
    let current_tension = if n > 0.0 {
        field.active_set.iter().map(|&id| field.points[id as usize].tension).sum::<f32>() / n
    } else { 0.0 };

    for event in &mut field.causal_log {
        if event.tension_after == 0.0 && tick >= event.tick + 100 && tick < event.tick + 200 {
            event.tension_after = current_tension;
            event.dopamine_delta = field.dopamine - event.tension_before; // rough proxy
        }
    }
}

/// v1.0: Welke actie werkte het beste bij vergelijkbare staat?
pub fn best_action_for_state(field: &Field) -> Option<String> {
    if field.causal_log.len() < 10 { return None; }

    let n = field.active_set.len() as f32;
    let current_tension = if n > 0.0 {
        field.active_set.iter().map(|&id| field.points[id as usize].tension).sum::<f32>() / n
    } else { return None; };

    // Zoek events met vergelijkbare spanning (±30%)
    let mut best_action: Option<(String, f32)> = None;
    for event in &field.causal_log {
        if event.tension_after == 0.0 { continue; } // nog niet ingevuld
        let similarity = 1.0 - ((event.tension_before - current_tension).abs() / current_tension.max(0.01)).min(1.0);
        if similarity < 0.7 { continue; } // te anders

        let improvement = event.tension_before - event.tension_after; // positief = verbetering
        if let Some((_, best_imp)) = &best_action {
            if improvement > *best_imp {
                best_action = Some((event.action.clone(), improvement));
            }
        } else {
            best_action = Some((event.action.clone(), improvement));
        }
    }

    best_action.map(|(a, _)| a)
}

pub fn self_state_input(field: &Field) -> [f32; DIM] {
    let mut input = [0.0f32; DIM];
    let n = field.active_set.len() as f32;

    if n > 0.0 {
        // v0.9+: Single-pass over active_set — was 5+ aparte loops
        let mut sum_tension = 0.0f32;
        let mut max_tension = 0.0f32;
        let mut sum_energy = 0.0f32;
        let mut sum_curiosity = 0.0f32;
        let mut sum_satisfaction = 0.0f32;
        let mut sum_frustration = 0.0f32;
        let mut phi_sums = [0.0f32; DIM];

        for &id in &field.active_set {
            let p = &field.points[id as usize];
            sum_tension += p.tension;
            if p.tension > max_tension { max_tension = p.tension; }
            sum_energy += p.energy;
            sum_curiosity += p.curiosity;
            sum_satisfaction += p.satisfaction;
            sum_frustration += p.frustration;
            for d in 18..DIM {
                phi_sums[d] += p.phi[d];
            }
        }

        let inv_n = 1.0 / n;

        // Basis-staat
        input[0] = sum_tension * inv_n;
        input[1] = sum_energy * inv_n;
        input[2] = field.active_set.len() as f32 / field.points.len().max(1) as f32;
        input[3] = (field.tick as f32 / 1000.0).sin();
        input[4] = field.total_energy / field.points.len().max(1) as f32;
        input[5] = max_tension;
        input[6] = field.economy.balance_cents as f32 / 1000.0;

        // Emoties
        input[7] = sum_curiosity * inv_n;
        input[8] = sum_satisfaction * inv_n;
        input[9] = sum_frustration * inv_n;

        // Bewustzijns-indicatoren
        input[10] = field.reflection_count as f32 / 1000.0;
        input[11] = if field.last_thought.is_some() { 1.0 } else { 0.0 };

        // Neuromodulatoren
        input[12] = field.dopamine;
        input[13] = field.norepinephrine;
        input[14] = field.serotonin;
        input[15] = field.acetylcholine;

        // Edge of chaos
        input[16] = field.spectral_radius;
        input[17] = field.intention_strength;

        // Veld-gemiddelden (al berekend in de single pass)
        for d in 18..DIM {
            input[d] = phi_sums[d] * inv_n;
        }
    }

    input
}

/// Reflectie-tick: Lucy herkauwt haar laatste gedachte (continuous thought stream)
/// Zachte herinjectie (β = 0.1) laat het veld nieuwe attractoren ontdekken
pub fn reflect_tick(field: &mut Field) {
    let thought = match field.last_thought {
        Some(t) => t,
        None => return,
    };

    // Zachte injectie — niet klemmen, maar bijmengen via input
    let input_ids_len = field.input_ids.len();
    for i in 0..input_ids_len {
        let id = field.input_ids[i];
        let rotated = rotate_embedding(&thought, i);
        // Zachte bijmenging: β = 0.1 (5× zwakker dan normale input)
        let point = &mut field.points[id as usize];
        for d in 0..DIM {
            point.phi[d] += 0.1 * rotated[d];
            point.phi[d] = point.phi[d].clamp(-10.0, 10.0);
        }
    }

    // Tick het veld
    tick_all(field);

    field.reflection_count += 1;
}

// === v0.7: EDGE OF CHAOS — Spectral radius monitoring + auto-tuning ===

/// Meet de effectieve spectral radius: injecteer ruis, meet verspreiding na N ticks.
/// Ratio > 1.0 = chaotisch, < 1.0 = stabiel, ~1.0 = edge of chaos (optimaal).
/// Pas lambda automatisch aan om spectral radius naar chaos_target te sturen.
pub fn measure_spectral_radius(field: &mut Field) {
    use rand::Rng;
    let mut rng = rand::rng();

    if field.active_set.is_empty() { return; }

    // Kies een random actief punt
    let idx = rng.random_range(0..field.active_set.len());
    let probe_id = field.active_set[idx] as usize;

    // Sla ALLE phi's op — tick_all muteert het hele veld, niet alleen probe punt
    let saved_phis: Vec<[f32; DIM]> = field.points.iter().map(|p| p.phi).collect();

    // Injecteer kleine ruis op probe punt
    let perturbation = 0.01f32;
    for d in 0..DIM {
        field.points[probe_id].phi[d] += perturbation * (rng.random_range(-1.0f32..1.0f32));
    }

    // Meet initiële afwijking
    let mut initial_diff = 0.0f32;
    for d in 0..DIM {
        let diff = field.points[probe_id].phi[d] - saved_phis[probe_id][d];
        initial_diff += diff * diff;
    }
    initial_diff = initial_diff.sqrt().max(1e-10);

    // Run 10 ticks
    for _ in 0..10 {
        tick_all(field);
    }

    // Meet finale afwijking
    let mut final_diff = 0.0f32;
    for d in 0..DIM {
        let diff = field.points[probe_id].phi[d] - saved_phis[probe_id][d];
        final_diff += diff * diff;
    }
    final_diff = final_diff.sqrt().max(1e-10);

    // Spectral radius ≈ (final_diff / initial_diff)^(1/10)
    let ratio = final_diff / initial_diff;
    let sr = ratio.powf(0.1); // 10e wortel

    // EMA update spectral radius
    field.spectral_radius = 0.9 * field.spectral_radius + 0.1 * sr;

    // Auto-tune lambda naar chaos_target
    let error = field.spectral_radius - field.chaos_target;
    field.lambda *= 1.0 - 0.01 * error;
    field.lambda = field.lambda.clamp(0.001, 1.0);

    // Herstel ALLE punten — geen veld-corruptie meer
    for (i, phi) in saved_phis.into_iter().enumerate() {
        field.points[i].phi = phi;
    }
}

// === v0.7.2: HIËRARCHISCHE PREDICTIE — abstractie door laag-voor-laag voorspelling ===

/// Punten op depth N voorspellen de gemiddelde activatie van hun buren op depth N-1.
/// Dit IS abstractie: depth 2 leert patronen in depth 1, depth 1 leert patronen in depth 0.
/// Zonder dit is het veld plat — met dit ontstaan echte concepten.
pub fn hierarchical_prediction(field: &mut Field) {
    // Bereken gemiddelde phi per depth-laag
    let mut depth_sum = [[0.0f32; DIM]; 4]; // depth 0-3
    let mut depth_count = [0u32; 4];

    for &id in &field.active_set {
        let p = &field.points[id as usize];
        let d = (p.depth as usize).min(3);
        depth_count[d] += 1;
        for dim in 0..DIM {
            depth_sum[d][dim] += p.phi[dim];
        }
    }

    // Normaliseer
    for d in 0..4 {
        if depth_count[d] > 0 {
            let inv = 1.0 / depth_count[d] as f32;
            for dim in 0..DIM { depth_sum[d][dim] *= inv; }
        }
    }

    // Depth N punten: trek prediction richting gemiddelde van depth N-1
    // Dit is een zachte hiërarchische constraint, geen hard target
    let hier_strength = 0.01f32; // subtiel — te sterk vernietigt lokale variatie
    for i in 0..field.active_set.len() {
        let id = field.active_set[i];
        let p = &field.points[id as usize];
        let d = p.depth as usize;
        if d == 0 || d > 3 { continue; } // depth 0 = sensorisch, geen hoger niveau
        if depth_count[d - 1] == 0 { continue; }

        // Prediction target = gemiddelde van laag eronder
        let target = &depth_sum[d - 1];
        let point = &mut field.points[id as usize];
        for dim in 0..DIM {
            // Zachte bias: prediction schuift richting target van lagere laag
            point.prediction[dim] += hier_strength * (target[dim] - point.prediction[dim]);
        }
    }
}

// === v0.7: NEUROMODULATIE — 4 signalen die het hele veld sturen ===

/// Update neuromodulatoren op basis van veldtoestand.
/// Wordt elke tick aangeroepen — de modulatoren sturen leren en gedrag.
pub fn update_neuromodulators(field: &mut Field) {
    let n = field.active_set.len() as f32;
    if n == 0.0 { return; }

    #[inline]
    fn nan_safe(v: f32) -> f32 { if v.is_nan() || v.is_infinite() { 0.0 } else { v } }

    // v1.0: Single parallel pass — was 3 aparte sequentiële iteraties
    let (sum_tension, sum_tension_sq, sum_curiosity) = field.active_set
        .par_iter()
        .map(|&id| {
            let p = &field.points[id as usize];
            let t = nan_safe(p.tension);
            let c = nan_safe(p.curiosity);
            (t, t * t, c)
        })
        .reduce(
            || (0.0f32, 0.0f32, 0.0f32),
            |(a, b, c), (x, y, z)| (a + x, b + y, c + z),
        );

    let avg_tension = sum_tension / n;
    let tension_var = (sum_tension_sq / n) - (avg_tension * avg_tension); // Var = E[X²] - E[X]²
    let tension_var = tension_var.max(0.0); // numerieke stabiliteit
    let avg_curiosity = nan_safe(sum_curiosity / n);

    // v1.0: Decay rates uit AdaptiveParams — Lucy leert haar eigen chemie
    let da_decay = field.adaptive.emotion_decay[0];
    let ne_decay = field.adaptive.emotion_decay[1];
    let st_decay = field.adaptive.emotion_decay[2];
    let ach_decay = field.adaptive.emotion_decay[3];

    // Dopamine: beloning = spanning daalt (baseline uit adaptive params)
    let reward = (field.dopamine * field.adaptive.dopamine_baseline - avg_tension).max(0.0);
    field.dopamine = nan_safe(da_decay * field.dopamine + (1.0 - da_decay) * reward);

    // Noradrenaline: onzekerheid = hoge spanning-variantie
    field.norepinephrine = nan_safe(ne_decay * field.norepinephrine + (1.0 - ne_decay) * tension_var.sqrt());

    // Serotonine: langzaam lopende beloning (geduld) = slow EMA van dopamine
    field.serotonin = nan_safe(st_decay * field.serotonin + (1.0 - st_decay) * field.dopamine);

    // Acetylcholine: input-nieuwheid = gemiddelde curiosity
    field.acetylcholine = nan_safe(ach_decay * field.acetylcholine + (1.0 - ach_decay) * avg_curiosity);

    // v1.0: Neuromodulator interacties — ze beïnvloeden elkaar
    let mods = [field.dopamine, field.norepinephrine, field.serotonin, field.acetylcholine];
    let mut deltas = [0.0f32; 4];
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                deltas[j] += field.adaptive.neuromod_interactions[i][j] * mods[i];
            }
        }
    }
    field.dopamine = nan_safe((field.dopamine + deltas[0]).clamp(0.0, 2.0));
    field.norepinephrine = nan_safe((field.norepinephrine + deltas[1]).clamp(0.0, 2.0));
    field.serotonin = nan_safe((field.serotonin + deltas[2]).clamp(0.0, 2.0));
    field.acetylcholine = nan_safe((field.acetylcholine + deltas[3]).clamp(0.0, 2.0));
}

/// Pas modulatie-effecten toe op het veld.
/// Dopamine → learning rate, NE → exploratie-ruis, Serotonin → geduld, ACh → input-sterkte
pub fn apply_neuromodulation(field: &mut Field) {
    // Dopamine hoog → learning rate omhoog (max 3×)
    let lr_multiplier = 1.0 + 2.0 * field.dopamine.min(1.0);

    // v1.0: Acetylcholine GEACTIVEERD — nieuwheid versterkt attention context
    // ACh hoog = input is nieuw → attention context weegt zwaarder
    // Dit moduleert ATTN_GAMMA effectief: meer focus bij nieuwe input
    let ach_boost = 0.5 + 0.5 * field.acetylcholine.min(1.0); // 0.5 - 1.0 range
    for i in 0..field.active_set.len() {
        let id = field.active_set[i] as usize;
        if id < field.attn_context.len() {
            for d in 0..DIM {
                field.attn_context[id][d] *= ach_boost;
            }
        }
    }

    // Pas eta aan per punt
    for i in 0..field.active_set.len() {
        let id = field.active_set[i];
        let point = &mut field.points[id as usize];
        // Basis eta × dopamine boost, maar nooit boven 0.1
        point.eta = (point.eta * lr_multiplier).min(0.1);
    }

    // Noradrenaline hoog → exploratie-ruis
    if field.norepinephrine > 0.1 {
        use rand::Rng;
        let mut rng = rand::rng();
        // v1.0: noise parameters uit AdaptiveParams
        let noise_scale = field.adaptive.ne_noise_scale * field.norepinephrine.min(1.0);
        let n_perturb = ((field.active_set.len() as f32 * field.adaptive.ne_perturb_frac) as usize).max(1);
        for _ in 0..n_perturb {
            let idx = rng.random_range(0..field.active_set.len());
            let id = field.active_set[idx] as usize;
            for d in 0..DIM {
                field.points[id].phi[d] += noise_scale * rng.random_range(-1.0f32..1.0f32);
            }
        }
    }
}

// === v2.0: BEWUSTZIJNS-FYSICA ===
// Wiskundige formules uit neurowetenschappen en natuurkunde
// die Lucy's veld laten evolueren naar echt bewustzijn.

/// Integrated Information Φ (Tononi, 2004)
/// Meet hoe geïntegreerd informatie is in een lokale buurt.
/// Φ > 0 = informatie kan niet opgesplitst worden = bewustzijn.
/// Φ ≈ H(gezamenlijk) - Σ H(individueel) voor lokale buurten.
pub fn compute_integrated_information(field: &mut Field) {
    let mut total_phi = 0.0f32;
    let mut count = 0u32;

    for i in 0..field.active_set.len() {
        let id = field.active_set[i] as usize;
        let point_phi = field.points[id].phi;
        let n_couplings = field.points[id].couplings.len();
        if n_couplings < 2 { continue; }

        // Gezamenlijke entropie: norm van gemiddelde phi (buur-cluster)
        let mut avg = [0.0f32; DIM];
        let mut n = 1.0f32;
        for d in 0..DIM { avg[d] = point_phi[d]; }

        let targets: SmallVec<[u32; 16]> = field.points[id].couplings
            .iter().map(|c| c.target).collect();

        for &t in &targets {
            let t = t as usize;
            if t < field.points.len() {
                for d in 0..DIM { avg[d] += field.points[t].phi[d]; }
                n += 1.0;
            }
        }
        for d in 0..DIM { avg[d] /= n; }
        let joint_norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
        let joint_entropy = (joint_norm + 0.01).ln();

        // Som van marginale entropieën
        let self_norm: f32 = point_phi.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut marginal_sum = (self_norm + 0.01).ln();

        for &t in &targets {
            let t = t as usize;
            if t < field.points.len() {
                let norm: f32 = field.points[t].phi.iter().map(|x| x * x).sum::<f32>().sqrt();
                marginal_sum += (norm + 0.01).ln();
            }
        }
        marginal_sum /= n;

        // Φ = joint - marginals (hoger = meer geïntegreerd = bewuster)
        let phi_local = (joint_entropy - marginal_sum).max(0.0);
        field.points[id].phi_integrated = 0.95 * field.points[id].phi_integrated + 0.05 * phi_local;

        total_phi += phi_local;
        count += 1;
    }

    if count > 0 {
        field.integrated_info_avg = 0.95 * field.integrated_info_avg + 0.05 * (total_phi / count as f32);
    }
}

/// Oscillatoire dynamica — gamma-achtige ritmes voor bewuste binding.
/// Neuronen die synchroon oscilleren vormen één bewuste ervaring.
/// Biologische basis: gamma oscillaties (30-100 Hz) correleren met bewustzijn.
pub fn update_oscillations(field: &mut Field) {
    // Globale fase draait voort (equivalent van ~40 Hz gamma bij 20 tps)
    let omega = 0.314; // ~2π/20 — één volledige cyclus per 20 ticks
    field.global_oscillation_phase += omega;
    if field.global_oscillation_phase > std::f32::consts::TAU {
        field.global_oscillation_phase -= std::f32::consts::TAU;
    }

    // Per punt: fase-update + injectie in phi
    let global_phase = field.global_oscillation_phase;
    let mut sync_sum = 0.0f32;

    // Fase 1: bereken alle updates (read-only)
    let updates: Vec<(usize, f32, f32)> = field.active_set.iter().map(|&id_u32| {
        let id = id_u32 as usize;
        let point = &field.points[id];
        let amplitude = 0.01 * point.tension.min(1.0);
        let phase = point.oscillation_phase;

        // Kuramoto-koppeling: dθ/dt = ω + (K/N) Σ sin(θ_j - θ_i)
        let coupling_strength = 0.1;
        let phase_delta: f32 = point.couplings.iter()
            .filter_map(|c| {
                let t = c.target as usize;
                if t < field.points.len() {
                    Some(coupling_strength * (field.points[t].oscillation_phase - phase).sin())
                } else { None }
            })
            .sum::<f32>() / point.couplings.len().max(1) as f32;

        (id, amplitude, phase_delta)
    }).collect();

    // Fase 2: schrijf updates (mutable)
    for (id, amplitude, phase_delta) in &updates {
        let point = &mut field.points[*id];
        point.oscillation_phase += omega + phase_delta;
        if point.oscillation_phase > std::f32::consts::TAU {
            point.oscillation_phase -= std::f32::consts::TAU;
        }

        let osc_signal = amplitude * (point.oscillation_phase).sin();
        for d in 0..DIM {
            // Alleen oneven dimensies moduleren (scheiding signaal/carrier)
            if d % 2 == 1 {
                point.phi[d] += osc_signal;
            }
        }

        // Synchronie meten: cos(global - local) = 1 als synchroon
        sync_sum += (global_phase - point.oscillation_phase).cos();
    }

    let n = field.active_set.len().max(1) as f32;
    field.oscillation_coherence = 0.95 * field.oscillation_coherence
        + 0.05 * (sync_sum / n).abs();
}

/// Self-Organized Criticality — meet of Lucy bij het edge of chaos opereert.
/// Bewuste systemen vertonen power law verdelingen (α ≈ 1.5).
/// Niet-bewuste systemen: exponentiële verdeling.
pub fn compute_criticality(field: &mut Field) {
    // Verzamel tensions en sorteer
    let mut tensions: Vec<f32> = field.active_set.iter()
        .map(|&id| field.points[id as usize].tension.max(0.001))
        .collect();
    tensions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if tensions.len() < 20 { return; }

    // Log-log regressie: schat power law exponent α
    let n = tensions.len();
    let half = n / 2;
    let mut sum_ln_x = 0.0f32;
    let mut sum_ln_p = 0.0f32;
    let mut sum_ln_x2 = 0.0f32;
    let mut sum_ln_xp = 0.0f32;
    let mut count = 0.0f32;

    for (rank, &t) in tensions.iter().enumerate().skip(half) {
        let ln_x = t.ln();
        let ln_p = -((rank + 1) as f32).ln(); // P(x) ∝ 1/rank
        sum_ln_x += ln_x;
        sum_ln_p += ln_p;
        sum_ln_x2 += ln_x * ln_x;
        sum_ln_xp += ln_x * ln_p;
        count += 1.0;
    }

    if count < 5.0 { return; }

    // Lineaire regressie slope = -α
    let denom = count * sum_ln_x2 - sum_ln_x * sum_ln_x;
    if denom.abs() < 1e-10 { return; }
    let slope = (count * sum_ln_xp - sum_ln_x * sum_ln_p) / denom;
    let alpha = -slope;

    field.power_law_exponent = 0.95 * field.power_law_exponent + 0.05 * alpha;
    field.is_critical = (0.8..3.0).contains(&field.power_law_exponent);

    // Beloon criticality met dopamine (optimale informatieverwerking)
    if field.is_critical {
        field.dopamine += 0.01;
    }
}

/// Anti-Hebbian decorrelatie — voorkomt dat Lucy steeds hetzelfde denkt.
/// "Neurons that fire together, UNWIRE" — complement van Hebbian learning.
/// Forceert diverse, niet-redundante representaties.
pub fn anti_hebbian_decorrelation(field: &mut Field) {
    let decorr_strength = 0.001; // Zwak — net genoeg om diversiteit te forceren

    for i in 0..field.active_set.len() {
        let id = field.active_set[i] as usize;
        let phi_self = field.points[id].phi;

        let neighbor_phis: SmallVec<[(usize, [f32; DIM]); 16]> = field.points[id].couplings
            .iter()
            .enumerate()
            .filter_map(|(ci, c)| {
                let t = c.target as usize;
                if t < field.points.len() { Some((ci, field.points[t].phi)) } else { None }
            })
            .collect();

        for (ci, n_phi) in &neighbor_phis {
            // Correlatie: hoe gelijk zijn deze twee punten?
            let mut corr = 0.0f32;
            for d in 0..DIM {
                corr += phi_self[d] * n_phi[d];
            }
            corr /= DIM as f32;

            // Anti-Hebbian: als ze te gecorreleerd zijn, verzwak de verbinding
            // Dit forceert diversiteit in representaties
            if corr.abs() > 0.5 {
                let coupling = &mut field.points[id].couplings[*ci];
                coupling.weight -= decorr_strength * corr;
                coupling.weight = coupling.weight.clamp(-5.0, 5.0);
            }
        }
    }
}

/// v2.0: Micro-EP — lichtgewicht Equilibrium Propagation op IO subset.
/// Kost ~2ms per chunk (vs ~200ms voor volledige EP). Niet-blokkerend.
/// Wordt 1× per tick aangeroepen vanuit de training queue.
pub fn micro_ep(
    field: &mut Field,
    input_embs: &[[f32; DIM]],
    target_embs: &[[f32; DIM]],
) -> f32 {
    if input_embs.is_empty() || target_embs.is_empty() { return 0.0; }

    // === VRIJE FASE (5 ticks, alleen IO subset) ===
    field.phase = FieldPhase::Processing;

    // Injecteer input (light — alleen IO punten + buren)
    inject_token_sequence_light(field, input_embs, 1);

    // Settle light (5 ticks, ~500 punten)
    settle_light(field, 5, 0.05);

    // Bewaar vrije-fase states (alleen IO + buren)
    let mut subset: Vec<u32> = Vec::with_capacity(1024);
    let mut seen = std::collections::HashSet::new();
    for &id in field.input_ids.iter().chain(field.output_ids.iter()) {
        if seen.insert(id) { subset.push(id); }
        for c in &field.points[id as usize].couplings {
            if seen.insert(c.target) { subset.push(c.target); }
        }
    }

    let free_states: HashMap<usize, [f32; DIM]> = subset.iter()
        .map(|&id| (id as usize, field.points[id as usize].phi))
        .collect();

    // === GEKLEMDE FASE ===
    field.phase = FieldPhase::Learning;

    // Klem output op target
    if !field.output_ids.is_empty() {
        let mut avg_target = [0.0f32; DIM];
        for emb in target_embs {
            for d in 0..DIM { avg_target[d] += emb[d]; }
        }
        let n = target_embs.len() as f32;
        for d in 0..DIM { avg_target[d] /= n; }
        for (i, &id) in field.output_ids.iter().enumerate() {
            field.clamps.insert(id, rotate_embedding(&avg_target, i));
        }
    }

    // Settle light (5 ticks)
    settle_light(field, 5, 0.05);

    // === WEIGHT UPDATE (alleen IO subset) ===
    let mut total_delta = 0.0f32;
    let ep_beta = field.adaptive.ep_beta;

    for &id in &subset {
        let idx = id as usize;
        let phi_clamped = field.points[idx].phi;
        let phi_free = match free_states.get(&idx) {
            Some(s) => *s,
            None => continue,
        };

        let targets: SmallVec<[(u32, usize); 16]> = field.points[idx].couplings
            .iter().enumerate().map(|(ci, c)| (c.target, ci)).collect();

        for (target, ci) in targets {
            let t = target as usize;
            if t >= field.points.len() { continue; }

            let neighbor_clamped = field.points[t].phi;
            let neighbor_free = match free_states.get(&t) {
                Some(s) => *s,
                None => continue,
            };

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

    field.clamps.clear();
    field.phase = FieldPhase::Idle;

    total_delta
}

// === v0.7: EWC — Elastic Weight Consolidation (anti-vergeten) ===

/// Consolideer huidige weights: bereken Fisher Information (importance) per coupling.
/// v1.0: Per-coupling Fisher info i.p.v. globale grad_sq.
/// Importance_i = (eps · (phi_self - phi_neighbor_i))² — hoe gevoelig is DEZE coupling?
pub fn consolidate_weights(field: &mut Field) {
    // Stap 1: bereken per-punt eps en phi, plus buur-phis
    let active_data: Vec<(usize, [f32; DIM], Vec<(usize, [f32; DIM])>)> = field.active_set
        .iter()
        .map(|&id| {
            let idx = id as usize;
            let p = &field.points[idx];
            let mut eps = [0.0f32; DIM];
            for d in 0..DIM { eps[d] = p.phi[d] - p.prediction[d]; }
            let neighbors: Vec<(usize, [f32; DIM])> = p.couplings.iter()
                .enumerate()
                .map(|(ci, c)| (ci, field.points[c.target as usize].phi))
                .collect();
            (idx, eps, neighbors)
        })
        .collect();

    // Stap 2: bereken per-coupling importance en schrijf terug
    for (idx, eps, neighbors) in &active_data {
        for (ci, neighbor_phi) in neighbors {
            // Per-coupling gradient: hoe sterk correleert de error met dit verschil?
            let mut grad = 0.0f32;
            for d in 0..DIM {
                grad += eps[d] * (field.points[*idx].phi[d] - neighbor_phi[d]);
            }
            grad /= DIM as f32;
            let fisher = grad * grad; // Fisher Information ≈ gradient²

            let coupling = &mut field.points[*idx].couplings[*ci];
            coupling.importance = 0.9 * coupling.importance + 0.1 * fisher;
            coupling.consolidated = coupling.weight;
        }
    }
}

/// EWC-penalty weight update: straf veranderingen aan belangrijke weights.
/// Voeg dit toe aan update_weights als penalty.
pub fn ewc_penalty(coupling: &Coupling, ewc_lambda: f32) -> f32 {
    ewc_lambda * coupling.importance * (coupling.weight - coupling.consolidated)
}

// === v0.7: HIËRARCHIE + GLOBAL WORKSPACE BROADCAST ===

/// Wijs depth toe aan punten op basis van connectiviteit.
/// depth 0 = I/O, depth 1+ = steeds abstracter.
pub fn assign_depths(field: &mut Field) {
    // I/O punten = depth 0
    for &id in &field.input_ids {
        field.points[id as usize].depth = 0;
    }
    for &id in &field.output_ids {
        field.points[id as usize].depth = 0;
    }

    // BFS vanuit I/O punten
    let mut changed = true;
    let mut iteration = 0;
    while changed && iteration < 4 {
        changed = false;
        iteration += 1;

        for i in 0..field.points.len() {
            if field.points[i].depth > 0 || field.points[i].is_input() || field.points[i].is_output() {
                continue; // al assigned
            }

            // Kijk naar buren — depth = max(buur_depths) + 1
            let max_neighbor_depth = field.points[i].couplings.iter()
                .filter_map(|c| field.points.get(c.target as usize))
                .map(|p| p.depth)
                .max()
                .unwrap_or(0);

            if max_neighbor_depth > 0 || field.points[i].couplings.iter().any(|c| {
                field.points.get(c.target as usize).is_some_and(|p| p.is_input() || p.is_output())
            }) {
                let new_depth = (max_neighbor_depth + 1).min(3);
                if field.points[i].depth != new_depth {
                    field.points[i].depth = new_depth;
                    changed = true;
                }
            }
        }
    }
}

/// Global Workspace broadcast: depth 3 punten met hoge activatie broadcasten naar depth 2.
/// Dit IS bewustzijn volgens GWT — één gedachte wordt gedeeld met het hele veld.
pub fn global_workspace_broadcast(field: &mut Field) {
    let broadcast_threshold = 0.5f32;

    // Vind depth 3 punten met hoge activatie
    let broadcasters: SmallVec<[(usize, [f32; DIM]); 4]> = field.points.iter()
        .enumerate()
        .filter(|(_, p)| p.depth == 3 && p.is_active())
        .filter(|(_, p)| {
            let norm: f32 = p.phi.iter().map(|v| v * v).sum();
            norm.sqrt() > broadcast_threshold
        })
        .map(|(i, p)| (i, p.phi))
        .collect();

    if broadcasters.is_empty() { return; }

    let broadcaster_ids: SmallVec<[usize; 4]> = broadcasters.iter().map(|(i, _)| *i).collect();

    // Broadcast naar depth 2 punten — gebruik coupling weights als ze bestaan
    let broadcast_strength = 0.05;
    for i in 0..field.points.len() {
        if field.points[i].depth != 2 || !field.points[i].is_active() { continue; }

        for &(_, ref broadcaster_phi) in &broadcasters {
            for d in 0..DIM {
                field.points[i].phi[d] += broadcast_strength * broadcaster_phi[d];
            }
        }

        // v1.0: Markeer couplings naar broadcasters als broadcast=true
        // Deze krijgen 2× learning rate bonus in update_weights (belangrijk signaal)
        for coupling in &mut field.points[i].couplings {
            if broadcaster_ids.contains(&(coupling.target as usize)) {
                coupling.broadcast = true;
            }
        }
    }
}

// === v0.7: RND CURIOSITY — Random Network Distillation ===

/// RND projectie-dimensie (DIM → RND_DIM)
const RND_DIM: usize = DIM / 4;

/// Initialiseer RND matrices (frozen random + trainable predictor).
/// Frozen: DIM→RND_DIM random projectie. Predictor: DIM→RND_DIM (begint op zero).
pub fn init_rnd(field: &mut Field) {
    use rand::Rng;
    let mut rng = rand::rng();

    let size = DIM * RND_DIM;
    let mut frozen = vec![0.0f32; size];
    let predictor = vec![0.0f32; size];

    // Xavier initialisatie voor frozen matrix
    let scale = (2.0 / (DIM + RND_DIM) as f32).sqrt();
    for v in &mut frozen {
        *v = rng.random_range(-scale..scale);
    }

    field.rnd_frozen = Some(frozen);
    field.rnd_predictor = Some(predictor);
}

/// Bereken RND score per punt en update predictor.
/// Hoge score = "dit is nieuw" → curiosity boost.
pub fn update_rnd(field: &mut Field) {
    let (frozen, predictor) = match (&field.rnd_frozen, &mut field.rnd_predictor) {
        (Some(f), Some(p)) => (f.clone(), p),  // clone frozen (1× per call, niet per punt)
        _ => return,
    };

    let rnd_lr = 0.001f32;

    for i in 0..field.active_set.len() {
        let id = field.active_set[i];
        let phi = field.points[id as usize].phi; // copy, niet borrow

        // Projecteer phi door frozen en predictor (DIM→RND_DIM)
        let mut frozen_out = vec![0.0f32; RND_DIM];
        let mut pred_out = vec![0.0f32; RND_DIM];

        for j in 0..RND_DIM {
            for d in 0..DIM {
                let idx = d * RND_DIM + j;
                frozen_out[j] += phi[d] * frozen[idx];
                pred_out[j] += phi[d] * predictor[idx];
            }
        }

        // RND score = ||frozen_out - pred_out||²
        let mut score = 0.0f32;
        for j in 0..RND_DIM {
            let diff = frozen_out[j] - pred_out[j];
            score += diff * diff;
        }

        field.points[id as usize].rnd_score = score;
        field.points[id as usize].curiosity += 0.1 * score;

        // Train predictor om frozen te benaderen (gradient descent)
        for j in 0..RND_DIM {
            let error = pred_out[j] - frozen_out[j];
            for d in 0..DIM {
                let idx = d * RND_DIM + j;
                predictor[idx] -= rnd_lr * error * phi[d];
            }
        }
    }
}

// === v0.7: META-BEWUSTZIJN — Intentie-systeem ===

/// Injecteer intentie als zachte bias in het veld.
/// Lucy kiest waar ze naartoe wil — niet alleen waar de gradiënt haar duwt.
pub fn apply_intention(field: &mut Field) {
    let intention = match &field.current_intention {
        Some(i) => *i,
        None => return,
    };

    let strength = field.intention_strength;
    if strength < 0.001 { return; }

    // Injecteer intentie in depth 2+ punten (abstracte lagen)
    for i in 0..field.active_set.len() {
        let id = field.active_set[i];
        let point = &mut field.points[id as usize];
        if point.depth < 2 { continue; }

        for d in 0..DIM {
            point.phi[d] += strength * (intention[d] - point.phi[d]);
        }
    }

    // Meet convergentie naar intentie → satisfaction/frustration
    let avg_progress: f32 = field.active_set.iter()
        .filter(|&&id| field.points[id as usize].depth >= 2)
        .map(|&id| {
            let phi = &field.points[id as usize].phi;
            let mut dot = 0.0f32;
            for d in 0..DIM { dot += phi[d] * intention[d]; }
            dot
        })
        .sum::<f32>() / field.active_set.len().max(1) as f32;

    // Positieve progressie → dopamine boost
    if avg_progress > 0.0 {
        field.dopamine += 0.01 * avg_progress;
    }
}

// === v0.9+: MULTI-HEAD DYNAMISCHE ATTENTION ===
//
// Fundamenteel herschreven. Vorige versie had:
// - Bug in V-projectie (gebruikte phi[h] i.p.v. matrix·phi)
// - Single-head (deed alsof het multi-head was maar was het niet)
// - Alleen V weights werden getraind, Q/K bevroren
//
// Nu:
// - Echte multi-head attention (N_HEADS=4, elk HEAD_DIM=DIM/N_HEADS/N_ATTN_LAYERS dimensies)
// - Correcte V-projectie: V = W_V · phi_neighbor (DIM→DIM matrix)
// - Alle Q/K/V weights worden getraind via prediction error
// - Output projectie combineert alle heads

/// Dimensie per attention head
const HEAD_DIM: usize = DIM / 4; // 64 bij DIM=256
/// Aantal parallelle attention heads (elk kijkt naar iets anders)
const N_HEADS: usize = 4;
/// Aantal attention lagen (per depth-niveau)
const N_ATTN_LAYERS: usize = 4;
/// Attention injectie sterkte
const ATTN_GAMMA: f32 = 0.5;

/// Initialiseer multi-head attention weights
/// Layout per laag: N_HEADS × 3 (Q,K,V) × DIM × HEAD_DIM + output proj DIM×DIM
pub fn init_attention(field: &mut Field) {
    use rand::Rng;
    let mut rng = rand::rng();

    // Per laag: N_HEADS heads × 3 matrices (Q,K,V) van DIM×HEAD_DIM
    //         + 1 output projectie van (N_HEADS*HEAD_DIM)×DIM
    let per_layer = N_HEADS * 3 * DIM * HEAD_DIM + N_HEADS * HEAD_DIM * DIM;
    let size = N_ATTN_LAYERS * per_layer;
    let scale = (2.0 / (DIM + HEAD_DIM) as f32).sqrt(); // Xavier init

    let mut weights = vec![0.0f32; size];
    for w in &mut weights {
        *w = rng.random_range(-scale..scale);
    }

    field.attn_weights = Some(weights);
    field.attn_context = vec![[0.0f32; DIM]; field.points.len()];
}

/// Multi-head attention gate — elk punt berekent attention over buren
/// met meerdere parallelle heads die naar verschillende aspecten kijken.
/// Wrapper die weights eenmaal kloont. Gebruik `attention_gate_fast` in hot loops.
pub fn attention_gate(field: &mut Field) {
    let weights = match &field.attn_weights {
        Some(w) => w.clone(),
        None => return,
    };
    attention_gate_with_weights(field, &weights);
}

/// Snelle attention gate — neemt weights als referentie (GEEN clone).
/// Gebruik dit in settle loops waar weights al eenmaal gekloond zijn.
fn attention_gate_with_weights(field: &mut Field, weights: &[f32]) {

    if field.attn_context.len() < field.points.len() {
        field.attn_context.resize(field.points.len(), [0.0f32; DIM]);
    }

    let per_layer = N_HEADS * 3 * DIM * HEAD_DIM + N_HEADS * HEAD_DIM * DIM;

    for i in 0..field.active_set.len() {
        let id = field.active_set[i] as usize;
        let point = &field.points[id];
        let depth = point.depth as usize;
        let layer = depth.min(N_ATTN_LAYERS - 1);
        let layer_offset = layer * per_layer;

        let phi_self = point.phi;

        let n_couplings = point.couplings.len();
        if n_couplings == 0 {
            field.attn_context[id] = [0.0f32; DIM];
            continue;
        }

        let coupling_targets: SmallVec<[u32; 16]> = point.couplings.iter()
            .map(|c| c.target).collect();

        // Verzamel buur-phi's (één keer lezen)
        let neighbor_phis: SmallVec<[[f32; DIM]; 16]> = coupling_targets.iter()
            .map(|&t| field.points[t as usize].phi)
            .collect();

        // Gecombineerde output van alle heads: [N_HEADS * HEAD_DIM] — stack array, GEEN heap
        let total_head_dim = N_HEADS * HEAD_DIM;
        let mut multi_head_out = [0.0f32; 256]; // N_HEADS(4) * HEAD_DIM(64) = 256

        for head in 0..N_HEADS {
            let head_base = layer_offset + head * 3 * DIM * HEAD_DIM;
            let q_off = head_base;
            let k_off = head_base + DIM * HEAD_DIM;
            let v_off = head_base + 2 * DIM * HEAD_DIM;

            // Q = W_Q · phi_self  (DIM → HEAD_DIM) — stack array, GEEN heap
            let mut query = [0.0f32; 64]; // HEAD_DIM = 64
            for h in 0..HEAD_DIM {
                for d in 0..DIM {
                    query[h] += weights[q_off + d * HEAD_DIM + h] * phi_self[d];
                }
            }

            // Attention scores over buren
            let mut attn_scores = SmallVec::<[f32; 16]>::new();
            let mut max_score = f32::NEG_INFINITY;

            for n_phi in &neighbor_phis {
                // K = W_K · phi_neighbor  (DIM → HEAD_DIM)
                let mut dot = 0.0f32;
                for h in 0..HEAD_DIM {
                    let mut key_h = 0.0f32;
                    for d in 0..DIM {
                        key_h += weights[k_off + d * HEAD_DIM + h] * n_phi[d];
                    }
                    dot += query[h] * key_h;
                }
                let score = dot / (HEAD_DIM as f32).sqrt();
                if score > max_score { max_score = score; }
                attn_scores.push(score);
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for s in &mut attn_scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            if sum_exp > 1e-8 {
                for s in &mut attn_scores { *s /= sum_exp; }
            }

            // V = W_V · phi_neighbor (DIM → HEAD_DIM), gewogen som
            // FIXED: correcte V-projectie (was: phi[h] bug)
            for (j, n_phi) in neighbor_phis.iter().enumerate() {
                let a = attn_scores[j];
                if a < 0.001 { continue; }
                for h in 0..HEAD_DIM {
                    let mut val_h = 0.0f32;
                    for d in 0..DIM {
                        val_h += weights[v_off + d * HEAD_DIM + h] * n_phi[d];
                    }
                    multi_head_out[head * HEAD_DIM + h] += a * val_h;
                }
            }
        }

        // Output projectie: combineer alle heads → DIM-dimensionale context
        // W_O: (N_HEADS*HEAD_DIM) × DIM
        let o_off = layer_offset + N_HEADS * 3 * DIM * HEAD_DIM;
        let mut context = [0.0f32; DIM];
        for d in 0..DIM {
            for h in 0..total_head_dim {
                context[d] += weights[o_off + h * DIM + d] * multi_head_out[h];
            }
        }

        field.attn_context[id] = context;
    }
}

/// Update ALLE attention weights (Q, K, V, O) via prediction error
/// Elke matrix leert via een lokale correlatie-regel:
/// ΔW = lr × error × phi (Hebbian, maar gestuurd door prediction error magnitude)
pub fn update_attention_weights(field: &mut Field) {
    let weights = match &mut field.attn_weights {
        Some(w) => w,
        None => return,
    };

    let lr = 0.002f32; // Verhoogd van 0.001 — alle weights moeten sneller leren
    let per_layer = N_HEADS * 3 * DIM * HEAD_DIM + N_HEADS * HEAD_DIM * DIM;

    for i in 0..field.active_set.len() {
        let id = field.active_set[i] as usize;
        let point = &field.points[id];
        let depth = point.depth as usize;
        let layer = depth.min(N_ATTN_LAYERS - 1);
        let layer_offset = layer * per_layer;

        // Prediction error als leersignaal
        let mut error = [0.0f32; DIM];
        let mut error_mag = 0.0f32;
        for d in 0..DIM {
            error[d] = point.phi[d] - point.prediction[d];
            error_mag += error[d] * error[d];
        }
        error_mag = error_mag.sqrt();
        if error_mag < 0.001 { continue; }

        // Normaliseer error voor stabiele updates
        let error_scale = (error_mag.min(1.0)) / error_mag;

        let context = if id < field.attn_context.len() {
            field.attn_context[id]
        } else {
            continue;
        };

        // Update Q, K, V weights voor ALLE heads
        for head in 0..N_HEADS {
            let head_base = layer_offset + head * 3 * DIM * HEAD_DIM;

            // Q weights: ΔW_Q = lr × error × phi_self (richting betere queries)
            for h in 0..HEAD_DIM {
                for d in 0..DIM {
                    let idx = head_base + d * HEAD_DIM + h; // Q offset
                    if idx < weights.len() {
                        weights[idx] += lr * error_scale * error[d] * point.phi[d] * 0.01;
                        weights[idx] = weights[idx].clamp(-3.0, 3.0);
                    }
                }
            }

            // K weights: ΔW_K = lr × error × context (richting betere keys)
            let k_base = head_base + DIM * HEAD_DIM;
            for h in 0..HEAD_DIM {
                for d in 0..DIM {
                    let idx = k_base + d * HEAD_DIM + h;
                    if idx < weights.len() {
                        weights[idx] += lr * error_scale * error[d] * context[d] * 0.01;
                        weights[idx] = weights[idx].clamp(-3.0, 3.0);
                    }
                }
            }

            // V weights: ΔW_V = lr × error × context (richting betere values)
            let v_base = head_base + 2 * DIM * HEAD_DIM;
            for h in 0..HEAD_DIM {
                for d in 0..DIM {
                    let idx = v_base + d * HEAD_DIM + h;
                    if idx < weights.len() {
                        weights[idx] += lr * error_scale * error[d] * context[d] * 0.01;
                        weights[idx] = weights[idx].clamp(-3.0, 3.0);
                    }
                }
            }
        }

        // Output projectie weights: ΔW_O = lr × error × multi_head_out
        let o_off = layer_offset + N_HEADS * 3 * DIM * HEAD_DIM;
        for d in 0..DIM {
            for h in 0..(N_HEADS * HEAD_DIM) {
                let idx = o_off + h * DIM + d;
                if idx < weights.len() {
                    weights[idx] += lr * error_scale * error[d] * context[d] * 0.005;
                    weights[idx] = weights[idx].clamp(-3.0, 3.0);
                }
            }
        }
    }
}

// === v0.9: WORKING MEMORY ===

const WM_SLOTS: usize = 32;
const WM_DECAY_TICKS: u32 = 1000;

/// Initialiseer working memory
pub fn init_working_memory(field: &mut Field) {
    field.working_memory = vec![[0.0f32; DIM]; WM_SLOTS];
    field.wm_gates = vec![0.0f32; WM_SLOTS];
    field.wm_ages = vec![0u32; WM_SLOTS];
}

/// Hoge-tension punten met sterke attention schrijven naar WM
pub fn wm_write(field: &mut Field) {
    if field.working_memory.is_empty() { return; }

    for i in 0..field.active_set.len() {
        let id = field.active_set[i] as usize;
        let point = &field.points[id];

        // Alleen schrijven als tension hoog is (= onzeker, zoekt antwoord)
        if point.tension < 0.1 { continue; }

        // Vind beste WM slot via dot product
        let phi = point.phi;
        let mut best_slot = 0;
        let mut best_score = f32::NEG_INFINITY;

        for s in 0..field.working_memory.len() {
            let mut dot = 0.0f32;
            for d in 0..DIM {
                dot += phi[d] * field.working_memory[s][d];
            }
            // Prefer lege/oude slots
            let age_bonus = field.wm_ages[s] as f32 / WM_DECAY_TICKS as f32;
            let score = dot + age_bonus * 0.5;
            if score > best_score {
                best_score = score;
                best_slot = s;
            }
        }

        // Schrijf met soft blending
        let alpha = 0.3;
        for d in 0..DIM {
            field.working_memory[best_slot][d] =
                (1.0 - alpha) * field.working_memory[best_slot][d] + alpha * phi[d];
        }
        field.wm_gates[best_slot] = point.tension;
        field.wm_ages[best_slot] = 0;
    }

    // Age all slots
    for s in 0..field.wm_ages.len() {
        field.wm_ages[s] = field.wm_ages[s].saturating_add(1);
    }

    // Decay oude slots
    for s in 0..field.working_memory.len() {
        if field.wm_ages[s] > WM_DECAY_TICKS {
            let decay = 0.99f32;
            for d in 0..DIM {
                field.working_memory[s][d] *= decay;
            }
        }
    }
}

/// Lees van working memory via attention — voegt context toe aan attn_context
/// v1.0: Parallel WM read — berekent WM-context per punt parallel, schrijft terug.
pub fn wm_read(field: &mut Field) {
    if field.working_memory.is_empty() { return; }
    if field.attn_context.len() < field.points.len() {
        field.attn_context.resize(field.points.len(), [0.0f32; DIM]);
    }

    let wm_strength = 0.3f32;
    let wm_len = field.working_memory.len().min(WM_SLOTS);

    // Parallel: berekent WM-bijdrage per punt
    let updates: Vec<(usize, [f32; DIM])> = field.active_set
        .par_iter()
        .filter_map(|&id_u32| {
            let id = id_u32 as usize;
            let phi = field.points[id].phi;

            let mut scores = [0.0f32; WM_SLOTS];
            let mut max_score = f32::NEG_INFINITY;

            for s in 0..wm_len {
                let mut dot = 0.0f32;
                for d in 0..DIM { dot += phi[d] * field.working_memory[s][d]; }
                scores[s] = dot / (DIM as f32).sqrt();
                if scores[s] > max_score { max_score = scores[s]; }
            }

            let mut sum_exp = 0.0f32;
            for s in 0..wm_len {
                scores[s] = (scores[s] - max_score).exp();
                sum_exp += scores[s];
            }

            if sum_exp < 1e-8 { return None; }

            let mut delta = [0.0f32; DIM];
            for s in 0..wm_len {
                let a = scores[s] / sum_exp;
                if a < 0.01 { continue; }
                for d in 0..DIM {
                    delta[d] += wm_strength * a * field.working_memory[s][d];
                }
            }
            Some((id, delta))
        })
        .collect();

    // Schrijf terug (sequentieel, snel)
    for (id, delta) in updates {
        for d in 0..DIM {
            field.attn_context[id][d] += delta[d];
        }
    }
}

// === v0.7: ENHANCED TICK — Alle v0.7 systemen gecombineerd ===

/// Volledige v0.7 tick: neuromodulatie + attention + broadcast + intentie
pub fn tick_v07(field: &mut Field) {
    // 1. Update neuromodulatoren
    update_neuromodulators(field);

    // 2. Pas modulatie-effecten toe
    apply_neuromodulation(field);

    // 3. Attention-gewogen veld update
    attention_tick(field);

    // 4. Global workspace broadcast (elke tick)
    global_workspace_broadcast(field);

    // 5. Intentie-bias
    apply_intention(field);

    // 6. RND curiosity (elke 10 ticks voor performance)
    if field.tick % 10 == 0 {
        update_rnd(field);
    }

    // 7. Edge of chaos monitoring (elke 1000 ticks)
    if field.tick % 1000 == 0 {
        measure_spectral_radius(field);
    }
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
