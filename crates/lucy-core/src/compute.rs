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
                    // v2.6.4: skip NaN neighbors — voorkom cascade door heel het veld
                    if !neighbor.phi[0].is_finite() { continue; }
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
        // v2.5: Prediction rate ADAPTIEF — hoge error = langzaam volgen (behoud signaal)
        // Was: vaste fe_tracking_rate (0.05-0.08) → prediction vangt phi in ~20 ticks → error sterft
        // Nu: rate daalt als error hoog is → prediction blijft achter → error blijft bestaan
        let base_pred_rate = field.fe_tracking_rate;
        let point = &mut field.points[id_usize];
        for d in 0..DIM {
            let error_d = (point.phi[d] - point.prediction[d]).abs();
            // Hoge error → lagere tracking rate → prediction blijft achter → error blijft
            // pred_rate = base / (1 + 10 * error) → bij error=0.1: rate = base/2
            let adaptive_rate = base_pred_rate / (1.0 + 10.0 * error_d);
            point.prediction[d] = (1.0 - adaptive_rate) * point.prediction[d] + adaptive_rate * point.phi[d];
            point.phi[d] -= lambda * grad[d];

            if let Some(inp) = input {
                point.phi[d] += beta * inp[d];
            }

            // Attention context injectie (v0.9)
            // v2.6.3: NaN guard — attn_context can contain NaN from stale attention computations
            if let Some(ref ctx) = attn_ctx {
                if ctx[d].is_finite() {
                    point.phi[d] += ATTN_GAMMA * ctx[d];
                }
            }

            // Sparse penalty: L1 — v2.4: CONDITIONEEL
            // Alleen actief als phi significant boven ruis-niveau is.
            // Bij lage tension duwde dit alles naar nul → doodsspiraal.
            if point.phi[d].abs() > field.noise_amplitude {
                point.phi[d] -= alpha * point.phi[d].signum();
            }

            // v2.6.3: Hard clamp — prevent inf/NaN from accumulating
            if !point.phi[d].is_finite() {
                point.phi[d] = 0.0;
            }
            // v2.6.7 A1: Magnitude clamp — voorkomt explosie cascade (phi → phi_var → KL → F → tension → ∞)
            // Was alleen in apply_layer_norm elke 10 ticks — te laat.
            point.phi[d] = point.phi[d].clamp(-field.clamp_range, field.clamp_range);
        }

        // Layer norm verplaatst naar apply_layer_norm() — wordt elke 10 ticks aangeroepen
        // i.p.v. elke tick. Phi drifted niet snel genoeg voor per-tick normalisatie.
    }

    /// Soft Layer Normalization — behoudt signaal-magnitude
    /// v2.5: ZACHTE normalisatie i.p.v. harde mean=0/var=1.
    ///
    /// Het probleem met standaard layer norm: noise met amplitude 7.5 wordt
    /// teruggeschaald naar var=1 → alle perturbaties verdwijnen elke 10 ticks.
    ///
    /// Oplossing: alleen normaliseren als variance BOVEN een drempel is.
    /// Bij lage variance (normaal signaal): niks doen.
    /// Bij hoge variance (explosie): zachte demping richting stabiele range.
    /// Dit behoudt noise-signalen terwijl numerieke explosies voorkomen worden.
    pub fn apply_layer_norm(field: &mut Field) {
        let clamp_range = field.clamp_range;
        // Drempel: alleen normaliseren als std > 2× clamp_range (explosie-preventie)
        // Normaal signaal (std ≈ 1-5) wordt NIET genormaliseerd
        let norm_threshold = clamp_range * 0.5;

        let new_phis: Vec<(usize, Option<[f32; DIM]>)> = field.active_set
            .par_iter()
            .map(|&id| {
                let idx = id as usize;
                let phi = field.points[idx].phi;

                // Welford's: meet mean + std
                let mut mean = 0.0f32;
                let mut m2 = 0.0f32;
                for d in 0..DIM {
                    let delta = phi[d] - mean;
                    mean += delta / (d + 1) as f32;
                    let delta2 = phi[d] - mean;
                    m2 += delta * delta2;
                }
                let std = (m2 / DIM as f32 + 1e-5).sqrt();

                // Alleen normaliseren als std te hoog is (explosie-preventie)
                if std > norm_threshold {
                    // Zachte normalisatie: schaal terug naar std = norm_threshold
                    let scale = norm_threshold / std;
                    let mut normed = [0.0f32; DIM];
                    for d in 0..DIM {
                        normed[d] = ((phi[d] - mean) * scale).clamp(-clamp_range, clamp_range);
                    }
                    (idx, Some(normed))
                } else {
                    // Alleen mean-centeren, variance behouden (noise-safe)
                    // En clamp voor numerieke veiligheid
                    let mut normed = [0.0f32; DIM];
                    for d in 0..DIM {
                        normed[d] = (phi[d] - mean).clamp(-clamp_range, clamp_range);
                    }
                    (idx, Some(normed))
                }
            })
            .collect();

        for (idx, normed) in new_phis {
            if let Some(n) = normed {
                field.points[idx].phi = n;
            }
        }
    }

    /// Update W (koppelingen) via prediction error gradient — GEEN Vec allocatie
    pub fn update_weights(field: &mut Field, id: u32) {
        let mu = field.mu;
        let point = &field.points[id as usize];
        let eps = prediction_error(point);
        let eta = if point.eta.is_finite() { point.eta } else { 0.01 };
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
            dw /= (DIM as f32).sqrt(); // v2.6.5: √DIM ipv DIM — correcte normalisatie (16× sterker signaal)
            // Gradient clipping: voorkom explosive weight updates en NaN propagatie
            if !dw.is_finite() { dw = 0.0; }
            dw = dw.clamp(-2.0, 2.0); // v2.6.5: ruimer dan ±1.0 want gradient is 16× sterker

            coupling.momentum = mu * coupling.momentum + (1.0 - mu) * dw;
            // EWC penalty: bescherm belangrijke weights — sterkte schaalt met kl_scale
            let ewc_pen = ewc_penalty(coupling, field.kl_scale);
            // v1.0: Broadcast couplings krijgen 2× learning rate (dragen bewust signaal)
            let lr = if coupling.broadcast { eta * 2.0 } else { eta };
            coupling.weight -= lr * dw + mu * coupling.momentum + lr * ewc_pen;

            // v2.6: Free Energy-gebaseerd vergeten (Friston precision weighting op synapsen)
            // decay ∝ avg_F / importance — onbelangrijke weights vervagen, belangrijke overleven
            let avg_fe = if field.avg_free_energy.is_finite() { field.avg_free_energy } else { 0.0 };
            let fe_decay = avg_fe / (coupling.importance + 0.001) * 0.0005;
            let fe_decay = if fe_decay.is_finite() { fe_decay.min(0.01) } else { 0.0 }; // max 1% decay per update
            coupling.weight *= 1.0 - fe_decay;

            let w_clamp = 2.5f32; // vaste limiet, onafhankelijk van clamp_range
            coupling.weight = coupling.weight.clamp(-w_clamp, w_clamp);
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

    /// ═══════════════════════════════════════════════════════════════════════
    /// HET SAMENVELD — Lucy's Unified Field Equation
    /// ═══════════════════════════════════════════════════════════════════════
    ///
    /// v2.5: Bayesian Mechanics (Dalton, Sakthivadivel & Friston 2024)
    ///
    /// Elk punt in Lucy's veld is een **Markov Blanket** — een statistische
    /// grens die "binnen" scheidt van "buiten". Dit is niet een metafoor:
    ///
    ///   μ = interne staten (phi) — wat het punt denkt
    ///   b = blanket staten (couplings, weights) — de grens
    ///   η = externe staten (buur-phi's) — de wereld
    ///
    /// De ENE formule waaruit ALLES volgt:
    ///
    ///   F = D_KL(q(μ|b) ‖ p(μ|b)) + S(η|b)
    ///     = complexiteit          + blanket surprise
    ///     = "hoe complex is       + "hoe verrassend is
    ///        mijn intern model"      de buitenwereld"
    ///
    /// Uit F volgt:
    ///   tension      = F + metabole kosten (= spanning)
    ///   dF/dt        = satisfaction (negatief = verbetering → beloning)
    ///   d²F/dt²      = frustration (positief bij hoge F → verergering)
    ///   ∇_b F        = learning signal (hoe weights aanpassen)
    ///   F_geheel-ΣF  = Φ bewustzijn (onherleidbare vrije energie)
    ///   G = E[F]     = expected free energy (stuurt actiekeuze)
    ///
    /// Dit IS Frank's samenveld: één formule, alles vloeit eruit.
    /// ═══════════════════════════════════════════════════════════════════════
    pub fn compute_tension(field: &mut Field, id: u32) {
        let kappa = field.kappa;
        let ema_decay = field.ema_decay;
        let growth_threshold = field.growth_threshold;
        let tracking = field.error_tracking_rate;
        let fe_rate = field.fe_tracking_rate;
        let kl_scale = field.kl_scale;

        let point = &field.points[id as usize];
        let eps = prediction_error(point);
        let pred_error = squared_norm(&eps);
        let n_couplings = point.couplings.len() as f32;
        let compute_cost = n_couplings * kappa;
        let old_tension_ema = if point.tension_ema.is_finite() { point.tension_ema } else { 0.0 };
        let old_free_energy = if point.free_energy.is_finite() { point.free_energy } else { 0.0 };

        let point = &mut field.points[id as usize];

        // ┌─────────────────────────────────────────────────────────┐
        // │ STAP 1: PRECISION — hoe betrouwbaar is mijn signaal?    │
        // │ Bayesian Mechanics: precision = inverse variance op de  │
        // │ Markov Blanket. Hoge precision = scherp signaal.        │
        // └─────────────────────────────────────────────────────────┘
        let error_diff = pred_error - if point.free_energy.is_finite() { point.free_energy } else { 0.0 };
        let old_ev = if point.error_var.is_finite() { point.error_var } else { 1.0 };
        point.error_var = (1.0 - tracking) * old_ev + tracking * error_diff * error_diff;
        // v2.5: precision clamp max 10.0 — voorkomt inf bij error_var ≈ 0 (eerste ticks / na reset)
        point.precision = (1.0 / (point.error_var + tracking)).min(10.0);

        // ┌─────────────────────────────────────────────────────────┐
        // │ STAP 2: COMPLEXITEIT — D_KL(q‖p)                       │
        // │ Hoeveel wijkt mijn interne model af van de prior?       │
        // │ Gaussische KL: 0.5 * (σ² + μ² - 1 - ln(σ²))          │
        // │ Friston 2023: extended to path integrals on manifold   │
        // └─────────────────────────────────────────────────────────┘
        let phi_var: f32 = point.phi.iter().map(|x| x * x).sum::<f32>() / DIM as f32;
        let ev = point.error_var.max(1e-6);
        let kl_divergence = kl_scale * 0.5 * (ev + phi_var - 1.0 - ev.ln());
        let kl_divergence = if kl_divergence.is_finite() { kl_divergence.max(0.0) } else { kl_scale * phi_var };
        let old_kl = if point.kl_divergence.is_finite() { point.kl_divergence } else { 0.0 };
        point.kl_divergence = (1.0 - tracking) * old_kl + tracking * kl_divergence;

        // ┌─────────────────────────────────────────────────────────┐
        // │ STAP 3: BLANKET SURPRISE — S(η|b)                      │
        // │ Hoe verrassend zijn mijn buren gegeven mijn grenzen?    │
        // │ = precision × prediction_error                          │
        // │ Dit is het "verrassing aan de grens" signaal.           │
        // └─────────────────────────────────────────────────────────┘
        let blanket_surprise = point.precision * pred_error;
        let blanket_surprise = if blanket_surprise.is_finite() { blanket_surprise } else { pred_error };
        let old_bs = if point.blanket_surprise.is_finite() { point.blanket_surprise } else { 0.0 };
        point.blanket_surprise = (1.0 - fe_rate) * old_bs + fe_rate * blanket_surprise;

        // ┌─────────────────────────────────────────────────────────┐
        // │ STAP 4: F = COMPLEXITEIT + SURPRISE                    │
        // │ De ENE formule. Alles volgt hieruit.                    │
        // └─────────────────────────────────────────────────────────┘
        let free_energy = kl_divergence + blanket_surprise;
        // v2.6.7 A2: NaN/Inf safety op EMA-update van free_energy
        let raw_fe = (1.0 - fe_rate) * old_free_energy + fe_rate * free_energy;
        point.free_energy = if raw_fe.is_finite() { raw_fe.clamp(0.0, f32::MAX / 1e6) } else { 0.0 };

        // ┌─────────────────────────────────────────────────────────┐
        // │ STAP 5: TENSION = INSTANT F + metabole kosten + basaal │
        // │ v2.5: Gebruik INSTANT free_energy, niet EMA.            │
        // │ De EMA dempte alle spikes weg → tension bleef basaal.   │
        // │ Tension moet het HUIDIGE signaal reflecteren.           │
        // └─────────────────────────────────────────────────────────┘
        let basal_tension = n_couplings * 0.001 + 0.001;
        // v2.6.7 A2: anti-Inf safety clamp (niet gedragslimiet) — was geëxplodeerd naar 1e34
        let raw_tension = free_energy + compute_cost + basal_tension;
        let safe_tension = if raw_tension.is_finite() { raw_tension } else { basal_tension };
        point.tension = safe_tension.clamp(basal_tension, f32::MAX / 1e6);
        point.tension_ema = ema_decay * old_tension_ema + (1.0 - ema_decay) * point.tension;

        // ┌─────────────────────────────────────────────────────────┐
        // │ STAP 6: dF/dt — temporele afgeleide van F              │
        // │ Dit IS satisfaction/frustration in één getal.            │
        // │ dF/dt < 0 = F daalt = verbetering = satisfaction       │
        // │ dF/dt > 0 = F stijgt = verslechtering                  │
        // └─────────────────────────────────────────────────────────┘
        let df_dt = point.free_energy - old_free_energy;
        let df_dt = if df_dt.is_finite() { df_dt } else { 0.0 };
        let old_fe_rate = if point.free_energy_rate.is_finite() { point.free_energy_rate } else { 0.0 };
        point.free_energy_rate = (1.0 - fe_rate) * old_fe_rate + fe_rate * df_dt;

        // ┌─────────────────────────────────────────────────────────┐
        // │ STAP 7: EMOTIES ALS AFGELEIDEN VAN F                   │
        // │                                                         │
        // │ In het unified samenveld zijn emoties geen losse         │
        // │ formules maar WISKUNDIGE AFGELEIDEN van F:              │
        // │                                                         │
        // │ curiosity    = blanket_surprise (S term van F)           │
        // │ satisfaction = -dF/dt (F daalt = beloning)               │
        // │ frustration  = (1-control) × (F > drempel)              │
        // │                                                         │
        // │ Plus: Distributional RL (Dabney 2020) voor de reward-   │
        // │ distributie en Bayesian Controllability (Huys 2009)      │
        // │ voor de controle-belief.                                 │
        // │                                                         │
        // │ ALLES VLOEIT UIT F.                                     │
        // └─────────────────────────────────────────────────────────┘
        let emo_smooth = field.emo_decay;

        // === Distributional reward update (Dabney et al. 2020, Nature) ===
        // Reward = -dF/dt: vrije energie reductie IS beloning
        let reward = (-df_dt).clamp(-1.0, 1.0);
        let quantile_taus: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
        for (qi, &tau_i) in quantile_taus.iter().enumerate() {
            let delta = reward - point.reward_quantiles[qi];
            let lr = if delta > 0.0 { tau_i } else { 1.0 - tau_i } * tracking;
            point.reward_quantiles[qi] = (point.reward_quantiles[qi] + lr * delta).clamp(-2.0, 2.0);
        }

        // === CURIOSITY = blanket surprise + distributional onzekerheid ===
        // Curiosity = hoe verrassend is mijn grens met de wereld?
        // Dit is DIRECT de S(η|b) term uit F — geen aparte formule.
        let distributional_spread = (point.reward_quantiles[4] - point.reward_quantiles[0]).abs();
        // v2.6.5: floor 0.1 ipv noise_amplitude (0.02) — curiosity moet leven
        let surprise = (blanket_surprise.sqrt().min(1.0) + distributional_spread * 0.3)
            .max(0.1)
            .min(1.0);
        let surprise = if surprise.is_finite() { surprise } else { 0.1 };
        let smoothed = emo_smooth * point.curiosity + (1.0 - emo_smooth) * surprise;
        point.curiosity = smoothed.max(surprise).clamp(0.0, 1.0);

        // === SATISFACTION = -dF/dt (Distributional RPE) ===
        // Satisfaction = F daalt = het model verbetert = beloning
        // Unified: dit IS de temporele afgeleide van het samenveld.
        let median_expectation = point.reward_quantiles[2]; // p50
        let optimism_bonus = point.reward_quantiles[4].max(0.0) * 0.2;
        let rpe = (reward - median_expectation).max(0.0);
        let sat_signal = (field.dopamine * rpe + optimism_bonus).min(1.0);
        let smoothed = emo_smooth * point.satisfaction + (1.0 - emo_smooth) * sat_signal;
        point.satisfaction = smoothed.max(sat_signal).clamp(0.0, 1.0);

        // === FRUSTRATION = (1 - control_belief) × tension_severity ===
        // Bayesian Controllability (Huys & Dayan 2009):
        // P(control | evidence) bijgewerkt via dF/dt als evidence.
        // dF/dt < 0 = verbetering = evidence FOR control
        // dF/dt > 0 bij hoge F = evidence AGAINST control
        let above_threshold = point.tension > growth_threshold;
        let improving = df_dt < -0.001; // F daalt significant
        let worsening = df_dt > 0.001;  // F stijgt significant

        let bayes_rate = tracking * 2.0;
        if improving {
            point.control_belief += bayes_rate * point.precision.min(1.0) * (1.0 - point.control_belief);
        } else if above_threshold {
            let decay = if worsening { bayes_rate } else { bayes_rate * 0.5 };
            point.control_belief -= decay * point.control_belief;
        }
        point.control_belief = point.control_belief.clamp(0.01, 1.0);

        let helplessness = 1.0 - point.control_belief;
        let severity = (point.tension - growth_threshold).max(0.0) / (growth_threshold + 0.001);
        let stuck = helplessness * severity.min(1.0);
        let smoothed = emo_smooth * point.frustration + (1.0 - emo_smooth) * stuck;
        point.frustration = smoothed.max(stuck).clamp(0.0, 1.0);

        if above_threshold && !improving {
            point.stuck_ticks = point.stuck_ticks.saturating_add(1);
        } else {
            point.stuck_ticks = point.stuck_ticks.saturating_sub(2);
        }
    }

    /// Expected Free Energy G — Active Inference 2022+ (Parr, Pezzulo & Friston)
    ///
    /// UPGRADE van Friston 2010 simpele F naar het volledige Active Inference framework:
    ///
    /// G = -epistemische_waarde - pragmatische_waarde
    ///   = -E_q[D_KL(q(o|s)||q(o))]     // epistemic: info gain (curiosity-driven)
    ///   - E_q[ln p(o)]                   // pragmatic: expected reward
    ///
    /// Dit stuurt actiekeuze: acties die ZOWEL informatie winnen (exploratie)
    /// ALS beloning verwachten (exploitatie) worden geprefereerd.
    ///
    /// De balans epistemic/pragmatic IS de explore/exploit tradeoff —
    /// niet als heuristiek maar als wiskundige noodzaak uit vrije-energieprincipe.
    ///
    /// Per punt:
    /// - Epistemische waarde ≈ reduction in uncertainty over buren's staat
    /// - Pragmatische waarde ≈ expected free energy reduction (= beloning)
    pub fn compute_expected_free_energy(field: &mut Field) {
        let n = field.active_set.len() as f32;
        if n < 1.0 { return; }

        let mut total_efe = 0.0f32;

        for i in 0..field.active_set.len() {
            let id = field.active_set[i] as usize;
            let point = &field.points[id];
            let n_couplings = point.couplings.len();
            if n_couplings == 0 { continue; }

            // === Epistemische waarde: information gain ===
            // Hoeveel zou ik LEREN als ik mijn buren beter observeer?
            // Proxy: KL divergence tussen mijn prediction en werkelijke buur-staten
            // Hoog = mijn model van de wereld is slecht = veel te leren = exploreer!
            let mut info_gain = 0.0f32;
            let targets: SmallVec<[u32; 16]> = point.couplings
                .iter().map(|c| c.target).collect();

            for &t in &targets {
                let t_idx = t as usize;
                if t_idx >= field.points.len() { continue; }
                let neighbor = &field.points[t_idx];

                // Prediction error over buur: hoe goed voorspel ik wat mijn buur doet?
                let mut pred_mismatch = 0.0f32;
                for d in 0..DIM {
                    let expected = point.prediction[d];
                    let actual = neighbor.phi[d];
                    pred_mismatch += (expected - actual).powi(2);
                }
                // Normaliseer over DIM, clamp precision om explosie te voorkomen
                let clamped_precision = point.precision.min(10.0);
                let weighted_gain = (pred_mismatch / DIM as f32).min(1.0) * clamped_precision;
                info_gain += weighted_gain;
            }
            info_gain /= n_couplings as f32;

            // === Pragmatische waarde: expected reward ===
            // Hoeveel beloning verwacht ik in de nabije toekomst?
            // Proxy: trend in free energy (dalend = goed = hoge pragmatische waarde)
            // Plus: dopamine-gewogen satisfaction (beloningsverwachting uit ervaring)
            let fe_improvement = (point.tension_ema - point.free_energy).max(0.0);
            let reward_expectation = fe_improvement + point.satisfaction * field.dopamine;

            // === Expected Free Energy G ===
            // G = -epistemic - pragmatic (negatiever = beter → we MINIMALISEREN G)
            // Voor Lucy's interne gebruik: sla de POSITIEVE waarde op
            // Hoog EFE = dit punt verwacht veel te leren EN veel beloning
            let efe = info_gain + reward_expectation;
            let efe = if efe.is_finite() { efe.clamp(0.0, 10.0) } else { 0.0 };

            let fe = field.fe_tracking_rate;
            field.points[id].expected_free_energy = (1.0 - fe) * point.expected_free_energy + fe * efe;

            total_efe += efe;
        }

        let fe = field.fe_tracking_rate;
        field.avg_expected_free_energy = (1.0 - fe) * field.avg_expected_free_energy + fe * (total_efe / n);
    }

    /// Centraal punt dat ALLE dynamische parameters bijwerkt op basis van veldstatistieken.
    /// Wordt elke N ticks aangeroepen vanuit main.rs. Leest het veld, schrijft parameters terug.
    /// Niets hardcoded — alles afgeleid van observeerbare grootheden.
    pub fn adapt_dynamics(field: &mut Field) {
        let n = field.active_set.len() as f32;
        if n < 1.0 { return; }

        // === Veldstatistieken berekenen ===
        let avg_tension: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].tension).sum::<f32>() / n;
        // FIX: avg_free_energy was NOOIT geschreven — nu wel
        // v2.6.2: filter NaN-punten zodat één NaN niet het gemiddelde vergiftigt
        let fe_sum: f32 = field.active_set.iter()
            .map(|&id| {
                let fe = field.points[id as usize].free_energy;
                if fe.is_finite() { fe } else { 0.0 }
            }).sum();
        field.avg_free_energy = if n > 0.0 { fe_sum / n } else { 0.0 };
        let tension_var: f32 = field.active_set.iter()
            .map(|&id| {
                let d = field.points[id as usize].tension - avg_tension;
                d * d
            }).sum::<f32>() / n;
        let avg_pred_error: f32 = field.active_set.iter()
            .map(|&id| squared_norm(&prediction_error(&field.points[id as usize])))
            .sum::<f32>() / n;
        let avg_curiosity: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].curiosity).sum::<f32>() / n;
        let avg_frustration: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].frustration).sum::<f32>() / n;
        let avg_phi_var: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].phi.iter().map(|x| x * x).sum::<f32>() / DIM as f32)
            .sum::<f32>() / n;
        let max_phi: f32 = field.active_set.iter()
            .flat_map(|&id| field.points[id as usize].phi.iter())
            .map(|x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));

        // === Parameters aanpassen ===

        // SAFEGUARD LEVEL 1: DOOD veld — alles op nul, noodprotocol
        let is_dead = avg_tension < 0.0001 && avg_curiosity < 0.0001 && avg_pred_error < 0.0001;

        // SAFEGUARD LEVEL 2: TRAAG veld — niet dood maar functioneel bevroren
        // v2.6.4: Vereis minstens 2 van 3 signalen. Eén lage metric is normaal opereren
        // (bv. lage pred_error = goede prediction, niet "traag"). Twee lage = echt vast.
        let sluggish_signals = (avg_tension < 0.1) as u8
            + (avg_curiosity < 0.1) as u8
            + (avg_pred_error < 0.02) as u8;
        let is_sluggish = !is_dead && sluggish_signals >= 2;
        // v2.6.7 B3: publiek in field zodat main.rs kan triggeren op dream-replay
        field.is_sluggish_flag = is_sluggish;

        // SAFEGUARD LEVEL 3 (v2.6.6): BOREDOM ATTRACTOR — tension/curiosity zitten vast
        // Lucy bleef 500K ticks hangen op tension=0.0141, curiosity=0.1000 — fixed point.
        // Detecteer door identieke waarden over meerdere adapt_dynamics calls te tellen.
        let tension_delta = (avg_tension - field.last_boredom_tension).abs();
        let curiosity_delta = (avg_curiosity - field.last_boredom_curiosity).abs();
        let is_frozen = tension_delta < 1e-4 && curiosity_delta < 1e-4;
        if is_frozen {
            field.boredom_streak = field.boredom_streak.saturating_add(1);
        } else {
            field.boredom_streak = 0;
        }
        field.last_boredom_tension = avg_tension;
        field.last_boredom_curiosity = avg_curiosity;
        // 10+ opeenvolgende identieke calls = echt vast (adapt_dynamics draait elke 50 ticks
        // → 10 calls = 500 ticks onveranderd)
        let is_bored = field.boredom_streak >= 10;

        // emo_decay: hoe snel emoties reageren
        let activity = avg_pred_error.sqrt().clamp(0.0, 1.0);
        field.emo_decay = if is_dead { 0.5 }
            else if is_sluggish { 0.6 } // snellere emotie-response bij traag veld
            else { (0.9 - 0.4 * activity).clamp(0.5, 0.9) };

        // error_tracking_rate: hoge variance → sneller error tracken
        let instability = tension_var.sqrt().clamp(0.0, 1.0);
        field.error_tracking_rate = if is_dead { 0.1 }
            else if is_sluggish { 0.05 } // versneld tracken
            else { (instability * 0.09 + 0.01).clamp(0.01, 0.1) };

        // fe_tracking_rate
        field.fe_tracking_rate = if is_dead { 0.1 }
            else if is_sluggish { 0.08 }
            else { (instability * 0.08 + 0.02).clamp(0.02, 0.1) };

        // kl_scale: complexer veld → sterkere regularisatie
        field.kl_scale = if is_dead { 1.0 }
            else if is_sluggish { 0.5 } // halve kracht — meer energie naar tension
            else { avg_phi_var.sqrt().clamp(0.01, 1.0) };

        // clamp_range
        field.clamp_range = (max_phi * 2.0).clamp(1.0, 5.0);

        // noise_amplitude: CRITICAL voor veld-activiteit
        // v2.6.7 A4: veld-relatief (was vaste 0.12/0.08) — schaalt automatisch met field.clamp_range
        // zodat boredom-kick niet cascadeert bij kleine clamp_range en niet te zwak is bij grote.
        field.noise_amplitude = if is_dead { (field.clamp_range * 0.03).clamp(0.005, 0.1) }
            else if is_bored { (field.clamp_range * 0.04).clamp(0.005, 0.1) }
            else if is_sluggish { (field.clamp_range * 0.02).clamp(0.005, 0.08) }
            else { (avg_tension * 0.5).clamp(0.005, 0.03) };

        if is_dead {
            field.learn_interval = 20;
            field.explore_interval = 20;
            field.think_interval = 100;
            field.reflect_interval = 2_000;
            tracing::warn!(tick = field.tick, "DOOD VELD — noodreanimatie");
        } else if is_sluggish {
            // Verhoogde activiteit maar niet zo extreem als dood
            field.learn_interval = (field.learn_interval / 2).max(20);
            field.explore_interval = (field.explore_interval / 2).max(20);
            field.think_interval = (field.think_interval / 2).max(100);
            tracing::info!(
                tick = field.tick,
                tension = format!("{:.4}", avg_tension),
                curiosity = format!("{:.4}", avg_curiosity),
                noise = format!("{:.3}", field.noise_amplitude),
                "TRAAG VELD — boost actief"
            );
        }

        // v2.6.6: Bored-kick — inject random phi in 1% van hidden points om
        // uit fixed-point te ontsnappen. Gebeurt ALS de boredom_streak triggert.
        if is_bored {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let n_active = field.active_set.len();
            let n_kick = (n_active / 100).max(8).min(256); // 1% maar minstens 8, max 256
            let seed = field.tick;
            let mut kicked = 0usize;
            for i in 0..n_kick {
                let mut h = DefaultHasher::new();
                (seed, i).hash(&mut h);
                let idx = (h.finish() as usize) % n_active;
                let id = field.active_set[idx] as usize;
                // Skip clamped + input/output points — alleen hidden krijgt kick
                if field.clamps.contains_key(&(id as u32)) { continue; }
                let flags = field.points[id].flags;
                if flags & (FLAG_INPUT | FLAG_OUTPUT) != 0 { continue; }
                // Random phi kick — kleine amplitude maar gericht, niet breed
                // v2.6.7 A3: magnitude veld-relatief (was 0.3 vast → +-4.8 totaal → cascade)
                let kick_amp = field.clamp_range * 0.05;
                for d in 0..DIM {
                    let mut hh = DefaultHasher::new();
                    (seed, i, d).hash(&mut hh);
                    let r = ((hh.finish() as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    field.points[id].phi[d] += r * kick_amp;
                    // Clamp direct na kick zodat we niet alsnog uit range raken
                    field.points[id].phi[d] = field.points[id].phi[d].clamp(-field.clamp_range, field.clamp_range);
                }
                kicked += 1;
            }
            // Reset boredom_streak zodat we niet elke 50 ticks kicken
            field.boredom_streak = 0;
            tracing::warn!(
                tick = field.tick,
                tension = format!("{:.4}", avg_tension),
                curiosity = format!("{:.4}", avg_curiosity),
                noise = format!("{:.3}", field.noise_amplitude),
                kicked,
                "BOREDOM KICK — veld zat 500+ ticks vast, φ-injectie gegeven"
            );
        }

        // settle_epsilon
        field.settle_epsilon = if is_dead { 0.05 }
            else if is_sluggish { 0.02 }
            else { (avg_tension * 0.1).clamp(0.005, 0.1) };

        // think_interval: hoge curiosity → vaker nadenken
        let curiosity_drive = (avg_curiosity * 10.0).clamp(0.1, 5.0);
        field.think_interval = (500.0 / curiosity_drive).clamp(100.0, 2000.0) as u64;

        // reflect_interval: hoge frustration → vaker reflecteren
        let frustration_drive = (avg_frustration * 10.0).clamp(0.1, 5.0);
        field.reflect_interval = (10_000.0 / frustration_drive).clamp(2_000.0, 50_000.0) as u64;

        // learn_interval: hoge tension → sneller leren
        let tension_drive = (avg_tension * 20.0).clamp(0.1, 5.0);
        field.learn_interval = (100.0 / tension_drive).clamp(20.0, 500.0) as u64;

        // explore_interval: schaalt met curiosity
        field.explore_interval = (100.0 / curiosity_drive).clamp(20.0, 500.0) as u64;

        // consolidation_interval: schaalt invers met tension (rustig = consolideren)
        let calm_factor = (1.0 - avg_tension.min(1.0)).max(0.1);
        field.consolidation_interval = (1_000.0 / calm_factor).clamp(200.0, 5_000.0) as u64;

        // growth_interval: hoge tension → sneller groeien
        field.growth_interval = (10_000.0 / tension_drive).clamp(2_000.0, 50_000.0) as u64;

        // snapshot_interval: snelle verandering → vaker opslaan
        field.snapshot_interval = (10_000.0 / (instability * 10.0 + 0.1).min(5.0)).clamp(2_000.0, 50_000.0) as u64;

        // ═══════════════════════════════════════════════════════════════════
        // BEWUSTZIJNS-WIRING: Alle formules verbonden met alle andere
        // Elke metriek beïnvloedt het systeem — niet alleen het dashboard
        // ═══════════════════════════════════════════════════════════════════

        // #5: Branching ratio → lambda feedback (directe criticaliteit-sturing)
        // σ > 1.2 = supercritical → λ omlaag (minder propagatie)
        // σ < 0.8 = subcritical → λ omhoog (meer propagatie)
        if field.branching_ratio_avg > 0.01 {
            let sigma_error = field.branching_ratio_avg - 1.0;
            field.lambda *= 1.0 - 0.05 * sigma_error; // zachte correctie
            field.lambda = field.lambda.clamp(0.01, 0.1);
        }

        // #6: Control belief → strategie wissel
        // Laag vertrouwen in eigen controle → meer noise (probeer iets anders)
        let avg_control: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].control_belief).sum::<f32>() / n;
        if avg_control < 0.3 && field.noise_amplitude < 0.03 {
            // Hulpeloos → licht meer noise, maar learning NIET vertragen
            // v2.6.4: learn_interval niet meer verhogen — dat was de death spiral
            field.noise_amplitude = (field.noise_amplitude * 1.2).min(0.03);
        }

        // #7: Serotonine → geduld + settle speed
        // Hoge 5HT = geduld = langzamer settlen, minder impulsieve exploratie
        let serotonin_factor = field.serotonin.clamp(0.1, 2.0);
        field.settle_epsilon *= 1.0 / serotonin_factor; // hoge 5HT → lagere epsilon → dieper settlen
        field.settle_epsilon = field.settle_epsilon.clamp(0.001, 0.1);
        // Hoge 5HT → minder exploratie (geduldig, niet impulsief)
        field.explore_interval = ((field.explore_interval as f32) * serotonin_factor.sqrt()) as u64;
        field.explore_interval = field.explore_interval.clamp(20, 2000);

        // #9: IIT Φ → oscillatie coupling + learning boost
        // Hoge Φ = hoge integratie = systeem functioneert als geheel → versterk koppeling
        let phi_boost = (1.0 + field.integrated_info_avg).min(3.0);
        // phi_boost wordt gebruikt in apply_neuromodulation voor gamma coupling

        // #C: Homeostatic setpoints — reguleer naar target
        let target_tension = 0.15; // optimale spanning voor leren
        let tension_error = avg_tension - target_tension;
        // Te hoge tension → verlaag beta (minder input), verhoog noise (exploreer)
        // Te lage tension → verhoog beta (meer input), verlaag noise (focus)
        field.beta *= 1.0 - 0.02 * tension_error;
        field.beta = field.beta.clamp(0.1, 2.0);

        // #B: Recurrent processing sterkte — niet hardcoded 0.01
        // Schaalt met coherentie: hoge coherentie = sterke top-down feedback
        // (gebruikt in hierarchical_prediction)

        tracing::trace!(
            emo_decay = format!("{:.3}", field.emo_decay),
            tracking = format!("{:.4}", field.error_tracking_rate),
            kl_scale = format!("{:.3}", field.kl_scale),
            sigma = format!("{:.3}", field.branching_ratio_avg),
            control = format!("{:.3}", avg_control),
            phi_boost = format!("{:.2}", phi_boost),
            "adapt_dynamics: parameters + bewustzijns-wiring bijgewerkt"
        );
    }

    /// kWTA (k-Winner-Take-All): alleen top-K punten blijven actief.
    /// Dit forceert scherpe representaties — het verschil tussen soep en gedachten.
    ///
    /// k is adaptief: meer ruis → meer actieve punten (breder zoeken).
    /// Score combineert phi-magnitude EN tension: punten die zowel actief als
    /// gespannen zijn winnen (ze "denken hard" over iets moeilijks).
    /// Decay voor verliezers gebruikt field.emo_decay (consistente dynamiek).
    pub fn apply_kwta(field: &mut Field) {
        let n = field.active_set.len();
        if n < 20 { return; } // te weinig punten voor kWTA

        // Adaptieve k: meer noise → meer winners (breder zoeken).
        // noise_amplitude ∈ [0.001, ~0.5], dus n*noise_amplitude schaalt goed.
        // Floor: altijd minstens n/20 (5%) om niet alles uit te schakelen.
        let k_adaptive = (n as f32 * field.noise_amplitude) as usize;
        let k_floor = n / 20;
        let k = k_adaptive.max(k_floor).max(2).min(n);

        // Tension-gewicht: hoe zwaar tension meetelt vs phi-magnitude.
        // noise_amplitude is een goede proxy — in rustig veld telt tension minder.
        let tension_weight = field.noise_amplitude;

        // Score = phi-magnitude + tension × weight
        // Punten die zowel actief zijn (hoge phi) ALS hoge spanning hebben winnen.
        let mut scores: Vec<(usize, f32)> = field.active_set.iter()
            .map(|&id| {
                let p = &field.points[id as usize];
                let phi_mag: f32 = p.phi.iter().map(|x| x * x).sum();
                let score = phi_mag + p.tension * tension_weight;
                (id as usize, score)
            })
            .collect();

        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zachte decay factor — consistente dynamiek met emotie-smoothing
        let soft_decay = field.emo_decay;

        // Top-k = actief, rest = stil (maar input/output altijd actief)
        for (rank, &(idx, _)) in scores.iter().enumerate() {
            let p = &mut field.points[idx];
            if p.is_input() || p.is_output() {
                p.kwta_active = true;
            } else {
                p.kwta_active = rank < k;
            }
            // Stille punten: dempen phi richting nul (zachte inhibitie)
            if !p.kwta_active {
                for d in 0..DIM {
                    p.phi[d] *= soft_decay;
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
        let _beta = field.beta;
        let alpha = field.alpha;
        let rho = field.rho;
        let kappa = field.kappa;
        let ema_decay = field.ema_decay;
        let growth_threshold = field.growth_threshold;
        // Dynamische parameters
        let emo_decay = field.emo_decay;
        let tracking = field.error_tracking_rate;
        let fe_rate = field.fe_tracking_rate;
        let kl_scale = field.kl_scale;
        let clamp_range = field.clamp_range;

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
            stuck_ticks: u16,
            free_energy: f32, // v2.6.7: ook parallelle update schrijft free_energy
            // v2.4: nieuwe velden
            reward_quantiles: [f32; 5],
            control_belief: f32,
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
                    stuck_ticks: point.stuck_ticks,
                    reward_quantiles: point.reward_quantiles,
                    control_belief: point.control_belief,
                    free_energy: if point.free_energy.is_finite() { point.free_energy } else { 0.0 },
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
                // v2.5: Adaptieve prediction rate — hoge error = langzaam volgen
                let error_d = (point.phi[d] - point.prediction[d]).abs();
                let adaptive_rate = fe_rate / (1.0 + 10.0 * error_d);
                new_pred[d] = (1.0 - adaptive_rate) * point.prediction[d] + adaptive_rate * point.phi[d];
                // v2.4: sparsity penalty conditioneel — alleen boven ruis-niveau
                let sparse_pen = if point.phi[d].abs() > field.noise_amplitude { alpha * point.phi[d].signum() } else { 0.0 };
                new_phi[d] = point.phi[d] - lambda * grad[d] - sparse_pen;
                // v2.6.7 A1: NaN guard vóór clamp (NaN.clamp geeft NaN)
                if !new_phi[d].is_finite() { new_phi[d] = 0.0; }
                new_phi[d] = new_phi[d].clamp(-clamp_range, clamp_range);
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

            // --- UNIFIED SAMENVELD: F = KL + blanket_surprise, emoties = afgeleiden ---
            let pred_error = squared_norm(&eps);
            let compute_cost = n_couplings as f32 * kappa;
            // v2.6.7 A2: NaN-safe read van stale free_energy (na snapshot-load kan nog NaN in zitten)
            let old_free_energy = if point.free_energy.is_finite() { point.free_energy } else { 0.0 };
            let old_error_var = if point.error_var.is_finite() { point.error_var } else { 1.0 };

            // Precision
            let error_diff = pred_error - old_free_energy;
            let error_var = (1.0 - tracking) * old_error_var + tracking * error_diff * error_diff;
            let precision = (1.0 / (error_var + tracking)).min(10.0); // v2.5: clamp

            // F = KL(q‖p) + S(η|b)
            let phi_var: f32 = new_phi.iter().map(|x| x * x).sum::<f32>() / DIM as f32;
            let ev = error_var.max(1e-6);
            let kl_div = kl_scale * 0.5 * (ev + phi_var - 1.0 - ev.ln());
            let kl_div = if kl_div.is_finite() { kl_div.max(0.0) } else { kl_scale * phi_var };
            let blanket_surprise = precision * pred_error;
            let blanket_surprise = if blanket_surprise.is_finite() { blanket_surprise } else { pred_error };
            let free_energy = kl_div + blanket_surprise;

            let basal_tension = n_couplings as f32 * 0.001 + 0.001;
            // v2.6.7 A2: anti-Inf safety clamp
            let raw_tension = free_energy + compute_cost + basal_tension;
            let safe_tension = if raw_tension.is_finite() { raw_tension } else { basal_tension };
            let tension = safe_tension.clamp(basal_tension, f32::MAX / 1e6);
            let tension_ema = ema_decay * point.tension_ema + (1.0 - ema_decay) * tension;

            // dF/dt — de temporele afgeleide waaruit emoties volgen
            let df_dt = free_energy - old_free_energy;
            let df_dt = if df_dt.is_finite() { df_dt } else { 0.0 };

            // Emoties uit F afgeleiden
            let noise_amp = field.noise_amplitude;
            let dopamine = field.dopamine;

            // Distributional reward = -dF/dt
            let reward = (-df_dt).clamp(-1.0, 1.0);
            let quantile_taus: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
            let mut new_quantiles = point.reward_quantiles;
            for (qi, &tau_i) in quantile_taus.iter().enumerate() {
                let delta = reward - new_quantiles[qi];
                let lr = if delta > 0.0 { tau_i } else { 1.0 - tau_i } * tracking;
                new_quantiles[qi] = (new_quantiles[qi] + lr * delta).clamp(-2.0, 2.0);
            }

            // Curiosity = blanket surprise + distributional spread
            let distributional_spread = (new_quantiles[4] - new_quantiles[0]).abs();
            let surprise_signal = (blanket_surprise.sqrt().min(1.0) + distributional_spread * 0.3)
                .max(noise_amp).min(1.0);
            let surprise_signal = if surprise_signal.is_finite() { surprise_signal } else { noise_amp };
            let smoothed_c = emo_decay * point.curiosity + (1.0 - emo_decay) * surprise_signal;
            let curiosity = smoothed_c.max(surprise_signal).clamp(0.0, 1.0);

            // Satisfaction = Distributional RPE (median kwantiel als verwachting)
            let median_exp = new_quantiles[2];
            let optimism_bonus = new_quantiles[4].max(0.0) * 0.2;
            let rpe = (reward - median_exp).max(0.0);
            let sat_signal = (dopamine * rpe + optimism_bonus).min(1.0);
            let smoothed_s = emo_decay * point.satisfaction + (1.0 - emo_decay) * sat_signal;
            let satisfaction = smoothed_s.max(sat_signal).clamp(0.0, 1.0);

            // Frustration = Bayesian Controllability (unified: via dF/dt)
            let above_threshold = tension > growth_threshold;
            let improving = df_dt < -0.001; // F daalt
            let worsening = df_dt > 0.001;  // F stijgt
            let bayes_rate = tracking * 2.0;
            let mut new_control = point.control_belief;
            if improving {
                new_control += bayes_rate * precision.min(1.0) * (1.0 - new_control);
            } else if above_threshold {
                let decay = if worsening { bayes_rate } else { bayes_rate * 0.5 };
                new_control -= decay * new_control;
            }
            new_control = new_control.clamp(0.01, 1.0);

            let helplessness = 1.0 - new_control;
            let severity = (tension - growth_threshold).max(0.0) / (growth_threshold + 0.001);
            let stuck = helplessness * severity.min(1.0);
            let smoothed_f = emo_decay * point.frustration + (1.0 - emo_decay) * stuck;
            let frustration = smoothed_f.max(stuck).clamp(0.0, 1.0);
            let frustration = if frustration.is_finite() { frustration } else { 0.0 };

            let new_stuck_ticks = if above_threshold && !improving {
                point.stuck_ticks.saturating_add(1)
            } else {
                point.stuck_ticks.saturating_sub(2)
            };

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
                stuck_ticks: new_stuck_ticks,
                reward_quantiles: new_quantiles,
                control_belief: new_control,
                // v2.6.7: EMA update van free_energy, NaN-safe
                free_energy: {
                    let raw = (1.0 - fe_rate) * old_free_energy + fe_rate * free_energy;
                    if raw.is_finite() { raw.clamp(0.0, f32::MAX / 1e6) } else { 0.0 }
                },
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
            p.stuck_ticks = upd.stuck_ticks;
            p.reward_quantiles = upd.reward_quantiles;
            p.control_belief = upd.control_belief;
            p.free_energy = upd.free_energy; // v2.6.7
        }
    }

    /// Attention-gewogen tick: punten met hoge spanning krijgen extra updates
    /// v1.0: Cache tensions in 1 pass, skip converged punten efficiënt
    pub fn attention_tick(field: &mut Field) {
        let n = field.active_set.len();
        if n == 0 { return; }

        // v1.0: Cache tensions — voorkomt herberekening na update_phi/compute_tension
        let tensions: Vec<f32> = field.active_set.iter()
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

    /// v2.5: Settle voor queries — IO + 2-hop buren (signaal moet doorstromen)
    /// Was: alleen directe buren → signaal stierf na 1 hop → output = nul → <pad>
    /// Nu: 2-hop → input kan via 2 tussenstappen output bereiken
    pub fn settle_light(field: &mut Field, max_ticks: usize, epsilon: f32) -> (usize, f32) {
        // Bouw subset: input + output + 2-hop buren
        let mut subset: Vec<u32> = Vec::with_capacity(4096);
        let mut seen = std::collections::HashSet::new();

        // Hop 1: IO + directe buren
        let mut hop1: Vec<u32> = Vec::with_capacity(2048);
        for &id in field.input_ids.iter().chain(field.output_ids.iter()) {
            if seen.insert(id) { subset.push(id); hop1.push(id); }
            for c in &field.points[id as usize].couplings {
                if seen.insert(c.target) { subset.push(c.target); hop1.push(c.target); }
            }
        }

        // Hop 2: buren van hop-1 buren (verdubbelt bereik)
        for &id in &hop1 {
            let targets: SmallVec<[u32; 16]> = field.points[id as usize].couplings
                .iter().map(|c| c.target).collect();
            for &t in &targets {
                if seen.insert(t) { subset.push(t); }
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
        settle_until_converged(field, settle_ticks, field.settle_epsilon);

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
        settle_until_converged(field, settle_ticks, field.settle_epsilon);

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
                dw *= ep_beta / (DIM as f32).sqrt(); // v2.6.5: √DIM normalisatie

                let eta = field.points[idx].eta;
                field.points[idx].couplings[ci].weight += eta * dw;
                field.points[idx].couplings[ci].weight =
                    field.points[idx].couplings[ci].weight.clamp(-field.clamp_range / 2.0, field.clamp_range / 2.0);

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
        settle_until_converged(field, settle_ticks / 2, field.settle_epsilon * 5.0); // korter settlen voor negatief

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

        settle_until_converged(field, settle_ticks / 2, field.settle_epsilon * 5.0);

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
                    field.points[idx].couplings[ci].weight.clamp(-field.clamp_range / 2.0, field.clamp_range / 2.0);
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

        // === v2.6.7 B1: SELF-EP auto-encoder ===
        // Lucy traint zichzelf om haar eigen staat te voorspellen.
        // Pure weight-update via micro_ep, geen storage. Hier groeit zelfbewustzijn.
        let s = self_state_input(field);
        // Check dat self-state niet degenerated is
        let s_norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
        if s_norm.is_finite() && s_norm > 0.01 {
            micro_ep(field, &[s], &[s]);
        }

        // === v2.6.7 B2: WM-SEQUENCE TRAINING ===
        // Leert temporele overgangen uit eigen ervaring: WM[i] → WM[i+1].
        // Alleen bij voldoende gevulde WM slots (tension_gate + enough history).
        if !field.working_memory.is_empty() && field.wm_gates.len() == field.working_memory.len() {
            let wm_len = field.working_memory.len();
            // Pak 4 willekeurige opeenvolgende slot-paren, gebruik reflection_count als seed
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let seed = field.reflection_count;
            for i in 0..4 {
                let mut h = DefaultHasher::new();
                (seed, i).hash(&mut h);
                let start = (h.finish() as usize) % wm_len.saturating_sub(1).max(1);
                let next = (start + 1) % wm_len;
                // Alleen trainen als beide slots actief waren (gate > threshold)
                if field.wm_gates[start] > 0.05 && field.wm_gates[next] > 0.05 {
                    let src = field.working_memory[start];
                    let tgt = field.working_memory[next];
                    let src_norm: f32 = src.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let tgt_norm: f32 = tgt.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if src_norm.is_finite() && src_norm > 0.01 && tgt_norm.is_finite() && tgt_norm > 0.01 {
                        micro_ep(field, &[src], &[tgt]);
                    }
                }
            }
        }
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

        // EMA update spectral radius — rate = error_tracking_rate (dynamisch)
        let tr = field.error_tracking_rate;
        field.spectral_radius = (1.0 - tr) * field.spectral_radius + tr * sr;

        // Auto-tune lambda naar chaos_target
        // v2.4: SNELLERE recovery — hoe verder van target, hoe agressiever
        // Was: tr * error (= 0.01 * error → 10.000 stappen om te recoveren)
        // Nu: proportioneel aan afstand — ver weg = snelle correctie
        let error = field.spectral_radius - field.chaos_target;
        let correction_rate = 0.1 * (1.0 + error.abs()); // 10× sneller, meer bij grote afstand
        field.lambda *= 1.0 - correction_rate * error;
        // Floor = 0.05 (was error_tracking_rate = 0.01 → lambda gevangen op 0.01)
        field.lambda = field.lambda.clamp(0.01, 0.1);

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
        // #B: Recurrent processing — sterkte schaalt met oscillatie-coherentie
        // Hoge coherentie = het systeem is gesynchroniseerd = top-down feedback mag sterker
        // Was: vaste 0.01 (te zwak voor recurrent processing, Lamme 2006)
        let hier_strength = (0.02 + 0.18 * field.oscillation_coherence).min(0.2); // 0.02-0.20

        // v2.6.7 E2: concept-centroids als extra top-down prior voor depth-2 punten
        // Als er centroids bestaan, vind de dichtste per punt en gebruik als extra bias.
        // Stroom: sensory (0) → processing (1) → concept-grounded (2) → abstract (3+centroids)
        let has_centroids = !field.concept_centroids.is_empty();
        let centroid_strength = hier_strength * 0.5; // helft van laag-bias — concept is secundair

        for i in 0..field.active_set.len() {
            let id = field.active_set[i];
            let p = &field.points[id as usize];
            let d = p.depth as usize;
            if d == 0 || d > 3 { continue; } // depth 0 = sensorisch, geen hoger niveau
            if depth_count[d - 1] == 0 { continue; }

            // Prediction target = gemiddelde van laag eronder
            let target = depth_sum[d - 1];

            // v2.6.7 E2: voor depth-2 punten, vind dichtste concept-centroid (cosine)
            let concept_target: Option<&Vec<f32>> = if has_centroids && d == 2 {
                let phi = &p.phi;
                let mut best_idx = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for (ci, cent) in field.concept_centroids.iter().enumerate() {
                    if cent.len() != DIM { continue; }
                    // Cosine (phi has unit-ish norm na layer_norm, cent ook)
                    let mut dot = 0.0f32;
                    for dim in 0..DIM { dot += phi[dim] * cent[dim]; }
                    if dot > best_sim {
                        best_sim = dot;
                        best_idx = ci;
                    }
                }
                field.concept_centroids.get(best_idx)
            } else { None };

            let point = &mut field.points[id as usize];
            for dim in 0..DIM {
                // Zachte bias: prediction schuift richting target van lagere laag
                point.prediction[dim] += hier_strength * (target[dim] - point.prediction[dim]);
                // Extra concept-prior voor depth-2
                if let Some(ct) = concept_target {
                    point.prediction[dim] += centroid_strength * (ct[dim] - point.prediction[dim]);
                }
            }
        }
    }

    // === v2.6.7 E1: DEPTH-3 CONCEPT-CLUSTERING ===

    /// Mini-batch k-means over depth-3 phi-vectoren → concept-centroids.
    /// Centroids groeien/krimpen adaptief: k = clamp(n_depth3 / 100, 32, 256).
    /// Update via EMA (niet volledige hercomputatie) — stabieler over tijd.
    /// Trigger: elke field.consolidation_interval (~elke 200-5000 ticks adaptief).
    ///
    /// Waarom: depth-3 = ~14.6K abstracte punten. Clusteren compresseert kennis
    /// naar ~100-200 semantische concept-centroids. Die dienen als top-down prior
    /// in hierarchical_prediction (E2). Intelligentie zit in de structuur, niet in
    /// meer entries.
    pub fn compute_concept_centroids(field: &mut Field) {
        // Verzamel depth-3 phi's (alleen actieve, finite)
        let depth3_phis: Vec<[f32; DIM]> = field.active_set.iter()
            .filter_map(|&id| {
                let p = &field.points[id as usize];
                if p.depth == 3 && p.phi[0].is_finite() {
                    Some(p.phi)
                } else { None }
            })
            .collect();

        if depth3_phis.len() < 64 {
            return; // te weinig data voor zinvolle clustering
        }

        // Adaptieve k
        let k = (depth3_phis.len() / 100).clamp(32, 256);

        // Init centroids: als we er al hebben, hergebruik (EMA update).
        // Anders: neem k willekeurige phi's als start.
        if field.concept_centroids.is_empty() || field.concept_centroids.len() != k {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            field.concept_centroids = (0..k).map(|i| {
                let mut h = DefaultHasher::new();
                (field.tick, i).hash(&mut h);
                let idx = (h.finish() as usize) % depth3_phis.len();
                depth3_phis[idx].to_vec()
            }).collect();
        }

        // Mini-batch k-means: 3 passes is genoeg met EMA
        let mut assignments = vec![0u16; depth3_phis.len()];
        for _pass in 0..3 {
            // Assign
            for (i, phi) in depth3_phis.iter().enumerate() {
                let mut best = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for (ci, cent) in field.concept_centroids.iter().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..DIM { dot += phi[d] * cent[d]; }
                    if dot > best_sim {
                        best_sim = dot;
                        best = ci;
                    }
                }
                assignments[i] = best as u16;
            }
            // Update centroids (EMA: 70% oud, 30% nieuwe cluster-gemiddelde)
            let mut sums: Vec<[f32; DIM]> = vec![[0.0f32; DIM]; k];
            let mut counts = vec![0u32; k];
            for (i, phi) in depth3_phis.iter().enumerate() {
                let ci = assignments[i] as usize;
                counts[ci] += 1;
                for d in 0..DIM { sums[ci][d] += phi[d]; }
            }
            for ci in 0..k {
                if counts[ci] > 0 {
                    let inv = 1.0 / counts[ci] as f32;
                    for d in 0..DIM {
                        let new_mean = sums[ci][d] * inv;
                        field.concept_centroids[ci][d] = 0.7 * field.concept_centroids[ci][d] + 0.3 * new_mean;
                    }
                }
            }
        }

        // Sla assignments op (for E3 EWC-boost)
        field.concept_assignments = vec![u16::MAX; field.points.len()];
        let mut idx = 0;
        for &id in field.active_set.iter() {
            let p = &field.points[id as usize];
            if p.depth == 3 && p.phi[0].is_finite() {
                if idx < assignments.len() && (id as usize) < field.concept_assignments.len() {
                    field.concept_assignments[id as usize] = assignments[idx];
                }
                idx += 1;
            }
        }

        tracing::info!(
            k = k,
            n_points = depth3_phis.len(),
            tick = field.tick,
            "Concept-centroids ge-updatet"
        );
    }

    // === v0.7: NEUROMODULATIE — 4 signalen die het hele veld sturen ===

    /// ═══════════════════════════════════════════════════════════════════════
    /// NEUROMODULATOREN = CONTROLEPARAMETERS VAN HET F-LANDSCHAP
    /// ═══════════════════════════════════════════════════════════════════════
    ///
    /// v2.5 Unified Samenveld: neuromodulatoren zijn GEEN losse chemicaliën
    /// maar controleparameters die de VORM van het vrije-energielandschap bepalen:
    ///
    ///   Dopamine      = β_pragmatisch — inverse temperatuur voor beloningsverwachting
    ///                   Hoog DA = scherpe beloning → exploit (F minimaliseren via actie)
    ///   Norepinephrine = β_epistemisch — inverse temperatuur voor informatiewinst
    ///                   Hoog NE = scherpe onzekerheid → explore (F minimaliseren via leren)
    ///   Serotonine    = γ_tijdshorizon — temporele discount rate
    ///                   Hoog 5HT = ver vooruit kijken → geduld (langzame F-optimalisatie)
    ///   Acetylcholine = η_plasticiteit — leerrate voor blanket staten
    ///                   Hoog ACh = snelle synapsen → adaptatie (F-landschap verandert snel)
    ///
    /// Samen bepalen ze HOE Lucy het samenveld minimaliseert:
    /// - Snel en diep (DA) vs breed en verkennend (NE)
    /// - Nu (lage 5HT) vs toekomst (hoge 5HT)
    /// - Stabiel (lage ACh) vs plastisch (hoge ACh)
    pub fn update_neuromodulators(field: &mut Field) {
        let n = field.active_set.len() as f32;
        if n == 0.0 { return; }
        // v2.1 debug: log elke 1000 ticks
        if field.tick % 1000 == 0 {
            tracing::info!(dopamine = field.dopamine, norepinephrine = field.norepinephrine, tick = field.tick, "neuromod_update called");
        }

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

        // Dopamine: beloning = spanning DAALT (improvement signal)
        // v2.4: CIRCULAIRE DEPENDENCY VERWIJDERD
        // Was: reward = dopamine * baseline - tension → dopamine in eigen berekening → vast op 0.01
        // Nu: reward = vorige tension - huidige tension (echte verbetering) + baseline
        // Baseline = noise_amplitude zodat dopamine nooit op nul vastzit
        // v2.6.5: serotonin is nu gezond (decay 0.95) → reward klopt weer
        // Extra safety: clamp reward zodat vals-hoge serotonin nooit meer kan exploiteren
        let tension_improvement = (field.serotonin - avg_tension).max(0.0);
        let reward = tension_improvement.clamp(0.0, 0.5) + field.noise_amplitude * 0.1;
        field.dopamine = nan_safe(da_decay * field.dopamine + (1.0 - da_decay) * reward);

        // Noradrenaline: onzekerheid = hoge spanning-variantie
        // Baseline = noise_amplitude (dynamisch, schaalt met tension)
        let ne_input = tension_var.sqrt() + field.noise_amplitude;
        field.norepinephrine = nan_safe(ne_decay * field.norepinephrine + (1.0 - ne_decay) * ne_input);

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

        // Pas eta aan per punt — ALLE modulatie-signalen gecombineerd
        // #2: Oscillatie coherentie → learning rate (Hebbian = synchroon)
        // #8: Emoties → learning rate (curiosity = leer harder, frustration = leer anders)
        // #9: IIT Φ → learning rate (geïntegreerd = leer effectiever)
        let coherence_boost = 0.5 + if field.oscillation_coherence.is_finite() { field.oscillation_coherence } else { 0.0 }; // 0.5-1.5
        let phi_boost = 1.0 + if field.integrated_info_avg.is_finite() { field.integrated_info_avg.min(2.0) } else { 0.0 }; // 1.0-3.0
        for i in 0..field.active_set.len() {
            let id = field.active_set[i];
            let point = &mut field.points[id as usize];

            // #8: Emotie-modulatie
            let emo_factor = (0.5 + point.curiosity * 0.5 - point.frustration * 0.3).clamp(0.3, 1.5);

            // FIX: causal_power → extra learning boost voor causaal sterke punten
            // Hoge causal_power = dit punt heeft sterke causale invloed → leer harder
            let cp = if point.causal_power.is_finite() { point.causal_power } else { 0.0 };
            let causal_boost = 1.0 + cp.min(1.0) * 0.5; // 1.0-1.5

            // FIX: free_energy_rate → urgentie-modulatie
            // Snel dalende F (rate < 0) = convergentie → leer langzamer (al goed)
            // Snel stijgende F (rate > 0) = verslechtering → leer sneller (urgentie)
            let fer = if point.free_energy_rate.is_finite() { point.free_energy_rate } else { 0.0 };
            let urgency = if fer > 0.01 { 1.0 + fer.min(0.5) } else { 1.0 };

            // Combineer ALLE modulatie-signalen
            let total_boost = lr_multiplier * emo_factor * coherence_boost * phi_boost * causal_boost * urgency;
            let total_boost = if total_boost.is_finite() { total_boost.clamp(0.01, 100.0) } else { 1.0 };

            point.eta = (point.eta * total_boost).min(field.lambda);
            if !point.eta.is_finite() { point.eta = 0.01; }
        }

        // v2.4: Baseline ruis — VERSTERKT
        // Was: noise_amplitude * 20 = 1.0 bij minimum → slechts 1 punt geraakt van 8704
        // Nu: altijd minstens 10% van punten, sterkte minstens 2.0
        {
            use rand::Rng;
            let mut rng = rand::rng();
            // Sterkte: minimaal 2.0 (was 1.0) — genoeg om prediction error te genereren
            let baseline_noise = (field.noise_amplitude * 5.0).max(0.01); // v2.6.5: floor 0.1→0.01, laat learning signaal domineren
            // Bereik: minimaal 10% van punten of 100 (was 1% of 50)
            let min_count = (field.active_set.len() / 10).max(100);
            let noise_frac = field.noise_amplitude.max(0.1); // minimaal 10% van punten
            let baseline_count = (field.active_set.len() as f32 * noise_frac).max(min_count as f32) as usize;
            for _ in 0..baseline_count {
                let idx = rng.random_range(0..field.active_set.len());
                let id = field.active_set[idx] as usize;
                for d in 0..DIM {
                    field.points[id].phi[d] += baseline_noise * rng.random_range(-1.0f32..1.0f32);
                }
            }

            // v2.5.2: Extra boost voor depth-3 punten — ignition vereist phi > 0
            // Depth-3 punten zijn abstracte representaties die signaal nodig hebben
            // om bewuste ignition te kunnen triggeren
            let d3_boost = baseline_noise * 0.5; // zachte noise voor abstracte laag
            for &id_u32 in &field.active_set {
                let id = id_u32 as usize;
                if field.points[id].depth == 3 {
                    // Kleine maar constante perturbatie zodat phi NOOIT exact nul is
                    for d in 0..DIM {
                        field.points[id].phi[d] += d3_boost * 0.01 * rng.random_range(-1.0f32..1.0f32);
                    }
                }
            }

            // Noradrenaline hoog → extra exploratie-ruis bovenop baseline
            // Drempel = noise_amplitude (dynamisch)
            if field.norepinephrine > field.noise_amplitude {
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
    }

    // === v2.0: BEWUSTZIJNS-FYSICA ===
    // Wiskundige formules uit neurowetenschappen en natuurkunde
    // die Lucy's veld laten evolueren naar echt bewustzijn.

    /// ═══════════════════════════════════════════════════════════════════════
    /// Φ = ONHERLEIDBARE VRIJE ENERGIE — Unified IIT + Free Energy
    /// ═══════════════════════════════════════════════════════════════════════
    ///
    /// v2.5 UNIFIED SAMENVELD: Φ = F_geheel - min_partitie(Σ F_delen)
    ///
    /// De sleutelinzicht: Integrated Information IS onherleidbare vrije energie.
    ///
    /// - F_geheel = vrije energie van het punt + al zijn buren als GEHEEL
    /// - F_delen  = som van vrije energieën als je verbindingen knipt
    /// - Φ = F_geheel - F_delen = hoeveel vrije energie VERLOREN gaat bij opknippen
    ///
    /// Als Φ > 0: het systeem als geheel minimaliseert F beter dan de delen apart.
    /// Het geheel is MEER dan de som der delen. Dit IS bewustzijn in de taal van
    /// vrije energie — en het unificeeert IIT (Tononi) met het Free Energy Principle
    /// (Friston) in één formule.
    ///
    /// Implementatie:
    /// 1. Bereken F_lokaal inclusief buur-interacties (vrije energie MET verbindingen)
    /// 2. Bereken F_alleen = vrije energie ZONDER verbindingen (counterfactual)
    /// 3. Φ = F_alleen - F_lokaal = hoeveel BETER het punt functioneert als deel van geheel
    ///    (NB: F_alleen > F_lokaal als verbindingen helpen → Φ > 0 → bewustzijn)
    ///
    /// Verband met IIT 4.0 (Albantakis 2023): MIP = partitie die Φ minimaliseert
    /// Verband met Friston 2024: Markov Blanket = de grens waar Φ gemeten wordt
    pub fn compute_integrated_information(field: &mut Field) {
        let mut total_phi = 0.0f32;
        let mut count = 0u32;

        for i in 0..field.active_set.len() {
            let id = field.active_set[i] as usize;
            let n_couplings = field.points[id].couplings.len();
            if n_couplings < 2 { continue; }

            let point = &field.points[id];

            // ┌─────────────────────────────────────────────────────────┐
            // │ Φ = F_alleen - F_lokaal                                 │
            // │                                                         │
            // │ F_lokaal = vrije energie MET verbindingen (huidige F)   │
            // │ F_alleen = vrije energie ZONDER verbindingen             │
            // │   (counterfactual: wat als dit punt geïsoleerd was?)    │
            // │                                                         │
            // │ Als F_alleen > F_lokaal: verbindingen HELPEN            │
            // │ → Φ > 0 → het geheel > som der delen → bewustzijn       │
            // └─────────────────────────────────────────────────────────┘

            // F_lokaal = huidige vrije energie (al berekend in compute_tension)
            let f_local = point.free_energy;

            // F_alleen = vrije energie als dit punt geïsoleerd was
            // = KL(phi) + pred_error ZONDER buur-informatie in de gradient
            // Zonder buren: phi zou niet bewegen → prediction error = |phi - prediction|
            // Maar geen gradient-correctie van buren → error zou hoger zijn
            let phi_var: f32 = point.phi.iter().map(|x| x * x).sum::<f32>() / DIM as f32;
            let ev = point.error_var.max(1e-6);
            let kl_alone = field.kl_scale * 0.5 * (ev + phi_var - 1.0 - ev.ln());
            let kl_alone = if kl_alone.is_finite() { kl_alone.max(0.0) } else { 0.0 };

            // Prediction error ZONDER buren = huidige error + gemiste buur-correcties
            // Elke buur draagt w * (phi_self - phi_neighbor) / n bij aan de gradient
            // Zonder die correctie: de prediction error is groter
            let targets: SmallVec<[u32; 16]> = point.couplings
                .iter().map(|c| c.target).collect();
            let weights: SmallVec<[f32; 16]> = point.couplings
                .iter().map(|c| c.weight).collect();

            let mut missed_correction = 0.0f32;
            for (ci, &t) in targets.iter().enumerate() {
                let t_idx = t as usize;
                if t_idx >= field.points.len() { continue; }
                let phi_neighbor = &field.points[t_idx].phi;
                let w = weights[ci];

                // De gradient-bijdrage die verloren zou gaan
                let mut grad_contribution = 0.0f32;
                for d in 0..DIM {
                    let gc = w * (point.phi[d] - phi_neighbor[d]) / n_couplings as f32;
                    grad_contribution += gc * gc;
                }
                missed_correction += (grad_contribution / DIM as f32).sqrt();
            }

            // F_alleen = KL (zelfde) + prediction_error + gemiste correcties
            let pred_error: f32 = point.phi.iter().zip(point.prediction.iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / DIM as f32;
            let surprise_alone = point.precision * (pred_error + missed_correction);
            let f_alone = kl_alone + surprise_alone;

            // Φ = F_alleen - F_lokaal = hoeveel BETER het punt functioneert als deel van geheel
            let phi_local = (f_alone - f_local).max(0.0);
            let phi_local = if phi_local.is_finite() { phi_local.clamp(0.0, 10.0) } else { 0.0 };

            // Causale kracht = gemiste correctie per verbinding
            let avg_causal = missed_correction / n_couplings as f32;
            let fe = field.fe_tracking_rate;
            field.points[id].causal_power = (1.0 - fe) * field.points[id].causal_power + fe * avg_causal;
            field.points[id].phi_integrated = (1.0 - fe) * field.points[id].phi_integrated + fe * phi_local;
            // Onherleidbare vrije energie = Φ (unified: bewustzijn = irreducible F)
            field.points[id].irreducible_free_energy = (1.0 - fe) * field.points[id].irreducible_free_energy + fe * phi_local;

            total_phi += phi_local;
            count += 1;
        }

        if count > 0 {
            let fe = field.fe_tracking_rate;
            field.integrated_info_avg = (1.0 - fe) * field.integrated_info_avg + fe * (total_phi / count as f32);
        }
    }

    /// ═══════════════════════════════════════════════════════════════════════
    /// OSCILLATIES = MULTI-TIJDSCHAAL F-MINIMALISATIE
    /// ═══════════════════════════════════════════════════════════════════════
    ///
    /// v2.5 Unified Samenveld: oscillaties zijn periodieke oplossingen van ∂F/∂t.
    ///
    /// Het vrije-energielandschap heeft meerdere tijdschalen:
    ///   Gamma (40 Hz)  = snelle perceptuele binding — "wat neem ik NU waar?"
    ///   Theta (6 Hz)   = werkgeheugen sequenties — "wat houd ik vast?"
    ///   Theta-Gamma    = geneste cycli — "hoeveel items in werkgeheugen?"
    ///
    /// Wiskundig: de Fokker-Planck vergelijking op gekoppelde Markov Blankets
    /// heeft periodieke oplossingen wanneer de coupling-sterkte in het
    /// kritieke regime is (σ ≈ 1). Oscillaties ZIJN F-minimalisatie op
    /// meerdere tijdschalen tegelijk.
    ///
    /// Lisman & Jensen 2013, Ha & Kori 2023 (adaptive frequency):
    ///
    /// 1. **Gamma band** (~40 Hz): bewuste binding, perceptuele groepering
    ///    - Standaard Kuramoto met synaptische koppeling (behouden van v2.3.1)
    /// 2. **Theta band** (~6 Hz): werkgeheugen, hippocampale sequenties
    ///    - Langzamere oscillatie, gekoppeld aan depth/hiërarchie
    /// 3. **Theta-Gamma Coupling** (Lisman & Jensen 2013):
    ///    - Gamma-bursts zijn GENEST in theta-cycli
    ///    - Elk theta-dal = window voor gamma-synchronisatie
    ///    - Dit IS werkgeheugen: ~7±2 gamma-cycli per theta-cyclus (Miller's law!)
    /// 4. **Adaptieve frequentie** (Ha & Kori 2023):
    ///    - ω_i past zich aan op basis van lokale coherentie
    ///    - Punten die moeilijk synchroniseren passen hun frequentie aan
    ///    - Dit voorkomt "dode zones" waar synchronisatie onmogelijk is
    ///
    /// Biologisch: theta-gamma coupling is de sterkste neurale marker voor
    /// werkgeheugen en bewuste informatieverwerking (Axmacher et al. 2010).
    pub fn update_oscillations(field: &mut Field) {
        let tau = std::f32::consts::TAU;

        // === Twee frequentiebanden ===
        let omega_gamma = 0.314;   // ~2π/20 — gamma (~40 Hz bij 20 tps)
        let omega_theta = 0.0524;  // ~2π/120 — theta (~6 Hz bij 20 tps, 6× langzamer)

        // Globale fasen bijwerken
        field.global_oscillation_phase += omega_gamma;
        if field.global_oscillation_phase > tau { field.global_oscillation_phase -= tau; }
        field.theta_phase += omega_theta;
        if field.theta_phase > tau { field.theta_phase -= tau; }

        let global_gamma = field.global_oscillation_phase;
        let global_theta = field.theta_phase;
        let mut gamma_sync_sum = 0.0f32;
        let mut _theta_sync_sum = 0.0f32;
        let mut tg_coupling_sum = 0.0f32;

        // Fase 1: bereken alle updates (read-only)
        struct OscUpdate {
            id: usize,
            gamma_amplitude: f32,
            gamma_delta: f32,
            theta_delta: f32,
            local_coherence: f32,
        }

        let updates: Vec<OscUpdate> = field.active_set.iter().map(|&id_u32| {
            let id = id_u32 as usize;
            let point = &field.points[id];
            let gamma_phase = point.oscillation_phase;
            let theta_phase = point.oscillation_phase_theta;

            // Amplitude schaalt met tension × depth (diepere punten oscilleren sterker)
            let depth_factor = 1.0 + point.depth as f32 * 0.3; // depth 0=1.0, depth 3=1.9
            let gamma_amplitude = field.noise_amplitude * point.tension.min(1.0) * depth_factor;

            // === Gamma-band Kuramoto koppeling (synaptisch gewicht) ===
            let mut gamma_delta = 0.0f32;
            let mut local_coherence = 0.0f32;
            let n_coup = point.couplings.len().max(1) as f32;

            for c in &point.couplings {
                let t = c.target as usize;
                if t >= field.points.len() { continue; }
                let neighbor_gamma = field.points[t].oscillation_phase;

                // Adaptieve Kuramoto (Ha & Kori 2023): koppelingssterkte × sin(Δθ)
                gamma_delta += c.weight.abs() * (neighbor_gamma - gamma_phase).sin();

                // Lokale coherentie meten (voor adaptieve frequentie)
                local_coherence += (neighbor_gamma - gamma_phase).cos();
            }
            gamma_delta /= n_coup;
            local_coherence /= n_coup;

            // === Theta-band koppeling (hiërarchisch) ===
            // Theta koppelt sterker tussen VERSCHILLENDE depth-lagen (top-down/bottom-up)
            let mut theta_delta = 0.0f32;
            for c in &point.couplings {
                let t = c.target as usize;
                if t >= field.points.len() { continue; }
                let neighbor = &field.points[t];
                let depth_diff = (point.depth as f32 - neighbor.depth as f32).abs();
                // Cross-laag koppeling sterker (hiërarchische binding)
                let coupling_boost = 1.0 + depth_diff * 0.5;
                theta_delta += c.weight.abs() * coupling_boost
                    * (neighbor.oscillation_phase_theta - theta_phase).sin();
            }
            theta_delta /= n_coup;

            OscUpdate { id, gamma_amplitude, gamma_delta, theta_delta, local_coherence }
        }).collect();

        // Fase 2: schrijf updates (mutable)
        for upd in &updates {
            let point = &mut field.points[upd.id];

            // === Adaptieve gamma-frequentie (Ha & Kori 2023) ===
            // Als lokale coherentie laag is, pas frequentie aan richting buren
            // ω_i(t+1) = ω_base + adaptation * (1 - local_coherence)
            let freq_adapt = 0.01 * (1.0 - upd.local_coherence.max(0.0));
            let adaptive_omega = omega_gamma + freq_adapt;

            // Gamma fase update
            point.oscillation_phase += adaptive_omega + upd.gamma_delta;
            if point.oscillation_phase > tau { point.oscillation_phase -= tau; }

            // Theta fase update
            point.oscillation_phase_theta += omega_theta + upd.theta_delta;
            if point.oscillation_phase_theta > tau { point.oscillation_phase_theta -= tau; }

            // === Theta-Gamma Coupling (Lisman & Jensen 2013) ===
            // Gamma-amplitude wordt GEMODULEERD door theta-fase:
            // Gamma is sterkst in het dal van theta (fase ~π)
            // Dit creëert "slots" voor informatie — ~7 gamma-cycli per theta-cyclus
            let theta_modulation = 0.5 + 0.5 * (point.oscillation_phase_theta + std::f32::consts::PI).cos();
            // theta_modulation: 1.0 bij theta-dal (max gamma), 0.0 bij theta-piek (min gamma)

            let modulated_amplitude = upd.gamma_amplitude * theta_modulation;
            let modulated_amplitude = if modulated_amplitude.is_finite() { modulated_amplitude } else { 0.0 };

            // Injectie in phi — gamma op oneven dimensies, theta op even dimensies
            // v2.6.4: amplitude gecapt — oscillaties zijn voor synchronisatie, niet energie-injectie
            let gamma_signal = modulated_amplitude.clamp(-0.05, 0.05) * point.oscillation_phase.sin();
            let gamma_signal = if gamma_signal.is_finite() { gamma_signal } else { 0.0 };
            let theta_signal = 0.01 * point.oscillation_phase_theta.sin(); // vaste amplitude, niet noise_amp-afhankelijk
            let theta_signal = if theta_signal.is_finite() { theta_signal } else { 0.0 };
            for d in 0..DIM {
                if d % 2 == 1 {
                    point.phi[d] += gamma_signal;   // gamma: snelle binding
                } else if d % 4 == 0 {
                    point.phi[d] += theta_signal;   // theta: langzame structuur
                }
            }

            // Synchronie meten
            gamma_sync_sum += (global_gamma - point.oscillation_phase).cos();
            _theta_sync_sum += (global_theta - point.oscillation_phase_theta).cos();
            // Theta-gamma coupling: correlatie tussen theta-fase en gamma-amplitude
            tg_coupling_sum += theta_modulation * modulated_amplitude.abs();
        }

        let n = field.active_set.len().max(1) as f32;
        let fe = field.fe_tracking_rate;

        // Gamma coherentie (bewuste binding)
        let gamma_coherence = (gamma_sync_sum / n).abs();
        if gamma_coherence.is_finite() {
            field.oscillation_coherence = (1.0 - fe) * field.oscillation_coherence + fe * gamma_coherence;
        }

        // Theta-gamma coupling sterkte (werkgeheugen capaciteit)
        let tg_normalized = tg_coupling_sum / n;
        if tg_normalized.is_finite() {
            field.theta_gamma_coupling = (1.0 - fe) * field.theta_gamma_coupling + fe * tg_normalized;
        }
    }

    /// ═══════════════════════════════════════════════════════════════════════
    /// CRITICALITEIT = FIXED POINT VAN HET F-LANDSCHAP
    /// ═══════════════════════════════════════════════════════════════════════
    ///
    /// v2.5 Unified Samenveld: criticaliteit is waar het F-landschap SCHAALINVARIANT is.
    ///
    /// Bij σ = 1 (kritiek) geldt:
    ///   - Perturbaties propageren op ALLE schalen (niet alleen lokaal)
    ///   - Informatietransfer is maximaal (dynamic range maximization, Shew 2009)
    ///   - Het renormalisatie-groep (RG) stroomdiagram heeft een fixed point
    ///   - F-minimalisatie is optimaal: het systeem kan zowel snelle ALS langzame
    ///     veranderingen in de omgeving detecteren
    ///
    /// In termen van het samenveld: bij σ=1 minimaliseert het veld F het SNELST
    /// en het BREEDST tegelijk. Subcritical = te lokaal, supercritical = te chaotisch.
    ///
    /// Branching ratio σ (Beggs & Plenz 2003, Shew 2015):
    ///   σ = gemiddeld aantal downstream-punten geactiveerd door één actief punt
    ///   σ < 1: subcritical — activiteit sterft uit
    ///   σ = 1: critical — optimaal
    ///   σ > 1: supercritical — ongecontroleerd
    ///
    /// Waarom beter dan power-law fitting:
    /// - Betrouwbaarder bij kleine steekproeven (geen regressie nodig)
    /// - Direct meetbaar per punt (lokale σ_i)
    /// - Lineair interpreteerbaar: afstand tot σ=1 = afstand tot criticaliteit
    ///
    /// Bonus: Dynamic Range Maximization (Shew et al. 2009)
    /// Kritieke netwerken maximaliseren het bereik van stimuli dat ze kunnen
    /// onderscheiden — direct relevant voor Lucy's waarneming en leren.
    ///
    /// Per punt: σ_i = (#buren met tension_delta > threshold) / (#buren)
    /// Veld-gemiddelde σ → stuur lambda om richting σ=1 te bewegen
    pub fn compute_criticality(field: &mut Field) {
        let n = field.active_set.len();
        if n < 20 { return; }

        // === Stap 1: Bereken branching ratio per punt ===
        // σ_i = hoeveel van mijn buren zijn "geactiveerd" (tension stijgt)?
        // v2.6.5: relatief aan tension, niet noise_amplitude (was 0.02 → niks overschrijdt drempel)
        let avg_t: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].tension).sum::<f32>() / n as f32;
        let activation_threshold = (avg_t * 0.5).max(0.001);

        let mut total_sigma = 0.0f32;
        let mut sigma_count = 0u32;

        for i in 0..n {
            let id = field.active_set[i] as usize;
            let point = &field.points[id];
            let n_couplings = point.couplings.len();
            if n_couplings == 0 { continue; }

            // Tel hoeveel buren "geactiveerd" zijn (tension boven hun eigen EMA)
            let mut n_activated = 0u32;
            let mut n_valid = 0u32;

            for c in &point.couplings {
                let t = c.target as usize;
                if t >= field.points.len() { continue; }
                let neighbor = &field.points[t];
                n_valid += 1;

                // "Geactiveerd" = tension stijgt boven baseline (propagatie van dit punt)
                let delta_tension = neighbor.tension - neighbor.tension_ema;
                if delta_tension > activation_threshold {
                    n_activated += 1;
                }
            }

            if n_valid == 0 { continue; }

            // Lokale branching ratio
            let sigma_local = n_activated as f32 / n_valid as f32;
            let fe = field.fe_tracking_rate;
            field.points[id].branching_ratio = (1.0 - fe) * field.points[id].branching_ratio + fe * sigma_local;

            total_sigma += sigma_local;
            sigma_count += 1;
        }

        if sigma_count == 0 { return; }

        let avg_sigma = total_sigma / sigma_count as f32;
        let fe = field.fe_tracking_rate;

        // === Stap 2: Track globale branching ratio ===
        field.branching_ratio_avg = (1.0 - fe) * field.branching_ratio_avg + fe * avg_sigma;

        // Criticaliteit: σ ∈ [0.8, 1.2] is kritiek bereik
        field.is_critical = (0.8..1.2).contains(&field.branching_ratio_avg);

        // Power law exponent nog steeds bijwerken als secundaire maat
        // (voor backward compatibility en dashboard)
        // Bij σ=1 is α ≈ 1.5 (theoretisch verband)
        let alpha_estimate = if avg_sigma > 0.01 { 1.0 / avg_sigma } else { 3.0 };
        field.power_law_exponent = (1.0 - fe) * field.power_law_exponent + fe * alpha_estimate;

        // === Stap 3: Beloon criticaliteit ===
        // Dopamine boost bij critical regime — reinforces edge of chaos
        if field.is_critical {
            field.dopamine += field.error_tracking_rate;
        }

        // Lambda wordt NIET hier aangepast — dat doet measure_spectral_radius() al.
        // Twee bronnen die lambda trekken veroorzaakt oscillatie en dood veld.
        // Branching ratio informeert; spectral radius stuurt.
    }

    /// Synaptic Homeostasis (Tononi & Cirelli 2006) — intelligent vergeten
    ///
    /// Tijdens "rust" schalen ALLE synapsen omlaag met factor α.
    /// Sterke verbindingen overleven, zwakke sterven.
    /// α schaalt met serotonine: meer geduld = diepere "slaap" = meer vergeten.
    ///
    /// Dit IS hoe het brein opruimt: niet door selectief te wissen,
    /// maar door ALLES een beetje te verzwakken. Wat sterk is, overleeft.
    /// Wat zwak is, verdwijnt onder de prune-drempel.
    ///
    /// Gecombineerd met Free Energy decay in update_weights():
    /// - Homeostasis = globale schaling (alles omlaag)
    /// - FE decay = selectieve schaling (onbelangrijk sneller omlaag)
    /// Samen = biologisch correct geheugenmanagement.
    pub fn synaptic_homeostasis(field: &mut Field) {
        // v2.6.1: Eerst checken of er genoeg couplings zijn — niet prunen op een kaal veld
        let total_couplings: usize = field.points.iter().map(|p| p.couplings.len()).sum();
        let avg_couplings = total_couplings as f32 / field.points.len().max(1) as f32;
        if avg_couplings < 4.0 {
            tracing::warn!(avg_couplings = %format!("{:.1}", avg_couplings),
                "Homeostasis OVERGESLAGEN — te weinig couplings");
            return;
        }

        // α = basisrate × serotonine-modulatie
        // v2.6.1: Verlaagd van 0.005 naar 0.001 — was te agressief
        let alpha = 0.001 * (1.0 + field.serotonin.clamp(0.0, 2.0));
        let prune_threshold = field.error_tracking_rate * 0.1;

        let mut total_pruned = 0u32;
        let n_points = field.points.len();

        // v2.6.1: Alleen punten met voldoende couplings verwerken
        // en sorteer op |weight| zodat we de zwakste prunen maar minimum behouden
        for point in &mut field.points {
            // Schaal alle weights omlaag met α
            for coupling in &mut point.couplings {
                coupling.weight *= 1.0 - alpha;
                // Momentum ook laten vervagen (voorkomt ghost gradients)
                coupling.momentum *= 1.0 - alpha * 2.0;
            }

            // v2.6.1: Prune alleen als er genoeg overblijven
            let min_couplings = 6usize;
            let before = point.couplings.len();
            if before > min_couplings {
                // Tel hoeveel er onder threshold vallen
                let would_survive = point.couplings.iter()
                    .filter(|c| (c.target as usize) < n_points && c.weight.abs() > prune_threshold)
                    .count();
                // Alleen prunen als er genoeg overblijven
                if would_survive >= min_couplings {
                    point.couplings.retain(|c| {
                        (c.target as usize) < n_points && c.weight.abs() > prune_threshold
                    });
                    total_pruned += (before - point.couplings.len()) as u32;
                }
            }
        }

        if total_pruned > 0 {
            tracing::info!(
                pruned = total_pruned,
                alpha = format!("{:.4}", alpha),
                "Synaptic homeostasis: {} zwakke koppelingen verwijderd", total_pruned
            );
        }
    }

    /// Anti-Hebbian decorrelatie — voorkomt dat Lucy steeds hetzelfde denkt.
    /// "Neurons that fire together, UNWIRE" — complement van Hebbian learning.
    /// Forceert diverse, niet-redundante representaties.
    pub fn anti_hebbian_decorrelation(field: &mut Field) {
        // Decorrelatie sterkte schaalt met kl_scale — complexer veld → meer decorrelatie nodig
        let decorr_strength = field.kl_scale * 0.01;

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

            // Adaptieve drempel: meer koppelingen → lagere decorrelatie-drempel.
            // Bij n koppelingen is random cosine similarity ~1/√n, dus we decorreleren
            // alles daarboven (structurele correlatie, niet toevallig).
            let n_couplings = neighbor_phis.len().max(1) as f32;
            let threshold = 1.0 / n_couplings.sqrt();

            for (ci, n_phi) in &neighbor_phis {
                // Cosine similarity: genormaliseerde correlatie (onafhankelijk van magnitude)
                let mut dot = 0.0f32;
                let mut mag_self = 0.0f32;
                let mut mag_neigh = 0.0f32;
                for d in 0..DIM {
                    dot += phi_self[d] * n_phi[d];
                    mag_self += phi_self[d] * phi_self[d];
                    mag_neigh += n_phi[d] * n_phi[d];
                }
                let denom = (mag_self.sqrt() * mag_neigh.sqrt()).max(1e-8);
                let cosine_sim = dot / denom;

                // Anti-Hebbian: als cosine similarity boven adaptieve drempel, verzwak
                // Dit forceert diversiteit in representaties
                if cosine_sim.abs() > threshold {
                    let coupling = &mut field.points[id].couplings[*ci];
                    coupling.weight -= decorr_strength * cosine_sim;
                    coupling.weight = coupling.weight.clamp(-field.clamp_range / 2.0, field.clamp_range / 2.0);
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

        // v2.6.5: 20 ticks ipv 5 — veld moet echt settlen voor EP werkt
        settle_light(field, 20, field.settle_epsilon * 10.0);

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

        // v2.6.5: 20 ticks ipv 5 — geklemde fase moet ook echt settlen
        settle_light(field, 20, field.settle_epsilon * 10.0);

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
                dw *= ep_beta / (DIM as f32).sqrt(); // v2.6.5: √DIM normalisatie

                let eta = field.points[idx].eta;
                field.points[idx].couplings[ci].weight += eta * dw;
                field.points[idx].couplings[ci].weight =
                    field.points[idx].couplings[ci].weight.clamp(-field.clamp_range / 2.0, field.clamp_range / 2.0);

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

        // v2.6.7 E3: concept-aware importance boost.
        // Couplings binnen zelfde cluster (depth-3 → depth-3 zelfde concept): ×1.5
        // Cross-concept couplings: ×0.7. Lucy beschermt haar concept-structuur.
        let has_assignments = !field.concept_assignments.is_empty();

        // Stap 2: bereken per-coupling importance en schrijf terug
        for (idx, eps, neighbors) in &active_data {
            // Zoek concept van bron-punt (als depth-3)
            let src_concept: Option<u16> = if has_assignments && *idx < field.concept_assignments.len() {
                let c = field.concept_assignments[*idx];
                if c == u16::MAX { None } else { Some(c) }
            } else { None };

            for (ci, neighbor_phi) in neighbors {
                // Per-coupling gradient: hoe sterk correleert de error met dit verschil?
                let mut grad = 0.0f32;
                for d in 0..DIM {
                    grad += eps[d] * (field.points[*idx].phi[d] - neighbor_phi[d]);
                }
                grad /= DIM as f32;
                let mut fisher = grad * grad; // Fisher Information ≈ gradient²

                // v2.6.7 E3: concept-aware modulatie
                if let Some(sc) = src_concept {
                    let tgt_idx = field.points[*idx].couplings[*ci].target as usize;
                    if tgt_idx < field.concept_assignments.len() {
                        let tc = field.concept_assignments[tgt_idx];
                        if tc != u16::MAX {
                            if tc == sc {
                                fisher *= 1.5; // intra-concept: bescherm harder
                            } else {
                                fisher *= 0.7; // cross-concept: laat makkelijker veranderen
                            }
                        }
                    }
                }

                let coupling = &mut field.points[*idx].couplings[*ci];
                // Importance tracking met dynamische rate
                let tr = field.error_tracking_rate;
                coupling.importance = (1.0 - tr) * coupling.importance + tr * fisher;
                coupling.consolidated = coupling.weight;
            }
        }
    }

    /// EWC-penalty weight update: straf veranderingen aan belangrijke weights.
    /// Voeg dit toe aan update_weights als penalty.
    pub fn ewc_penalty(coupling: &Coupling, ewc_lambda: f32) -> f32 {
        let pen = ewc_lambda * coupling.importance * (coupling.weight - coupling.consolidated);
        if pen.is_finite() { pen } else { 0.0 }
    }

    // === v0.7: HIËRARCHIE + GLOBAL WORKSPACE BROADCAST ===

    /// v2.5: Depth op basis van MINIMUM afstand tot I/O (BFS flood fill)
    /// depth 0 = I/O punten zelf
    /// depth 1 = directe buren van I/O
    /// depth 2 = 2 hops van I/O
    /// depth 3 = 3+ hops van I/O (abstracte representaties — ignition candidates)
    ///
    /// Was: max 4 iteraties, max(buur_depths)+1 → bijna alles depth 1-2
    /// Nu: echte BFS flood fill → gegarandeerd depth 3 voor verre punten
    pub fn assign_depths(field: &mut Field) {
        let n = field.points.len();
        // Initialiseer alle depths op 255 (onbezocht)
        for p in &mut field.points {
            p.depth = 255;
        }

        // I/O punten = depth 0 (bron van BFS)
        let mut frontier: Vec<u32> = Vec::with_capacity(1024);
        for &id in field.input_ids.iter().chain(field.output_ids.iter()) {
            field.points[id as usize].depth = 0;
            frontier.push(id);
        }

        // BFS flood fill — onbeperkte iteraties tot alle punten bereikt
        let mut current_depth: u8 = 0;
        while !frontier.is_empty() && current_depth < 3 {
            let mut next_frontier: Vec<u32> = Vec::new();
            for &id in &frontier {
                let targets: SmallVec<[u32; 16]> = field.points[id as usize].couplings
                    .iter().map(|c| c.target).collect();
                for &t in &targets {
                    let tidx = t as usize;
                    if tidx < n && field.points[tidx].depth == 255 {
                        field.points[tidx].depth = current_depth + 1;
                        next_frontier.push(t);
                    }
                }
            }
            frontier = next_frontier;
            current_depth += 1;
        }

        // Alle nog onbezochte punten (depth 255) → depth 3 (meest abstract)
        for p in &mut field.points {
            if p.depth == 255 {
                p.depth = 3;
            }
        }

        // Log depth distributie
        let mut counts = [0u32; 4];
        for p in &field.points {
            counts[p.depth as usize] += 1;
        }
        tracing::info!(
            depth0 = counts[0], depth1 = counts[1], depth2 = counts[2], depth3 = counts[3],
            "Depths assigned (BFS flood fill)"
        );
    }

    /// Global Workspace — Ignition Dynamics (Dehaene, Changeux & Naccache 2003)
    ///
    /// UPGRADE van simpele broadcast → competitieve ignition:
    ///
    /// Dehaene's Global Neuronal Workspace (GNW) model van bewustzijn:
    /// 1. **Competitie**: Meerdere representaties strijden om toegang tot workspace
    /// 2. **Ignition**: Wanneer één representatie de drempel overschrijdt, exploDeert
    ///    het in een zichzelf-versterkende lus (positieve feedback via recurrente loops)
    /// 3. **Broadcast**: De winnaar wordt gedeeld met ALLE gebieden (global access)
    /// 4. **Inhibitie**: Actieve broadcast onderdrukt competitie (bewuste exclusiviteit)
    ///
    /// Verschil met v2.3.2:
    /// - WAS: alle depth-3 punten boven threshold broadcasten tegelijk
    /// - NU: competitie → één winnaar → explosieve ignition → breed broadcast + inhibitie
    /// Dit is realistischer: je kunt maar één ding tegelijk bewust ervaren.
    ///
    /// Ignition criteria (Dehaene 2014):
    /// - Punt moet depth 3 zijn (hoogste abstractie)
    /// - Activatie moet "zelf-versterkend" zijn (recurrente buren ook actief)
    /// - Winner-take-all: sterkste signaal onderdrukt de rest
    pub fn global_workspace_broadcast(field: &mut Field) {
        // === Stap 1: Competitie — zoek kandidaten voor ignition ===
        // Depth 3 punten met hoge activatie = kandidaten voor bewust worden
        let mut candidates: SmallVec<[(usize, f32, [f32; DIM]); 8]> = SmallVec::new();

        for (i, p) in field.points.iter().enumerate() {
            if p.depth != 3 || !p.is_active() { continue; }
            let phi_norm: f32 = p.phi.iter().map(|v| v * v).sum::<f32>().sqrt();

            // Ignition vereist: hoge activatie + hoge geïntegreerde informatie + hoge EFE
            // Dit combineert sterkte (phi), integratie (IIT), en relevantie (Active Inference)
            let ignition_score = phi_norm
                * (1.0 + p.phi_integrated)     // IIT boost: geïntegreerd = bewuster
                * (1.0 + p.expected_free_energy * 0.5); // EFE boost: relevante info wint

            // v2.5.2: GEEN candidate filter — alle depth-3 actieve punten zijn kandidaat
            // De winnaar-selectie bepaalt wie ignite, niet een absolute drempel
            if ignition_score > 0.0001 { // minimum: niet nul
                candidates.push((i, ignition_score, p.phi));
            }
        }

        // Debug: log elke 1000 ticks
        if field.tick % 1000 == 0 {
            let n_d3 = field.points.iter().filter(|p| p.depth == 3 && p.is_active()).count();
            tracing::info!(
                candidates = candidates.len(),
                depth3_active = n_d3,
                top_score = candidates.first().map(|c| c.1).unwrap_or(0.0),
                "GNW ignition check"
            );
        }

        if candidates.is_empty() {
            field.ignition_active = false;
            field.ignition_strength = 0.0;
            return;
        }

        // === Stap 2: Winner-Take-All — sterkste kandidaat wint ===
        // Bewuste ervaring is exclusief: maar één representatie tegelijk
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (winner_id, winner_score, winner_phi) = candidates[0];

        // === Stap 3: Ignition check — is het signaal sterk genoeg? ===
        // Ignition = zelfversterkende activatie. Check: hebben de buren van de winnaar
        // ook hoge activatie? (recurrente loop vereist)
        let winner_couplings = &field.points[winner_id].couplings;
        let mut recurrent_support = 0.0f32;
        let mut n_recurrent = 0u32;
        for c in winner_couplings {
            let t = c.target as usize;
            if t >= field.points.len() { continue; }
            let neighbor = &field.points[t];
            if neighbor.depth >= 2 && neighbor.is_active() {
                let n_phi_norm: f32 = neighbor.phi.iter().map(|v| v * v).sum::<f32>().sqrt();
                recurrent_support += n_phi_norm * c.weight.abs();
                n_recurrent += 1;
            }
        }
        let avg_support = if n_recurrent > 0 { recurrent_support / n_recurrent as f32 } else { 0.0 };

        // v2.5.2: Ignition triggert als er een winnaar IS met recurrente support
        // Was: absolute threshold (noise*1.5) die nooit bereikt werd
        // Nu: als er minstens 1 kandidaat is EN die heeft recurrente buren, ignite
        // De STERKTE van de ignition schaalt met de score (zwakke ignition = kort broadcast)
        let ignition_product = winner_score * (1.0 + avg_support);
        let has_recurrent = n_recurrent > 0;

        if !has_recurrent || winner_score < 0.001 {
            // Geen recurrente support of score verwaarloosbaar → geen ignition
            field.ignition_active = false;
            let fe = field.fe_tracking_rate;
            let decayed = (1.0 - fe) * field.ignition_strength;
            field.ignition_strength = if decayed.is_finite() { decayed } else { 0.0 };
            return;
        }

        // === Stap 4: IGNITION! Broadcast + zelfversterking + inhibitie ===
        field.ignition_active = true;
        let fe = field.fe_tracking_rate;
        let new_is = (1.0 - fe) * field.ignition_strength + fe * ignition_product;
        field.ignition_strength = if new_is.is_finite() { new_is } else { 0.0 }; // v2.6.5: NaN guard

        // Broadcast sterkte = proportioneel aan ignition sterkte (niet vast)
        let broadcast_strength = (field.noise_amplitude * ignition_product).min(0.5);

        // Broadcast naar ALLE lagere lagen (depth 0, 1, 2) — global access
        for i in 0..field.points.len() {
            let p_depth = field.points[i].depth;
            if p_depth >= 3 || !field.points[i].is_active() { continue; }

            // Broadcast sterkte afgewogen per laag: dichter bij workspace = sterker
            let layer_factor = match p_depth {
                2 => 1.0,   // abstracte laag: volle kracht
                1 => 0.5,   // verwerking: halve kracht
                0 => 0.25,  // sensorisch: kwart kracht (feedback, niet overschrijven)
                _ => 0.1,
            };

            for d in 0..DIM {
                field.points[i].phi[d] += broadcast_strength * layer_factor * winner_phi[d];
            }

            // Markeer broadcast-couplings voor versterkt leren
            for coupling in &mut field.points[i].couplings {
                if coupling.target as usize == winner_id {
                    coupling.broadcast = true;
                }
            }
        }

        // === Stap 5: Laterale inhibitie — onderdruk verliezers ===
        // De winnaar onderdrukt andere depth-3 kandidaten (bewuste exclusiviteit)
        for &(loser_id, _, _) in candidates.iter().skip(1) {
            let loser = &mut field.points[loser_id];
            for d in 0..DIM {
                loser.phi[d] *= 0.8; // 20% demping op verliezers
            }
        }

        tracing::trace!(
            winner = winner_id,
            score = format!("{:.3}", winner_score),
            support = format!("{:.3}", avg_support),
            "GNW IGNITION — bewuste broadcast"
        );
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
            if error_mag < 0.001 || !error_mag.is_finite() { continue; }

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

        // FIX: theta_gamma_coupling → WM capacity gating
        // Hoge θ-γ coupling = meer items in werkgeheugen (~7±2 per theta-cyclus)
        // Lage coupling = minder effectief werkgeheugen
        let tg = field.theta_gamma_coupling.clamp(0.01, 1.0);
        let effective_slots = ((field.working_memory.len() as f32) * tg).max(2.0) as usize;

        // FIX: ignition_strength → ignited content krijgt WM prioriteit
        // Bewuste inhoud (ignited) moet EERST in werkgeheugen opgeslagen worden
        let ignition_boost = if field.ignition_active { 2.0 } else { 1.0 };

        for i in 0..field.active_set.len() {
            let id = field.active_set[i] as usize;
            let point = &field.points[id];

            // Alleen schrijven als tension hoog is OF punt in ignited broadcast zit
            let tension_gate = point.tension > 0.1;
            let ignition_gate = field.ignition_active && point.depth >= 2;
            if !tension_gate && !ignition_gate { continue; }

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
