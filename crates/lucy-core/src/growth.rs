//! # growth — Veldgroei, Meta-Leren en Energie-Economie
//!
//! Lucy's brein groeit en past zich aan. Dit module beheert drie processen:
//!
//! ## 1. Groei (maybe_grow)
//!
//! Als een punt persistent hoge spanning heeft en leren niet helpt,
//! wordt het gesplitst in twee kindpunten:
//! - Φ wordt geperturbeerd (±noise_amplitude per dimensie) voor diversiteit
//! - Energie wordt gehalveerd (mitose-analogie)
//! - Spanning-EMA wordt gereset (schone lei)
//! - Kinderen worden onderling gekoppeld (±clamp_range/10)
//!
//! Hierna worden "dode" koppelingen (gewicht < error_tracking_rate * 0.1) verwijderd (pruning).
//!
//! ## 2. Meta-Leren (maybe_meta_learn)
//!
//! Als spanning hoog maar stabiel is, past Lucy haar leersnelheid η aan:
//! - Spanning stijgt → η omhoog (proportioneel aan eta * error_tracking_rate)
//! - Spanning stabiel → η omlaag (half zo snel als omhoog)
//! - Bereik: [eta * 0.01, field.lambda]
//!
//! Dit is "leren hoe te leren" — elk punt optimaliseert zijn eigen leersnelheid.
//!
//! ## 3. Energie Management (manage_energy)
//!
//! Elk punt heeft beperkte energie — een metabolisme-analogie:
//! - **Kosten**: schalen mee met noise_amplitude (actief veld = duurder)
//! - **Beloning**: schalen mee met error_tracking_rate (snel leren = meer beloning)
//! - **Sensorisch inkomen**: schalen mee met noise_amplitude
//! - **Energie-pool**: injectie schaalt met noise_amplitude
//! - **Deactivatie**: punten op 0 energie worden inactief, geven energie terug aan pool
//! - **Propagatie**: spanning > growth_threshold activeert buren
//! - **Spontane reactivatie**: als activity ratio te laag, worden random punten gewekt

use lucy_state::*;
use rand::Rng;

/// Groei: splits veldpunten waar spanning persistent hoog is en leren niet helpt.
pub fn maybe_grow(field: &mut Field) {
    let threshold = field.growth_threshold;
    let mut rng = rand::rng();
    let mut new_points: Vec<(usize, FieldPoint, FieldPoint)> = Vec::new();

    // Perturbatie schaalt met noise_amplitude: meer exploratie-ruis → grotere mutaties bij split.
    // Dit zorgt dat kindpunten in een rustig veld dicht bij ouder blijven,
    // en in een turbulent veld meer diversiteit krijgen.
    let perturbation_range = field.noise_amplitude;

    // Koppelingssterkte schaalt met clamp_range/10: bredere dynamiek → sterkere initiële koppeling.
    // Factor 10 omdat clamp_range (1-50) het volle phi-bereik dekt; koppelingen moeten subtiel starten.
    let coupling_range = field.clamp_range / 10.0;

    // Pruning drempel schaalt met error_tracking_rate * 0.1:
    // als het veld errors sneller trackt, is het strenger op zwakke koppelingen.
    let prune_threshold = field.error_tracking_rate * 0.1;

    for &id in &field.active_set {
        let p = &field.points[id as usize];

        // Groei-conditie: spanning hoog EN trend vlak (leren helpt niet meer) EN energie beschikbaar
        let stuck = p.tension > threshold
            && (p.tension - p.tension_ema).abs() < threshold * 0.1
            && p.energy > 0.5;

        if stuck {
            // Split dit punt in twee subpunten
            let mut child_a = p.clone();
            let mut child_b = p.clone();

            // Perturbatie voor diversiteit — schaalt met noise_amplitude
            for d in 0..DIM {
                let delta = rng.random_range(-perturbation_range..perturbation_range);
                child_a.phi[d] += delta;
                child_b.phi[d] -= delta;
            }

            // Verdeel energie — 50/50 split (mitose: elk kind krijgt de helft)
            child_a.energy = p.energy * 0.5;
            child_b.energy = p.energy * 0.5;

            // Reset spanning (nieuw begin)
            child_a.tension_ema = 0.0;
            child_b.tension_ema = 0.0;

            new_points.push((id as usize, child_a, child_b));
        }
    }

    // Voer de splits uit
    for (original_id, child_a, child_b) in new_points {
        // Vervang het origineel met child_a
        field.points[original_id] = child_a;

        // Voeg child_b toe als nieuw punt
        let new_id = field.points.len() as u32;
        field.points.push(child_b);

        // Koppel de twee kinderen aan elkaar — sterkte schaalt met clamp_range/10
        field.points[original_id].couplings.push(Coupling {
            target: new_id,
            weight: rng.random_range(-coupling_range..coupling_range),
            momentum: 0.0,
            ..Default::default()
        });
        field.points[new_id as usize].couplings.push(Coupling {
            target: original_id as u32,
            weight: rng.random_range(-coupling_range..coupling_range),
            momentum: 0.0,
            ..Default::default()
        });

        tracing::debug!(original_id, new_id, "Field point split");
    }

    // Prune: verwijder dode koppelingen — drempel schaalt met error_tracking_rate
    let n_points = field.points.len();
    for point in &mut field.points {
        point.couplings.retain(|c| {
            (c.target as usize) < n_points && c.weight.abs() > prune_threshold
        });
    }

    // Update active set
    field.rebuild_active_set();
}

/// Meta-leren: pas η aan als spanning niet daalt.
/// η-aanpassingen zijn proportioneel aan het punt's eigen eta en de globale error_tracking_rate.
/// Dit voorkomt dat punten met kleine η te hard springen, en dat punten met grote η te weinig bewegen.
pub fn maybe_meta_learn(field: &mut Field) {
    let etr = field.error_tracking_rate;
    let eta_upper = field.lambda; // lambda is de veld-update schaalfactor — natuurlijk plafond voor leersnelheid

    for &id in &field.active_set {
        let p = &mut field.points[id as usize];

        // Als spanning hoog en stabiel → pas leersnelheid aan
        if p.tension > field.growth_threshold * 0.5
            && (p.tension - p.tension_ema).abs() < field.growth_threshold * 0.2
        {
            // η-delta proportioneel aan huidige eta * error_tracking_rate:
            // - Grote eta + snel trackend veld → grotere stappen (het veld is actief bezig)
            // - Kleine eta + langzaam veld → fijnere aanpassing
            // Asymmetrie: omhoog 2x sneller dan omlaag, omdat vastzitten duurder is dan overshoot
            let delta_eta = if p.tension > p.tension_ema {
                // spanning stijgt → leer sneller
                p.eta * etr
            } else {
                // spanning stabiel → leer iets langzamer (half zo snel als versnelling)
                -(p.eta * etr * 0.5)
            };

            // Clamp: ondergrens = 1% van huidige eta (nooit helemaal stil),
            // bovengrens = field.lambda (leersnelheid mag niet boven veld-update schaalfactor)
            let eta_lower = p.eta * 0.01;
            p.eta = (p.eta + delta_eta).clamp(eta_lower, eta_upper);
        }
    }
}

/// Energie management: kosten, beloning, herverdeling, propagatie.
/// Alle kostenparameters schalen met velddynamiek zodat ze meegroeien/krimpen
/// met Lucy's activiteitsniveau.
pub fn manage_energy(field: &mut Field) {
    // Activatiekosten schalen met noise_amplitude:
    // meer exploratie-ruis = actiever veld = hogere metabolische kosten.
    // Factor 0.01 houdt het in dezelfde orde als de oude 0.0001 bij noise_amplitude=0.01.
    let activation_cost = field.noise_amplitude * 0.01;

    // Predictiebeloning schaalt met error_tracking_rate:
    // veld dat errors sneller trackt, beloont verbetering proportioneel meer.
    // Factor 0.1 houdt het in dezelfde orde als oude 0.001 bij etr=0.01.
    let prediction_reward = field.error_tracking_rate * 0.1;

    // Sensorisch inkomen schaalt met noise_amplitude:
    // actief explorerend veld investeert meer in sensorische input.
    // Factor 0.5 houdt het in orde van oude 0.005 bij noise_amplitude=0.01.
    let sensory_income = field.noise_amplitude * 0.5;

    let n_active = field.active_set.len() as f32;

    // === 1. Kosten en beloningen per punt ===
    for &id in &field.active_set {
        let p = &mut field.points[id as usize];

        // Kosten: actief zijn kost energie — schaalt met velddynamiek
        p.energy -= activation_cost;

        // Beloning: als spanning daalt (prediction verbetert), krijg energie terug
        // tension_ema > tension betekent spanning was hoger → het gaat beter
        if p.tension_ema > p.tension && p.tension_ema > 0.0 {
            let improvement = (p.tension_ema - p.tension).min(0.1);
            p.energy += prediction_reward * improvement;
        }

        // Emotie-gestuurde energie: curiosity geeft extra energie (motivatie om te leren)
        // Schaalt met noise_amplitude zodat in actieve fase emoties meer impact hebben
        p.energy += p.curiosity * field.noise_amplitude * 0.1;
        // Satisfaction geeft een kleine bonus (beloning voor goed werk)
        p.energy += p.satisfaction * field.noise_amplitude * 0.05;

        // Sensorische punten krijgen energie van "buiten" (metabolisme)
        if p.is_sensory() {
            p.energy += sensory_income;
        }

        // Clamp
        p.energy = p.energy.clamp(0.0, 5.0);
    }

    // === 2. Energie-pool herverdeling ===
    // Injectie schaalt met noise_amplitude: actiever veld herverdeelt sneller.
    // Factor noise_amplitude * 0.01 vervangt oude 0.0001; max per punt = noise_amplitude.
    if n_active > 0.0 {
        let pool_injection = (field.total_energy * field.noise_amplitude * 0.01)
            .min(field.noise_amplitude * n_active);
        let per_point = pool_injection / n_active;
        for &id in &field.active_set {
            field.points[id as usize].energy += per_point;
        }
        // Pool raakt langzaam op maar wordt aangevuld door deactivatie
        field.total_energy -= pool_injection;
        field.total_energy = field.total_energy.max(0.0);
    }

    // === 3. Deactiveer lege punten, reclaim energie naar pool ===
    for &id in &field.active_set {
        let p = &mut field.points[id as usize];
        if p.energy <= 0.0 {
            p.energy = 0.0;
            p.flags &= !FLAG_ACTIVE;
            // Geef restenergie terug aan pool
            field.total_energy += 0.1;
        }
    }

    // === 4. Propagatie: spanning activeert buren ===
    // Drempel = growth_threshold zelf (niet een willekeurige 0.3 fractie).
    // Punt met spanning boven groeidrempel is "gestressed" genoeg om hulp te rekruteren.
    let propagation_threshold = field.growth_threshold;
    let mut to_activate: Vec<u32> = Vec::new();
    for &id in &field.active_set {
        let p = &field.points[id as usize];
        if p.tension > propagation_threshold {
            for c in &p.couplings {
                let target = c.target as usize;
                if target < field.points.len() && !field.points[target].is_active() {
                    to_activate.push(c.target);
                }
            }
        }
    }

    for id in to_activate {
        let p = &mut field.points[id as usize];
        if p.energy <= 0.0 {
            p.energy = 0.3; // Genoeg energie om een tijdje actief te blijven
        }
        p.flags |= FLAG_ACTIVE;
    }

    // === 5. Spontane heractivatie: als te weinig actief, wakker random punten ===
    // Drempel relatief aan totaal aantal punten in plaats van hardcoded 0.1.
    // settle_epsilon is de convergentiedrempel — als het veld makkelijker settlet,
    // is er minder spontane activatie nodig (hogere epsilon = relaxer veld).
    let min_active_ratio = field.settle_epsilon.clamp(0.05, 0.2);
    let active_ratio = field.active_set.len() as f32 / field.points.len().max(1) as f32;
    if active_ratio < min_active_ratio {
        let mut rng = rand::rng();
        use rand::Rng;
        let n_wake = (field.points.len() / 20).max(5); // wakker ~5% of minstens 5
        for _ in 0..n_wake {
            let idx = rng.random_range(0..field.points.len());
            if !field.points[idx].is_active() {
                field.points[idx].energy = 0.5;
                field.points[idx].flags |= FLAG_ACTIVE;
                field.total_energy -= 0.5;
                if field.total_energy <= 0.0 {
                    break;
                }
            }
        }
    }

    field.rebuild_active_set();
}
