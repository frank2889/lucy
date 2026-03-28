//! # growth — Veldgroei, Meta-Leren en Energie-Economie
//!
//! Lucy's brein groeit en past zich aan. Dit module beheert drie processen:
//!
//! ## 1. Groei (maybe_grow)
//!
//! Als een punt persistent hoge spanning heeft en leren niet helpt,
//! wordt het gesplitst in twee kindpunten:
//! - Φ wordt geperturbeerd (±0.1 per dimensie) voor diversiteit
//! - Energie wordt gehalveerd (mitose-analogie)
//! - Spanning-EMA wordt gereset (schone lei)
//! - Kinderen worden onderling gekoppeld
//!
//! Hierna worden "dode" koppelingen (gewicht < 0.001) verwijderd (pruning).
//!
//! ## 2. Meta-Leren (maybe_meta_learn)
//!
//! Als spanning hoog maar stabiel is, past Lucy haar leersnelheid η aan:
//! - Spanning stijgt → η omhoog (sneller leren)
//! - Spanning stabiel → η omlaag (fijnere aanpassing)
//! - Bereik: 0.0001 ≤ η ≤ 0.1
//!
//! Dit is "leren hoe te leren" — elk punt optimaliseert zijn eigen leersnelheid.
//!
//! ## 3. Energie Management (manage_energy)
//!
//! Elk punt heeft beperkte energie — een metabolisme-analogie:
//! - **Kosten**: actief zijn kost 0.0001 per tick
//! - **Beloning**: dalende spanning (goede voorspelling) geeft energie terug
//! - **Sensorisch inkomen**: input-punten krijgen externe energie (0.005/tick)
//! - **Energie-pool**: gedeelde reserve die langzaam verdeeld wordt
//! - **Deactivatie**: punten op 0 energie worden inactief, geven energie terug aan pool
//! - **Propagatie**: hoge spanning activeert inactieve buren (rekrutering)
//! - **Spontane reactivatie**: als <10% actief, worden random punten gewekt
//!
//! Dit systeem voorkomt ongecontroleerde groei en dwingt efficiëntie af:
//! punten die niet bijdragen sterven uit, punten die nodig zijn worden geactiveerd.

use lucy_state::*;
use rand::Rng;

/// Groei: splits veldpunten waar spanning persistent hoog is en leren niet helpt.
pub fn maybe_grow(field: &mut Field) {
    let threshold = field.growth_threshold;
    let mut rng = rand::rng();
    let mut new_points: Vec<(usize, FieldPoint, FieldPoint)> = Vec::new();

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

            // Perturbatie voor diversiteit
            for d in 0..DIM {
                let delta = rng.random_range(-0.1..0.1);
                child_a.phi[d] += delta;
                child_b.phi[d] -= delta;
            }

            // Verdeel energie
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

        // Koppel de twee kinderen aan elkaar
        field.points[original_id].couplings.push(Coupling {
            target: new_id,
            weight: rng.random_range(-0.3..0.3),
            momentum: 0.0,
        });
        field.points[new_id as usize].couplings.push(Coupling {
            target: original_id as u32,
            weight: rng.random_range(-0.3..0.3),
            momentum: 0.0,
        });

        tracing::debug!(original_id, new_id, "Field point split");
    }

    // Prune: verwijder dode koppelingen (naar punten die niet meer bestaan of lang inactief)
    let n_points = field.points.len();
    for point in &mut field.points {
        point.couplings.retain(|c| {
            (c.target as usize) < n_points && c.weight.abs() > 0.001
        });
    }

    // Update active set
    field.rebuild_active_set();
}

/// Meta-leren: pas η aan als spanning niet daalt
pub fn maybe_meta_learn(field: &mut Field) {
    for &id in &field.active_set {
        let p = &mut field.points[id as usize];

        // Als spanning hoog en stabiel → pas leersnelheid aan
        if p.tension > field.growth_threshold * 0.5
            && (p.tension - p.tension_ema).abs() < field.growth_threshold * 0.2
        {
            // Escalatie niveau 1: η aanpassen
            // Als spanning niet daalt, probeer sneller of langzamer te leren
            let delta_eta = if p.tension > p.tension_ema {
                0.001 // spanning stijgt → leer sneller
            } else {
                -0.0005 // spanning stabiel → leer iets langzamer
            };
            p.eta = (p.eta + delta_eta).clamp(0.0001, 0.1);
        }
    }
}

/// Energie management: kosten, beloning, herverdeling, propagatie
pub fn manage_energy(field: &mut Field) {
    let activation_cost = 0.0001; // Verlaagd: 10x zuiniger
    let prediction_reward = 0.001; // Beloning voor dalende spanning
    let sensory_income = 0.005; // Sensorische punten krijgen energie van "buiten"
    let n_active = field.active_set.len() as f32;

    // === 1. Kosten en beloningen per punt ===
    for &id in &field.active_set {
        let p = &mut field.points[id as usize];

        // Kosten: actief zijn kost energie
        p.energy -= activation_cost;

        // Beloning: als spanning daalt (prediction verbetert), krijg energie terug
        // tension_ema > tension betekent spanning was hoger → het gaat beter
        if p.tension_ema > p.tension && p.tension_ema > 0.0 {
            let improvement = (p.tension_ema - p.tension).min(0.1);
            p.energy += prediction_reward * improvement;
        }

        // Sensorische punten krijgen energie van "buiten" (metabolisme)
        if p.is_sensory() {
            p.energy += sensory_income;
        }

        // Clamp
        p.energy = p.energy.clamp(0.0, 5.0);
    }

    // === 2. Energie-pool herverdeling ===
    // Een fractie van total_energy wordt elke tick verdeeld over actieve punten
    if n_active > 0.0 {
        let pool_injection = (field.total_energy * 0.0001).min(0.01 * n_active);
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
    let mut to_activate: Vec<u32> = Vec::new();
    for &id in &field.active_set {
        let p = &field.points[id as usize];
        if p.tension > field.growth_threshold * 0.3 {
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
    let active_ratio = field.active_set.len() as f32 / field.points.len().max(1) as f32;
    if active_ratio < 0.1 {
        // Minder dan 10% actief → wakker punten met meeste restenergie
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
