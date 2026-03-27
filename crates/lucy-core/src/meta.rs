use lucy_state::*;

/// Interne simulatie: kopieer een subset van het veld en reken hypothetisch door
/// Retourneert de verwachte spanningsreductie van de simulatie
pub fn simulate(field: &Field, point_id: u32, steps: usize) -> f32 {
    let point = &field.points[point_id as usize];

    // Kopieer alleen dit punt en zijn directe buren (efficiënt)
    let mut sim_phi = point.phi;
    let mut sim_prediction = point.prediction;
    let initial_tension = point.tension;

    // Simuleer N stappen vooruit
    for _ in 0..steps {
        // Bereken prediction error
        let mut eps_sq = 0.0f32;
        for d in 0..DIM {
            let e = sim_phi[d] - sim_prediction[d];
            eps_sq += e * e;
        }

        // Simpele Φ update (zonder echte buren, alleen eigen dynamiek)
        for d in 0..DIM {
            sim_prediction[d] = sim_phi[d];
            sim_phi[d] -= field.alpha * sim_phi[d].signum(); // sparse penalty
        }

        // Als spanning al laag genoeg, stop vroeg
        if eps_sq < 0.01 {
            break;
        }
    }

    // Bereken einSpanning
    let mut final_tension = 0.0f32;
    for d in 0..DIM {
        let e = sim_phi[d] - sim_prediction[d];
        final_tension += e * e;
    }

    // Positief = spanning verminderd, negatief = spanning verhoogd
    initial_tension - final_tension
}

/// Vastloop-escalatie: als leren niet werkt, escaleer naar structurele verandering
pub fn check_escalation(field: &Field, id: u32) -> EscalationLevel {
    let p = &field.points[id as usize];

    if p.tension <= field.growth_threshold * 0.3 {
        return EscalationLevel::None;
    }

    let trend_flat = (p.tension - p.tension_ema).abs() < field.growth_threshold * 0.1;

    if !trend_flat {
        // Spanning is hoog maar verandert nog → leren werkt, geen escalatie
        return EscalationLevel::None;
    }

    // Spanning hoog EN stabiel → escaleer
    if p.eta > 0.05 {
        // η is al hoog aangestuurd → meta-leren heeft niet gewerkt
        if p.energy > 0.5 {
            EscalationLevel::Grow // Niveau 3: splits
        } else {
            EscalationLevel::MetaLearn // Niveau 2: verander leerregel nog meer
        }
    } else {
        EscalationLevel::MetaLearn // Niveau 1: pas η aan
    }
}

#[derive(Debug, PartialEq)]
pub enum EscalationLevel {
    None,
    MetaLearn,
    Grow,
    // CodeModify zou niveau 4 zijn — dat komt later wanneer Lucy er klaar voor is
}
