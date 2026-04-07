//! # meta — Meta-Cognitie en Escalatiedetectie
//!
//! Lucy's vermogen om over zichzelf na te denken. Dit module detecteert
//! wanneer het normale leerproces vastloopt en escaleert naar structurele
//! veranderingen.
//!
//! ## Escalatieniveaus
//!
//! Wanneer een punt hoge, stabiele spanning heeft (leren helpt niet):
//!
//! 1. **None** — spanning laag of nog dalend, geen actie nodig
//! 2. **MetaLearn** — pas leersnelheid η aan (zachte interventie)
//! 3. **Grow** — splits het punt in twee kinderen (structurele verandering)
//!
//! Toekomstig niveau 4: **CodeModify** — Lucy past haar eigen code aan.
//! Dit is het ultieme doel van vrije wil, maar vereist meer volwassenheid.
//!
//! ## Simulatie
//!
//! `simulate()` runt een hypothetische vooruit-simulatie van een punt
//! (zonder de echte staat te veranderen). Retourneert de verwachte
//! spanningsreductie. Wordt gebruikt voor "vooruitdenken" — kan dit
//! punt zichzelf oplossen zonder externe hulp?

use lucy_state::*;

/// Interne simulatie: kopieer een subset van het veld en reken hypothetisch door.
/// Retourneert de verwachte spanningsreductie van de simulatie.
/// Gebruikt voor counterfactueel denken: "wat als ik niets doe?"
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

    // v1.0: drempels uit AdaptiveParams i.p.v. hardcoded
    if p.tension <= field.growth_threshold * field.adaptive.escalation_tension_low {
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
            // Counterfactueel denken: "wat als ik niets doe?"
            let expected_reduction = simulate(field, id, 20);
            if expected_reduction > field.growth_threshold * 0.1 {
                // Simulatie voorspelt dat spanning zelf zal dalen → patience
                EscalationLevel::MetaLearn
            } else if p.frustration > field.adaptive.escalation_frust_thresh {
                // Persistente frustratie + groei helpt niet → pas jezelf aan
                EscalationLevel::CodeModify
            } else {
                // Simulatie voorspelt dat spanning blijft → groei nodig
                EscalationLevel::Grow
            }
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
    /// Niveau 4: Lucy past haar eigen hyperparameters of code aan
    CodeModify,
}
