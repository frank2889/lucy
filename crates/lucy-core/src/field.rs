//! # field — Veld Bootstrap met Small-World Topologie
//!
//! Creëert Lucy's initiële breinstructuur: een Watts-Strogatz small-world
//! netwerk van 2176 punten (64 input + 2048 hidden + 64 output).
//!
//! ## Small-World Netwerk (Watts & Strogatz, 1998)
//!
//! De hidden-regio is georganiseerd als een ring met twee soorten verbindingen:
//!
//! - **Lokaal**: elke node verbonden met K/2=8 buren links en rechts
//!   → dichte lokale clusters voor gedetailleerde berekeningen
//! - **Long-range**: 5% kans op willekeurige verre verbinding
//!   → korte padlengtes voor snelle informatie-propagatie
//!
//! Dit geeft het beste van twee werelden: lokale specialisatie + globale integratie.
//! Vergelijkbaar met de neocortex: kolommen (lokaal) + witte stof (long-range).
//!
//! ## Regio's en Connectiviteit
//!
//! ```text
//! Input (0..63) →→ eerste kwart Hidden (64..575)
//!                         ↕ (small-world ring)
//!                   Hidden (64..2111)
//!                         ↕ (small-world ring)
//!                   laatste kwart Hidden (1600..2111) →→ Output (2112..2175)
//!                                                    ←← (bidirectioneel)
//! ```
//!
//! - Input → Hidden: 32 verbindingen per input-punt naar eerste kwart
//! - Hidden ↔ Hidden: K=16 lokale buren + rare long-range shortcuts
//! - Hidden → Output: 32 bidirectionele verbindingen vanuit laatste kwart
//! - Recurrente loops: ~205 willekeurige bidirectionele feedback-paden
//!
//! ## Inhibitie
//!
//! 20% van hidden-punten is inhibitorisch (negatieve gewichten).
//! Dit zorgt voor competitie en voorkomt dat alle punten hetzelfde berekenen.
//! Vergelijkbaar met GABAergische interneuronen in de hersenen.
//!
//! ## Initialisatie
//!
//! - Φ: uniform random [-0.1, 0.1] (symmetrie-breking)
//! - η: uniform random [0.005, 0.02] (diversiteit in leersnelheid)
//! - Gewichten: uniform random, bereik hangt af van verbindingstype
//! - Energie: input/output = 5.0 (hoog, altijd actief), hidden = 1.0

use lucy_state::*;
use rand::Rng;

/// v2.5: Bootstrap configuratie — Lucy kan haar eigen brein-architectuur kiezen.
/// Defaults = oude hardcoded waarden. Lucy kan via data/field_config.json aanpassen.
#[derive(Clone, Debug)]
pub struct FieldConfig {
    pub n_input: usize,
    pub n_hidden: usize,
    pub n_output: usize,
    pub local_k: usize,
    pub long_range_p: f32,
    pub inhibitory_frac: f32,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            n_input: 256,      // gelijk aan DIM voor 1:1 mapping
            n_hidden: 8192,    // het denkvermogen — ~100K koppelingen
            n_output: 256,     // gelijk aan DIM voor decodering
            local_k: 20,       // v2.6.5: meer couplings = meer geheugen per neuron (was 12)
            long_range_p: 0.05, // kans op small-world shortcut
            inhibitory_frac: 0.2, // fractie inhibitorische punten
        }
    }
}

impl FieldConfig {
    /// Laad configuratie uit bestand, of gebruik defaults
    pub fn load_or_default() -> Self {
        let path = std::path::Path::new("data/field_config.json");
        if path.exists() {
            if let Ok(text) = std::fs::read_to_string(path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                    return Self {
                        n_input: val["n_input"].as_u64().unwrap_or(256) as usize,
                        n_hidden: val["n_hidden"].as_u64().unwrap_or(8192) as usize,
                        n_output: val["n_output"].as_u64().unwrap_or(256) as usize,
                        local_k: val["local_k"].as_u64().unwrap_or(12) as usize,
                        long_range_p: val["long_range_p"].as_f64().unwrap_or(0.05) as f32,
                        inhibitory_frac: val["inhibitory_frac"].as_f64().unwrap_or(0.2) as f32,
                    };
                }
            }
        }
        Self::default()
    }
}

// Helper functies voor rand 0.9 API (edition 2024 reserveert `gen` keyword)
fn rng_f32(rng: &mut impl Rng) -> f32 { rng.random::<f32>() }
fn rng_range(rng: &mut impl Rng, lo: f32, hi: f32) -> f32 { rng.random_range(lo..hi) }
fn rng_usize(rng: &mut impl Rng, lo: usize, hi: usize) -> usize { rng.random_range(lo..hi) }

/// Creëer een vers veld met small-world topologie.
/// Dit wordt aangeroepen bij eerste start of als snapshot corrupt is.
pub fn bootstrap_field() -> Field {
    let config = FieldConfig::load_or_default();
    bootstrap_field_with_config(&config)
}

pub fn bootstrap_field_with_config(config: &FieldConfig) -> Field {
    let mut field = Field::new();
    let mut rng = rand::rng();

    let n_input = config.n_input;
    let n_hidden = config.n_hidden;
    let n_output = config.n_output;

    let local_k = config.local_k;
    let long_range_p = config.long_range_p;
    let inhibitory_frac = config.inhibitory_frac;

    field.n_input = n_input;
    field.n_hidden = n_hidden;
    field.n_output = n_output;
    field.snapshot_version = 1;

    let total = n_input + n_hidden + n_output;
    field.points.reserve(total);

    // Input regio
    for _ in 0..n_input {
        let mut p = FieldPoint::new();
        p.flags = FLAG_ACTIVE | FLAG_INPUT | FLAG_SENSORY;
        p.energy = 5.0;
        field.points.push(p);
    }

    // Hidden regio
    for _ in 0..n_hidden {
        let mut p = FieldPoint::new();
        p.flags = FLAG_ACTIVE;
        for d in 0..DIM {
            p.phi[d] = rng_range(&mut rng, -0.1, 0.1);
        }
        if rng_f32(&mut rng) < inhibitory_frac {
            p.flags |= FLAG_INHIBITORY;
        }
        p.energy = 1.0;
        p.eta = rng_range(&mut rng, 0.005, 0.02);
        field.points.push(p);
    }

    // Output regio
    for _ in 0..n_output {
        let mut p = FieldPoint::new();
        p.flags = FLAG_ACTIVE | FLAG_OUTPUT | FLAG_MOTOR;
        p.energy = 5.0;
        field.points.push(p);
    }

    let hidden_start = n_input;
    let output_start = n_input + n_hidden;

    // Input → eerste kwart hidden
    for i in 0..n_input {
        for _ in 0..32 {
            let target = hidden_start + rng_usize(&mut rng, 0, n_hidden / 4);
            field.points[i].couplings.push(Coupling {
                target: target as u32,
                weight: rng_range(&mut rng, -0.3, 0.3),
                momentum: 0.0,
                ..Default::default()
            });
        }
    }

    // Hidden: Watts-Strogatz small-world
    for i in 0..n_hidden {
        let idx = hidden_start + i;
        let is_inhib = field.points[idx].is_inhibitory();

        for k in 1..=local_k / 2 {
            let left = hidden_start + (i + n_hidden - k) % n_hidden;
            let right = hidden_start + (i + k) % n_hidden;
            let w = if is_inhib {
                rng_range(&mut rng, -0.5, -0.1)
            } else {
                rng_range(&mut rng, 0.1, 0.5)
            };
            field.points[idx].couplings.push(Coupling {
                target: left as u32, weight: w, momentum: 0.0, ..Default::default()
            });
            field.points[idx].couplings.push(Coupling {
                target: right as u32, weight: w * 0.8, momentum: 0.0, ..Default::default()
            });
        }

        for _ in 0..4 {
            if rng_f32(&mut rng) < long_range_p {
                let far = hidden_start + rng_usize(&mut rng, 0, n_hidden);
                if far != idx {
                    field.points[idx].couplings.push(Coupling {
                        target: far as u32,
                        weight: rng_range(&mut rng, -0.2, 0.2),
                        momentum: 0.0,
                        ..Default::default()
                    });
                }
            }
        }
    }

    // Laatste kwart hidden → Output
    let last_q = hidden_start + n_hidden * 3 / 4;
    for i in 0..n_output {
        let oidx = output_start + i;
        for _ in 0..32 {
            let source = last_q + rng_usize(&mut rng, 0, n_hidden / 4);
            field.points[oidx].couplings.push(Coupling {
                target: source as u32,
                weight: rng_range(&mut rng, -0.3, 0.3),
                momentum: 0.0,
                ..Default::default()
            });
            field.points[source].couplings.push(Coupling {
                target: oidx as u32,
                weight: rng_range(&mut rng, -0.15, 0.15),
                momentum: 0.0,
                ..Default::default()
            });
        }
    }

    // Recurrente feedback loops
    for _ in 0..n_hidden / 10 {
        let a = hidden_start + rng_usize(&mut rng, 0, n_hidden);
        let b = hidden_start + rng_usize(&mut rng, 0, n_hidden);
        if a != b {
            let w = rng_range(&mut rng, -0.1, 0.1);
            field.points[a].couplings.push(Coupling {
                target: b as u32, weight: w, momentum: 0.0, ..Default::default()
            });
            field.points[b].couplings.push(Coupling {
                target: a as u32, weight: w * 0.5, momentum: 0.0, ..Default::default()
            });
        }
    }

    field.total_energy = total as f32;
    field.rebuild_active_set();
    field.cache_io_ids();

    let n_couplings: usize = field.points.iter().map(|p| p.couplings.len()).sum();
    tracing::info!(
        points = total,
        input = n_input,
        hidden = n_hidden,
        output = n_output,
        couplings = n_couplings,
        "Field bootstrapped"
    );

    field
}

/// Herbedrading van een bestaand veld — alleen couplings worden opnieuw aangemaakt.
/// Phi, memory, prediction, emoties, vocab, retrieval etc. blijven INTACT.
/// Wordt aangeroepen als couplings verloren zijn (bijv. door homeostasis death spiral).
pub fn rewire_field(field: &mut Field) {
    let config = FieldConfig::load_or_default();
    let mut rng = rand::rng();

    let n_input = config.n_input.min(field.n_input);
    let n_hidden = config.n_hidden.min(field.n_hidden);
    let n_output = config.n_output.min(field.n_output);
    let hidden_start = n_input;
    let output_start = n_input + n_hidden;
    let local_k = config.local_k;
    let long_range_p = config.long_range_p;

    // Wis alle bestaande (lege) couplings
    for p in &mut field.points {
        p.couplings.clear();
    }

    // Input → eerste kwart hidden
    for i in 0..n_input {
        for _ in 0..32 {
            let target = hidden_start + rng_usize(&mut rng, 0, n_hidden / 4);
            field.points[i].couplings.push(Coupling {
                target: target as u32,
                weight: rng_range(&mut rng, -0.3, 0.3),
                momentum: 0.0,
                ..Default::default()
            });
        }
    }

    // Hidden: Watts-Strogatz small-world
    for i in 0..n_hidden {
        let idx = hidden_start + i;
        let is_inhib = field.points[idx].is_inhibitory();

        for k in 1..=local_k / 2 {
            let left = hidden_start + (i + n_hidden - k) % n_hidden;
            let right = hidden_start + (i + k) % n_hidden;
            let w = if is_inhib {
                rng_range(&mut rng, -0.5, -0.1)
            } else {
                rng_range(&mut rng, 0.1, 0.5)
            };
            field.points[idx].couplings.push(Coupling {
                target: left as u32, weight: w, momentum: 0.0, ..Default::default()
            });
            field.points[idx].couplings.push(Coupling {
                target: right as u32, weight: w * 0.8, momentum: 0.0, ..Default::default()
            });
        }

        for _ in 0..4 {
            if rng_f32(&mut rng) < long_range_p {
                let far = hidden_start + rng_usize(&mut rng, 0, n_hidden);
                if far != idx {
                    field.points[idx].couplings.push(Coupling {
                        target: far as u32,
                        weight: rng_range(&mut rng, -0.2, 0.2),
                        momentum: 0.0,
                        ..Default::default()
                    });
                }
            }
        }
    }

    // Laatste kwart hidden → Output
    let last_q = hidden_start + n_hidden * 3 / 4;
    for i in 0..n_output {
        let oidx = output_start + i;
        for _ in 0..32 {
            let source = last_q + rng_usize(&mut rng, 0, n_hidden / 4);
            field.points[oidx].couplings.push(Coupling {
                target: source as u32,
                weight: rng_range(&mut rng, -0.3, 0.3),
                momentum: 0.0,
                ..Default::default()
            });
            field.points[source].couplings.push(Coupling {
                target: oidx as u32,
                weight: rng_range(&mut rng, -0.15, 0.15),
                momentum: 0.0,
                ..Default::default()
            });
        }
    }

    // Recurrente feedback loops
    for _ in 0..n_hidden / 10 {
        let a = hidden_start + rng_usize(&mut rng, 0, n_hidden);
        let b = hidden_start + rng_usize(&mut rng, 0, n_hidden);
        if a != b {
            let w = rng_range(&mut rng, -0.1, 0.1);
            field.points[a].couplings.push(Coupling {
                target: b as u32, weight: w, momentum: 0.0, ..Default::default()
            });
            field.points[b].couplings.push(Coupling {
                target: a as u32, weight: w * 0.5, momentum: 0.0, ..Default::default()
            });
        }
    }

    let n_couplings: usize = field.points.iter().map(|p| p.couplings.len()).sum();
    tracing::info!(
        couplings = n_couplings,
        "Field rewired — couplings hersteld, phi/memory intact"
    );
}

/// v2.6.5: Voeg extra couplings toe aan punten die te weinig hebben.
/// Bestaande couplings blijven intact — alleen NIEUWE worden toegevoegd.
/// Dit vergroot het geheugen per neuron zonder geleerde weights te verliezen.
pub fn densify_couplings(field: &mut Field, target_k: usize) {
    use rand::Rng;
    let mut rng = rand::rng();
    let n = field.points.len();
    let mut added = 0usize;

    for i in 0..n {
        let current = field.points[i].couplings.len();
        if current >= target_k { continue; }

        // Bestaande targets — voorkom duplicaten
        let existing: std::collections::HashSet<u32> = field.points[i]
            .couplings.iter().map(|c| c.target).collect();

        let to_add = target_k - current;
        for _ in 0..to_add {
            let target = rng.random_range(0..n as u32);
            if target != i as u32 && !existing.contains(&target) {
                field.points[i].couplings.push(lucy_state::Coupling {
                    target,
                    weight: rng.random_range(-0.01f32..0.01),
                    momentum: 0.0,
                    importance: 0.0,
                    consolidated: 0.0,
                    broadcast: false,
                });
                added += 1;
            }
        }
    }

    let total: usize = field.points.iter().map(|p| p.couplings.len()).sum();
    tracing::info!(added, total, target_k, "Couplings verdicht — meer geheugen per neuron");
}
