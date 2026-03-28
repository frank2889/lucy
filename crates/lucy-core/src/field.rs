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

/// Aantal input-punten — gelijk aan DIM voor 1:1 mapping met embeddings
const N_INPUT: usize = 64;
/// Aantal hidden-punten — het denkvermogen. 2048 geeft ~70K koppelingen.
const N_HIDDEN: usize = 2048;
/// Aantal output-punten — gelijk aan DIM voor decodering via dot product
const N_OUTPUT: usize = 64;
/// Watts-Strogatz K parameter: elke node verbonden met K dichtstbijzijnde buren
const LOCAL_K: usize = 16;
/// Kans op long-range verbinding per slot (small-world shortcut)
const LONG_RANGE_P: f32 = 0.05;
/// Fractie inhibitorische punten (negatieve gewichten)
const INHIBITORY_FRAC: f32 = 0.2;

// Helper functies voor rand 0.9 API (edition 2024 reserveert `gen` keyword)
fn rng_f32(rng: &mut impl Rng) -> f32 { rng.random::<f32>() }
fn rng_range(rng: &mut impl Rng, lo: f32, hi: f32) -> f32 { rng.random_range(lo..hi) }
fn rng_usize(rng: &mut impl Rng, lo: usize, hi: usize) -> usize { rng.random_range(lo..hi) }

/// Creëer een vers veld met small-world topologie.
/// Dit wordt aangeroepen bij eerste start of als snapshot corrupt is.
pub fn bootstrap_field() -> Field {
    let mut field = Field::new();
    let mut rng = rand::rng();

    let total = N_INPUT + N_HIDDEN + N_OUTPUT;
    field.points.reserve(total);

    // Input regio
    for _ in 0..N_INPUT {
        let mut p = FieldPoint::new();
        p.flags = FLAG_ACTIVE | FLAG_INPUT | FLAG_SENSORY;
        p.energy = 5.0;
        field.points.push(p);
    }

    // Hidden regio
    for _ in 0..N_HIDDEN {
        let mut p = FieldPoint::new();
        p.flags = FLAG_ACTIVE;
        for d in 0..DIM {
            p.phi[d] = rng_range(&mut rng, -0.1, 0.1);
        }
        if rng_f32(&mut rng) < INHIBITORY_FRAC {
            p.flags |= FLAG_INHIBITORY;
        }
        p.energy = 1.0;
        p.eta = rng_range(&mut rng, 0.005, 0.02);
        field.points.push(p);
    }

    // Output regio
    for _ in 0..N_OUTPUT {
        let mut p = FieldPoint::new();
        p.flags = FLAG_ACTIVE | FLAG_OUTPUT | FLAG_MOTOR;
        p.energy = 5.0;
        field.points.push(p);
    }

    let hidden_start = N_INPUT;
    let output_start = N_INPUT + N_HIDDEN;

    // Input → eerste kwart hidden
    for i in 0..N_INPUT {
        for _ in 0..32 {
            let target = hidden_start + rng_usize(&mut rng, 0, N_HIDDEN / 4);
            field.points[i].couplings.push(Coupling {
                target: target as u32,
                weight: rng_range(&mut rng, -0.3, 0.3),
                momentum: 0.0,
            });
        }
    }

    // Hidden: Watts-Strogatz small-world
    for i in 0..N_HIDDEN {
        let idx = hidden_start + i;
        let is_inhib = field.points[idx].is_inhibitory();

        for k in 1..=LOCAL_K / 2 {
            let left = hidden_start + (i + N_HIDDEN - k) % N_HIDDEN;
            let right = hidden_start + (i + k) % N_HIDDEN;
            let w = if is_inhib {
                rng_range(&mut rng, -0.5, -0.1)
            } else {
                rng_range(&mut rng, 0.1, 0.5)
            };
            field.points[idx].couplings.push(Coupling {
                target: left as u32, weight: w, momentum: 0.0,
            });
            field.points[idx].couplings.push(Coupling {
                target: right as u32, weight: w * 0.8, momentum: 0.0,
            });
        }

        for _ in 0..4 {
            if rng_f32(&mut rng) < LONG_RANGE_P {
                let far = hidden_start + rng_usize(&mut rng, 0, N_HIDDEN);
                if far != idx {
                    field.points[idx].couplings.push(Coupling {
                        target: far as u32,
                        weight: rng_range(&mut rng, -0.2, 0.2),
                        momentum: 0.0,
                    });
                }
            }
        }
    }

    // Laatste kwart hidden → Output
    let last_q = hidden_start + N_HIDDEN * 3 / 4;
    for i in 0..N_OUTPUT {
        let oidx = output_start + i;
        for _ in 0..32 {
            let source = last_q + rng_usize(&mut rng, 0, N_HIDDEN / 4);
            field.points[oidx].couplings.push(Coupling {
                target: source as u32,
                weight: rng_range(&mut rng, -0.3, 0.3),
                momentum: 0.0,
            });
            field.points[source].couplings.push(Coupling {
                target: oidx as u32,
                weight: rng_range(&mut rng, -0.15, 0.15),
                momentum: 0.0,
            });
        }
    }

    // Recurrente feedback loops
    for _ in 0..N_HIDDEN / 10 {
        let a = hidden_start + rng_usize(&mut rng, 0, N_HIDDEN);
        let b = hidden_start + rng_usize(&mut rng, 0, N_HIDDEN);
        if a != b {
            let w = rng_range(&mut rng, -0.1, 0.1);
            field.points[a].couplings.push(Coupling {
                target: b as u32, weight: w, momentum: 0.0,
            });
            field.points[b].couplings.push(Coupling {
                target: a as u32, weight: w * 0.5, momentum: 0.0,
            });
        }
    }

    field.total_energy = total as f32;
    field.rebuild_active_set();
    field.cache_io_ids();

    let n_couplings: usize = field.points.iter().map(|p| p.couplings.len()).sum();
    tracing::info!(
        points = total,
        input = N_INPUT,
        hidden = N_HIDDEN,
        output = N_OUTPUT,
        couplings = n_couplings,
        "Field bootstrapped"
    );

    field
}
