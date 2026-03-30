//! # curriculum — Lucy's Leerstof
//!
//! Lucy leert niet random — ze heeft een curriculum. Gestructureerde leerstof
//! die haar stap voor stap slimmer maakt.
//!
//! ## Niveaus
//!
//! 1. **Eigen code** — Rust syntax, haar eigen architectuur (al actief)
//! 2. **HTML/CSS** — zodat ze haar dashboard kan begrijpen en aanpassen
//! 3. **Taalpatronen** — zinnen, grammatica, semantiek
//! 4. **Webinhoud** — Wikipedia, DuckDuckGo resultaten (al actief)
//!
//! ## HTML Training
//!
//! Lucy leert HTML door:
//! 1. Haar eigen dashboard.html te lezen en te leren voorspellen
//! 2. HTML snippets als training paren (input → output)
//! 3. Patronen: tag openen → content → tag sluiten

use lucy_state::*;
use crate::encoding::Vocabulary;
use crate::compute;

/// HTML trainingsvoorbeelden — van simpel naar complex
const HTML_LESSONS: &[(&str, &str)] = &[
    // Level 1: basis tags
    ("<h1>", "titel</h1>"),
    ("<p>", "tekst</p>"),
    ("<div>", "inhoud</div>"),
    ("<div class=\"card\">", "<h2>Titel</h2></div>"),
    ("<span class=\"value\">", "0</span>"),

    // Level 2: geneste structuur
    ("<div class=\"card\"><h2>", "Status</h2><div class=\"stat\">waarde</div></div>"),
    ("<div class=\"stat\"><span class=\"label\">", "Naam</span><span class=\"value\">0</span></div>"),
    ("<div class=\"grid\">", "<div class=\"card\">inhoud</div></div>"),

    // Level 3: JavaScript patronen
    ("document.getElementById('", "element').textContent = waarde;"),
    ("fetch(API + '/", "status').then(r => r.json())"),
    ("const res = await fetch(", "API + '/status');"),

    // Level 4: dashboard-specifieke patronen
    ("<div class=\"card\"><h2>Nieuw</h2>", "<div class=\"stat\"><span class=\"label\">Label</span><span class=\"value\" id=\"new-val\">-</span></div></div>"),
    ("document.getElementById('new-val').textContent = ", "d.field?.points || '-';"),

    // Level 5: CSS
    (".card { background:", " #12121a; border: 1px solid #222; border-radius: 8px; padding: 15px; }"),
    (".value { color:", " #fff; font-weight: bold; }"),
];

/// Train Lucy op het HTML curriculum
pub fn train_html(field: &mut Field, vocab: &mut Vocabulary) -> usize {
    let mut total_trained = 0;

    for (input, target) in HTML_LESSONS {
        let input_embs = crate::encoding::encode_text(vocab, input);
        let target_embs = crate::encoding::encode_text(vocab, target);

        if input_embs.is_empty() || target_embs.is_empty() {
            continue;
        }

        compute::equilibrium_propagation(
            field,
            &input_embs,
            &target_embs,
            2,  // ticks per token
            10, // settle ticks
            0.3,
        );

        total_trained += input_embs.len() + target_embs.len();
    }

    total_trained
}

/// Train Lucy op haar eigen dashboard (ze leert haar eigen code voorspellen)
pub fn train_on_own_dashboard(field: &mut Field, vocab: &mut Vocabulary) -> usize {
    let dashboard = match std::fs::read_to_string("data/dashboard.html") {
        Ok(html) => html,
        Err(_) => return 0,
    };

    // Split in chunks en train via sequential prediction
    let token_ids = vocab.tokenize(&dashboard);
    if token_ids.len() < 6 { return 0; }

    let mut trained = 0;
    let chunk_size = 8;

    for chunk in token_ids.windows(chunk_size).step_by(chunk_size / 2) {
        if chunk.len() < 4 { break; }

        let mid = chunk.len() / 2;
        let input_embs: Vec<[f32; DIM]> = chunk[..mid].iter()
            .enumerate()
            .map(|(pos, &id)| {
                let mut v = *vocab.embed(id);
                add_pos(&mut v, pos);
                v
            })
            .collect();

        let target_embs: Vec<[f32; DIM]> = chunk[mid..].iter()
            .enumerate()
            .map(|(pos, &id)| {
                let mut v = *vocab.embed(id);
                add_pos(&mut v, pos);
                v
            })
            .collect();

        compute::equilibrium_propagation(
            field,
            &input_embs,
            &target_embs,
            2,
            10,
            0.3,
        );

        trained += chunk.len();
    }

    trained
}

/// Positional encoding
fn add_pos(v: &mut [f32; DIM], pos: usize) {
    let pos = pos as f32;
    for d in (0..DIM).step_by(2) {
        let freq = 1.0 / (10000.0f32).powf(d as f32 / DIM as f32);
        v[d] += 0.1 * (pos * freq).sin();
        if d + 1 < DIM {
            v[d + 1] += 0.1 * (pos * freq).cos();
        }
    }
}
