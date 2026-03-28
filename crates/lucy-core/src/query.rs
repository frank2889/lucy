//! # query — Vraagverwerking via het Samenveld
//!
//! Dit is de pipeline die een menselijke vraag omzet in een antwoord,
//! volledig berekend door het veld — geen lookup, geen externe AI.
//!
//! ## Query Pipeline
//!
//! ```text
//! Tekst → Tokenize → Embeddings → Injecteer in Input-regio
//!                                       ↓
//!                              Veld settlet (max 300 ticks)
//!                              "Lucy denkt na"
//!                                       ↓
//!                              Lees Output-regio → Decode Token
//!                                       ↓
//!                              Feed token terug → Mini-settle (20 ticks)
//!                                       ↓
//!                              Herhaal autoregressief (max 50 tokens)
//!                                       ↓
//!                              Tokens → Tekst antwoord
//! ```
//!
//! ## Constanten
//!
//! - `TICKS_PER_TOKEN = 10` — hoeveel ticks per geïnjecteerd input-token
//! - `MAX_SETTLE_TICKS = 300` — maximale denktijd (convergentie of timeout)
//! - `SETTLE_EPSILON = 0.005` — convergentiedrempel (spanning < dit = klaar)
//! - `MAX_OUTPUT_TOKENS = 50` — maximale antwoordlengte
//!
//! ## Training
//!
//! `train_pair()` accepteert een input/target tekstpaar en traint het veld
//! via equilibrium propagation. Dit wordt gebruikt door de `/feed` API endpoint.

use lucy_state::*;
use crate::compute;
use crate::encoding::{self, Vocabulary};
use crate::decoding;

/// Hoeveel ticks het veld draait per geïnjecteerd input-token.
/// Hoger = meer tijd om elke token te verwerken, maar langzamer.
const TICKS_PER_TOKEN: usize = 10;

/// Maximaal aantal ticks voor het settle-proces ("nadenken").
/// Na 300 ticks zonder convergentie geeft Lucy haar beste antwoord.
const MAX_SETTLE_TICKS: usize = 300;

/// Convergentiedrempel: als gemiddelde spanning lager is, stopt het settlen.
const SETTLE_EPSILON: f32 = 0.005;

/// Maximaal aantal output-tokens per antwoord.
const MAX_OUTPUT_TOKENS: usize = 50;

/// Verwerk een vraag door het veld
/// Encode → Inject → Settle → Decode (autoregressief)
pub fn process_query(
    field: &mut Field,
    vocab: &Vocabulary,
    question: &str,
) -> QueryResult {
    let embeddings = encoding::encode_text_readonly(vocab, question);
    if embeddings.is_empty() {
        return QueryResult {
            answer: String::new(),
            settle_ticks: 0,
            final_tension: 0.0,
            tokens_generated: 0,
        };
    }

    // Injecteer vraag-tokens in het veld
    field.phase = FieldPhase::Processing;
    compute::inject_token_sequence(field, &embeddings, TICKS_PER_TOKEN);

    // Settle — het veld "denkt na"
    let (settle_ticks, tension) = compute::settle_until_converged(
        field, MAX_SETTLE_TICKS, SETTLE_EPSILON,
    );

    // Decodeer output autoregressief
    let mut output_ids = Vec::new();
    let eos_id = *vocab.token_to_id.get("<eos>").unwrap_or(&2);

    for _ in 0..MAX_OUTPUT_TOKENS {
        // Lees output-regio
        let output_phi = decoding::read_output_region(field);

        // Decodeer naar token
        let (token_id, _score) = decoding::decode_token(&output_phi, vocab);

        if token_id == eos_id || token_id <= 2 {
            break; // Stop bij EOS of speciale tokens
        }

        output_ids.push(token_id);

        // Feed token terug als input voor volgende token
        let emb = vocab.embed(token_id);
        let input_ids = field.input_ids.clone();
        for &id in &input_ids {
            field.clamps.insert(id, emb.to_vec());
        }

        // Mini-settle voor context update
        compute::settle_until_converged(field, 20, SETTLE_EPSILON);

        // Verwijder clamps
        for &id in &input_ids {
            field.clamps.remove(&id);
        }
    }

    field.phase = FieldPhase::Idle;
    field.clamps.clear();

    let answer = decoding::tokens_to_text(&output_ids, vocab);

    QueryResult {
        answer,
        settle_ticks,
        final_tension: tension,
        tokens_generated: output_ids.len(),
    }
}

/// Train het veld met een input-output paar via equilibrium propagation
pub fn train_pair(
    field: &mut Field,
    vocab: &mut Vocabulary,
    input_text: &str,
    target_text: &str,
) -> f32 {
    let input_embs = encoding::encode_text(vocab, input_text);
    let target_embs = encoding::encode_text(vocab, target_text);

    if input_embs.is_empty() || target_embs.is_empty() {
        return 0.0;
    }

    compute::equilibrium_propagation(
        field,
        &input_embs,
        &target_embs,
        TICKS_PER_TOKEN,
        MAX_SETTLE_TICKS / 2,
        0.5, // EP beta
    )
}

pub struct QueryResult {
    pub answer: String,
    pub settle_ticks: usize,
    pub final_tension: f32,
    pub tokens_generated: usize,
}
