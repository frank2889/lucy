//! # decoding — Veld-Output naar Tekst
//!
//! De omgekeerde weg van encoding: vertaalt de 64-dimensionale output van
//! het veld terug naar leesbare tokens/woorden.
//!
//! ## Getied Gewichten (Tied Weights)
//!
//! De decoder heeft GEEN eigen parameters. In plaats daarvan wordt de
//! embedding matrix (uit encoding.rs) hergebruikt als decoder via de
//! transpose: `score(token) = dot(output_phi, embedding[token])`.
//!
//! Dit halveert het totale aantal parameters en dwingt de embeddings om
//! zowel goed te encoderen als te decoderen — een bekende techniek uit
//! taalmodellen (Press & Wolf, 2017).
//!
//! ## Autoregressieve Generatie
//!
//! Lucy genereert tekst token-voor-token:
//! 1. Lees output-regio → gemiddelde Φ van 64 output-punten
//! 2. Dot product met alle embeddings → beste token
//! 3. Feed token terug als input
//! 4. Mini-settle (20 ticks) voor context update
//! 5. Herhaal tot EOS of max tokens
//!
//! ## L2 Normalisatie
//!
//! Output-regio Φ wordt L2-genormaliseerd voor stabiele cosine similarity.
//! Zonder normalisatie zouden punten met hoge magnitude altijd winnen.

use lucy_state::DIM;
use crate::encoding::Vocabulary;

/// Decodeer output-regio phi naar een token ID.
/// Gebruikt getied gewichten: decoder = transpose van embedding matrix.
/// Dit halveert het aantal parameters.
pub fn decode_token(output_phi: &[f32; DIM], vocab: &Vocabulary) -> (u32, f32) {
    let mut best_score = f32::MIN;
    let mut best_id = 0u32;

    for id in 0..vocab.vocab_size() {
        let emb = vocab.embed(id as u32);
        // Dot product = cosine similarity (embeddings zijn genormaliseerd)
        let score: f32 = (0..DIM).map(|d| output_phi[d] * emb[d]).sum();
        if score > best_score {
            best_score = score;
            best_id = id as u32;
        }
    }

    (best_id, best_score)
}

/// Gemiddelde phi van een set output punten
pub fn read_output_region(field: &lucy_state::Field) -> [f32; DIM] {
    let mut avg = [0.0f32; DIM];
    let n = field.output_ids.len();
    if n == 0 { return avg; }

    for &id in &field.output_ids {
        let phi = &field.points[id as usize].phi;
        for d in 0..DIM {
            avg[d] += phi[d];
        }
    }

    let inv_n = 1.0 / n as f32;
    for d in 0..DIM {
        avg[d] *= inv_n;
    }

    // L2 normaliseer voor betere cosine similarity matching
    let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for d in 0..DIM { avg[d] /= norm; }
    }

    avg
}

/// Decodeer meerdere tokens vanuit het veld (autoregressief)
/// Na elke token: feed terug als input, settle opnieuw, lees output
pub fn tokens_to_text(ids: &[u32], vocab: &Vocabulary) -> String {
    let eos_id = *vocab.token_to_id.get("<eos>").unwrap_or(&2);

    ids.iter()
        .take_while(|&&id| id != eos_id)
        .map(|&id| vocab.token_name(id).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}
