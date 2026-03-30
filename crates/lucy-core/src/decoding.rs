//! # decoding — Veld-Output naar Tekst (v0.9+)
//!
//! De omgekeerde weg van encoding: vertaalt de DIM-dimensionale output van
//! het veld terug naar leesbare tokens/woorden.
//!
//! ## v0.9+ Verbeteringen
//!
//! - **Temperature sampling**: controleert diversiteit van output (0.1=veilig, 1.0=creatief)
//! - **Top-p (nucleus) sampling**: alleen tokens met cumulatieve prob > p overwegen
//! - **Repetition penalty**: straft herhaalde tokens af → voorkomt loops
//! - **Grotere candidate pool**: 256 i.p.v. 64 → betere token selectie
//! - **Hogere COARSE_DIMS**: 128 i.p.v. 64 → nauwkeuriger pre-filtering

use lucy_state::DIM;
use crate::encoding::Vocabulary;
use rand::Rng;
use std::collections::HashSet;

/// Eerste 128 dimensies voor snelle coarse filtering (50% van DIM=256)
const COARSE_DIMS: usize = 128;

/// Aantal candidates voor fine pass
const TOP_K_CANDIDATES: usize = 256;

/// Temperature: controleert diversiteit. Lager = veiliger, hoger = creatiever.
const TEMPERATURE: f32 = 0.7;

/// Nucleus sampling drempel: alleen tokens in de top-p van de kansverdeling
const TOP_P: f32 = 0.9;

/// Repetition penalty: factor waarmee herhaalde tokens worden afgestraft
const REPETITION_PENALTY: f32 = 1.3;

/// Decodeer output-regio phi naar een token ID.
/// v0.9+: Temperature sampling + nucleus filtering + repetition penalty.
pub fn decode_token(output_phi: &[f32; DIM], vocab: &Vocabulary) -> (u32, f32) {
    decode_token_with_context(output_phi, vocab, &[])
}

/// Decodeer met context van eerder gegenereerde tokens (voor repetition penalty)
pub fn decode_token_with_context(
    output_phi: &[f32; DIM],
    vocab: &Vocabulary,
    previous_tokens: &[u32],
) -> (u32, f32) {
    let vocab_size = vocab.vocab_size();
    if vocab_size == 0 { return (0, 0.0); }

    // Pass 1: coarse score op eerste COARSE_DIMS dimensies
    let mut scores: Vec<(u32, f32)> = (0..vocab_size as u32)
        .map(|id| {
            let emb = vocab.embed(id);
            let coarse: f32 = (0..COARSE_DIMS.min(DIM)).map(|d| output_phi[d] * emb[d]).sum();
            (id, coarse)
        })
        .collect();

    // Sorteer op coarse score, pak top-K candidates
    let k = TOP_K_CANDIDATES.min(scores.len());
    scores.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Pass 2: full dot product op top-K candidates
    let mut candidates: Vec<(u32, f32)> = scores[..k].iter()
        .map(|&(id, _)| {
            let emb = vocab.embed(id);
            let score: f32 = (0..DIM).map(|d| output_phi[d] * emb[d]).sum();
            (id, score)
        })
        .collect();

    // Repetition penalty: straf eerder gegenereerde tokens af
    // v0.9+: HashSet voor O(1) lookup (was: O(n) contains per candidate)
    if !previous_tokens.is_empty() {
        let prev_set: HashSet<u32> = previous_tokens.iter().copied().collect();
        for (id, score) in &mut candidates {
            if prev_set.contains(id) {
                if *score > 0.0 {
                    *score /= REPETITION_PENALTY;
                } else {
                    *score *= REPETITION_PENALTY;
                }
            }
        }
    }

    // v0.9+: In-place temperature scaling + softmax (was: 3 aparte Vec allocaties)
    let temp = TEMPERATURE.max(0.01);
    let max_score = candidates.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);

    // Transformeer candidates in-place naar probabilities
    let mut total = 0.0f32;
    for (_, s) in &mut candidates {
        *s = ((*s - max_score) / temp).exp();
        total += *s;
    }
    if total > 1e-8 {
        for (_, p) in &mut candidates { *p /= total; }
    }

    // Sorteer op probability (hoog → laag) voor nucleus sampling
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-p (nucleus) filtering — hergebruik candidates slice, geen nieuwe Vec
    let mut cumulative = 0.0f32;
    let mut nucleus_len = candidates.len();
    for (i, &(_, prob)) in candidates.iter().enumerate() {
        cumulative += prob;
        if cumulative >= TOP_P {
            nucleus_len = i + 1;
            break;
        }
    }
    let nucleus = &mut candidates[..nucleus_len];

    // Hernormaliseer nucleus + forceer cumsum=1.0 (voorkomt float drift → NaN)
    let nucleus_total: f32 = nucleus.iter().map(|(_, p)| *p).sum();
    if nucleus_total > 1e-8 {
        for entry in nucleus.iter_mut() { entry.1 /= nucleus_total; }
        // Forceer laatste element zodat cumsum exact 1.0 bereikt
        if nucleus.len() > 1 {
            let rest: f32 = nucleus[..nucleus.len() - 1].iter().map(|(_, p)| p).sum();
            nucleus.last_mut().unwrap().1 = (1.0 - rest).max(0.0);
        }
    }

    // Sample uit de nucleus verdeling
    let mut rng = rand::rng();
    let threshold: f32 = rng.random_range(0.0..1.0f32);
    let mut cumsum = 0.0f32;
    for &(id, prob) in nucleus.iter() {
        cumsum += prob;
        if cumsum >= threshold {
            return (id, prob);
        }
    }

    // Fallback
    nucleus.last().map(|&(id, p)| (id, p)).unwrap_or((0, 0.0))
}

/// Gemiddelde phi van output punten — de-roteert elk punt (inverse van input-differentiatie)
/// Output-punt `i` werd gevoed met embedding geroteerd met `i`, dus we roteren terug
pub fn read_output_region(field: &lucy_state::Field) -> [f32; DIM] {
    let mut avg = [0.0f32; DIM];
    let n = field.output_ids.len();
    if n == 0 { return avg; }

    for (i, &id) in field.output_ids.iter().enumerate() {
        let phi = &field.points[id as usize].phi;
        // De-roteer: dimensie d van punt i → dimensie (d - i) mod DIM
        for d in 0..DIM {
            let original_d = (d + DIM - i) % DIM;
            avg[original_d] += phi[d];
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
