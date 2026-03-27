use lucy_state::DIM;

/// Converteer tekst naar een 16-dimensionale fingerprint
/// Gebruikt character trigram hashing — snel, deterministisch, geen dependencies
pub fn text_to_fingerprint(text: &str) -> [f32; DIM] {
    let mut fp = [0.0f32; DIM];
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().filter(|c| c.is_alphanumeric() || c.is_whitespace()).collect();

    if chars.len() < 3 {
        // Te kort voor trigrams, gebruik individuele chars
        for (i, c) in chars.iter().enumerate() {
            let dim = (*c as usize) % DIM;
            fp[dim] += 1.0 / (i as f32 + 1.0);
        }
    } else {
        // Trigram hashing
        for window in chars.windows(3) {
            let hash = trigram_hash(window[0], window[1], window[2]);
            let dim = (hash as usize) % DIM;
            let value = ((hash >> 16) as f32 / 65535.0) * 2.0 - 1.0;
            fp[dim] += value;
        }
    }

    // L2 normalisatie
    let norm = fp.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for d in 0..DIM {
            fp[d] /= norm;
        }
    }

    fp
}

/// Snelle hash voor character trigram
fn trigram_hash(a: char, b: char, c: char) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    h = h.wrapping_mul(0x100000001b3);
    h ^= a as u64;
    h = h.wrapping_mul(0x100000001b3);
    h ^= b as u64;
    h = h.wrapping_mul(0x100000001b3);
    h ^= c as u64;
    h
}

/// Cosine similarity tussen twee fingerprints (ze zijn al genormaliseerd → dot product)
pub fn similarity(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut dot = 0.0f32;
    for d in 0..DIM {
        dot += a[d] * b[d];
    }
    dot
}

/// Simpele woordoverlap als secundaire check (voorkomt fingerprint collisions)
pub fn word_overlap(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    let intersection = words_a.intersection(&words_b).count() as f32;
    let union = words_a.union(&words_b).count() as f32;

    if union > 0.0 { intersection / union } else { 0.0 }
}
