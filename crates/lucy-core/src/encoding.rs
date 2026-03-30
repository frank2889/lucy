//! # encoding — Tekst naar Veld-Input (v0.9: BPE + DIM=256)
//!
//! Lucy's taalvermogen begint hier. Dit module vertaalt menselijke tekst
//! naar 256-dimensionale vectoren die het veld kan verwerken.
//!
//! ## Tokenizer
//!
//! BPE (Byte Pair Encoding) tokenizer:
//! - Splitst tekst in subwoord-eenheden die Lucy ZELF leert
//! - 256 byte-level fallback tokens → nooit meer `<unk>`
//! - Leert merge-regels uit gelezen tekst (learn_bpe)
//! - Groeit autonoom: niets hardcoded
//!
//! ## Vocabulaire
//!
//! Groeit incrementeel vanuit wat Lucy leest:
//! - Maximaal 32768 tokens (past in ~32MB bij DIM=256)
//! - LRU eviction: als vol, wordt het minst gebruikte token vervangen
//! - Elke token heeft een geleerde 256-dimensionale embedding
//!
//! ## Embeddings
//!
//! - Geïnitialiseerd met deterministische pseudo-random waarden (gebaseerd op token ID)
//! - L2-genormaliseerd voor stabiele cosine similarity
//! - Worden geleerd via equilibrium propagation (prediction error past gewichten aan)
//!
//! ## Positional Encoding
//!
//! Sinusoïdale positie-signalen (Vaswani et al., 2017)

use lucy_state::DIM;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::HashMap;

/// Maximale vocabulaire grootte. 32K tokens × 256 floats × 4 bytes = 32MB.
const MAX_VOCAB: usize = 32768;

/// Aantal gereserveerde byte-level tokens (0-255 = individuele bytes)
const BYTE_TOKENS: usize = 256;

/// Speciale tokens (na byte tokens)
const SPECIAL_PAD: u32 = 256;
const SPECIAL_EOS: u32 = 257;
const SPECIAL_START: u32 = 258; // eerste vrije slot voor BPE merges

/// Lucy's vocabulaire — BPE tokenizer die zichzelf leert
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    /// Geleerde embeddings — elke token heeft een DIM-vector
    pub embeddings: Vec<Embedding>,
    /// Hoe vaak elk token gezien is (voor LRU eviction)
    pub counts: Vec<u32>,
    /// BPE merge-regels: (token_a, token_b) → merged_token
    /// Lucy leert deze zelf uit tekst die ze leest
    #[serde(default)]
    pub merges: Vec<(String, String)>,
}

/// Wrapper voor [f32; DIM] met serde support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embedding {
    #[serde(with = "BigArray")]
    pub v: [f32; DIM],
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            embeddings: Vec::new(),
            counts: Vec::new(),
            merges: Vec::new(),
        };

        // Byte-level tokens (0-255): elk mogelijk byte als fallback
        for b in 0..BYTE_TOKENS {
            let token = format!("<{:02x}>", b);
            vocab.id_to_token.push(token.clone());
            vocab.token_to_id.insert(token, b as u32);
            vocab.embeddings.push(Embedding { v: random_embedding(b as u32) });
            vocab.counts.push(0);
        }

        // Speciale tokens
        vocab.push_token("<pad>");   // 256
        vocab.push_token("<eos>");   // 257

        vocab
    }

    /// Interne push — voegt token toe zonder eviction check
    fn push_token(&mut self, token: &str) -> u32 {
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        self.embeddings.push(Embedding { v: random_embedding(id) });
        self.counts.push(0);
        id
    }

    /// Voeg een token toe met random embedding (met eviction als vol)
    fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }

        // Als vol, verwijder het minst gebruikte token (niet speciale tokens)
        if self.id_to_token.len() >= MAX_VOCAB {
            self.evict_least_used();
        }

        self.push_token(token)
    }

    /// Verwijder het minst gebruikte token
    fn evict_least_used(&mut self) {
        // Skip byte tokens (0-255) en speciale tokens (256-257)
        let skip = SPECIAL_START as usize;
        if self.counts.len() <= skip { return; }

        let min_idx = self.counts.iter().enumerate()
            .skip(skip)
            .min_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(skip);

        let token = self.id_to_token[min_idx].clone();
        self.token_to_id.remove(&token);
        self.id_to_token[min_idx] = "<evicted>".to_string();
        self.counts[min_idx] = 0;
    }

    /// BPE tokenize — splits tekst in subwoord-eenheden
    pub fn tokenize(&mut self, text: &str) -> Vec<u32> {
        let lower = text.to_lowercase();
        let words = split_into_words(&lower);
        let mut all_ids = Vec::new();

        for word in words {
            let mut tokens = self.word_to_initial_tokens(word);

            // Pas BPE merges toe (greedy, van eerste naar laatste merge)
            for (a, b) in &self.merges {
                let merged = format!("{}{}", a, b);
                let mut i = 0;
                while i + 1 < tokens.len() {
                    if tokens[i] == *a && tokens[i + 1] == *b {
                        tokens[i] = merged.clone();
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            // Converteer tokens naar IDs
            for tok in &tokens {
                let id = self.add_token(tok);
                self.counts[id as usize] = self.counts[id as usize].saturating_add(1);
                all_ids.push(id);
            }
        }

        all_ids
    }

    /// Tokenize zonder nieuwe tokens toe te voegen (voor inferentie)
    pub fn tokenize_readonly(&self, text: &str) -> Vec<u32> {
        let lower = text.to_lowercase();
        let words = split_into_words(&lower);
        let mut all_ids = Vec::new();

        for word in words {
            let mut tokens = self.word_to_initial_tokens_readonly(word);

            // Pas BPE merges toe
            for (a, b) in &self.merges {
                let merged = format!("{}{}", a, b);
                let mut i = 0;
                while i + 1 < tokens.len() {
                    if tokens[i] == *a && tokens[i + 1] == *b {
                        tokens[i] = merged.clone();
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            for tok in &tokens {
                let id = self.token_to_id.get(tok.as_str()).copied().unwrap_or(0);
                all_ids.push(id);
            }
        }

        all_ids
    }

    /// Splits woord in initiële tokens (karakters of byte fallbacks)
    fn word_to_initial_tokens(&mut self, word: &str) -> Vec<String> {
        word.chars().map(|c| {
            let s = c.to_string();
            // Probeer als karakter-token
            if self.token_to_id.contains_key(&s) || s.len() == 1 {
                let _ = self.add_token(&s);
                s
            } else {
                // Fallback: byte-level encoding
                s.bytes().map(|b| format!("<{:02x}>", b)).collect::<Vec<_>>().join("")
            }
        }).collect()
    }

    /// Readonly versie van word_to_initial_tokens
    fn word_to_initial_tokens_readonly(&self, word: &str) -> Vec<String> {
        word.chars().map(|c| c.to_string()).collect()
    }

    /// Lucy leert BPE merges uit een corpus tekst
    /// Ze beslist zelf welke paren het meest voorkomen
    pub fn learn_bpe(&mut self, corpus: &str, num_merges: usize) {
        let lower = corpus.to_lowercase();
        let words = split_into_words(&lower);

        // Splits elk woord in karakter-tokens
        let mut word_tokens: Vec<Vec<String>> = words
            .iter()
            .map(|w| w.chars().map(|c| c.to_string()).collect())
            .collect();

        for _ in 0..num_merges {
            if self.id_to_token.len() >= MAX_VOCAB { break; }

            // Tel alle aangrenzende paren
            let mut pair_counts: HashMap<(String, String), u32> = HashMap::new();
            for tokens in &word_tokens {
                for i in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_counts.entry(pair).or_default() += 1;
                }
            }

            // Vind meest-frequente paar
            let best = pair_counts.into_iter()
                .max_by_key(|(_, count)| *count);

            let (pair, count) = match best {
                Some((p, c)) if c >= 2 => (p, c),
                _ => break, // geen paren meer met count >= 2
            };

            let merged = format!("{}{}", pair.0, pair.1);

            // Voeg merged token toe aan vocab
            self.add_token(&merged);
            self.merges.push(pair.clone());

            // Pas merge toe op alle woorden
            for tokens in &mut word_tokens {
                let mut i = 0;
                while i + 1 < tokens.len() {
                    if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                        tokens[i] = merged.clone();
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            if count < 3 { break; } // niet genoeg data voor meer merges
        }

        tracing::info!(
            merges = self.merges.len(),
            vocab = self.id_to_token.len(),
            "Lucy leerde BPE merges"
        );
    }

    /// Haal embedding op voor een token ID
    pub fn embed(&self, id: u32) -> &[f32; DIM] {
        if (id as usize) < self.embeddings.len() {
            &self.embeddings[id as usize].v
        } else {
            &self.embeddings[0].v // byte 0x00 als fallback
        }
    }

    /// Mutable embedding voor learning
    #[allow(dead_code)]
    pub fn embed_mut(&mut self, id: u32) -> &mut [f32; DIM] {
        let idx = if (id as usize) < self.embeddings.len() { id as usize } else { 0 };
        &mut self.embeddings[idx].v
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn token_name(&self, id: u32) -> &str {
        if (id as usize) < self.id_to_token.len() {
            &self.id_to_token[id as usize]
        } else {
            "<00>"
        }
    }

    /// Atomic save: schrijf naar .tmp, dan rename
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let tmp = path.with_extension("tmp");
        let bytes = bincode::serialize(self)?;
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        tracing::info!(vocab_size = self.id_to_token.len(), merges = self.merges.len(), "Vocabulary saved (atomic)");
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let vocab: Self = bincode::deserialize(&bytes)?;
        tracing::info!(vocab_size = vocab.id_to_token.len(), merges = vocab.merges.len(), "Vocabulary loaded");
        Ok(vocab)
    }
}

/// Encodeer tekst als sequentie van DIM-vectoren met positional encoding
pub fn encode_text(vocab: &mut Vocabulary, text: &str) -> Vec<[f32; DIM]> {
    let ids = vocab.tokenize(text);
    ids.iter().enumerate().map(|(pos, &id)| {
        let mut v = *vocab.embed(id);
        add_positional_encoding(&mut v, pos);
        v
    }).collect()
}

/// Encodeer zonder vocab te muteren (voor inferentie)
pub fn encode_text_readonly(vocab: &Vocabulary, text: &str) -> Vec<[f32; DIM]> {
    let ids = vocab.tokenize_readonly(text);
    ids.iter().enumerate().map(|(pos, &id)| {
        let mut v = *vocab.embed(id);
        add_positional_encoding(&mut v, pos);
        v
    }).collect()
}

/// Precomputed frequency tabel — eenmalig berekend, hergebruikt elke call
static POSITIONAL_FREQS: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(|| {
    let mut freqs = vec![0.0f32; DIM / 2];
    for i in 0..DIM / 2 {
        let d = (i * 2) as f32;
        freqs[i] = 1.0 / (10000.0f32).powf(d / DIM as f32);
    }
    freqs
});

/// Sinusoïdale positional encoding (met precomputed frequencies)
fn add_positional_encoding(v: &mut [f32; DIM], pos: usize) {
    let pos = pos as f32;
    let freqs = &*POSITIONAL_FREQS;
    for i in 0..DIM / 2 {
        let d = i * 2;
        let angle = pos * freqs[i];
        v[d] += 0.1 * angle.sin();
        if d + 1 < DIM {
            v[d + 1] += 0.1 * angle.cos();
        }
    }
}

/// Deterministische pseudo-random embedding op basis van token ID
fn random_embedding(id: u32) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    let mut hash = id as u64 ^ 0x517cc1b727220a95;
    for d in 0..DIM {
        hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v[d] = ((hash >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
    }
    // L2 normaliseer
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for d in 0..DIM { v[d] /= norm; }
    }
    v
}

/// Split tekst in woorden (op whitespace en interpunctie)
fn split_into_words(text: &str) -> Vec<&str> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?')
        .filter(|w| !w.is_empty())
        .collect()
}

/// Converteer token IDs terug naar tekst
pub fn tokens_to_text(vocab: &Vocabulary, ids: &[u32]) -> String {
    ids.iter()
        .map(|&id| vocab.token_name(id))
        .collect::<Vec<_>>()
        .join(" ")
}
