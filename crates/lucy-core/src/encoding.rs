//! # encoding — Tekst naar Veld-Input
//!
//! Lucy's taalvermogen begint hier. Dit module vertaalt menselijke tekst
//! naar 64-dimensionale vectoren die het veld kan verwerken.
//!
//! ## Tokenizer
//!
//! Eenvoudige woord-level tokenizer:
//! - Splitst op niet-alfanumerieke karakters (behalve apostrof)
//! - Lowercase voor case-insensitiviteit
//! - Drie speciale tokens: `<unk>` (0), `<pad>` (1), `<eos>` (2)
//!
//! ## Vocabulaire
//!
//! Groeit incrementeel vanuit wat Lucy leest:
//! - Maximaal 8192 tokens (geheugen-efficiënt voor $6/maand server)
//! - LRU eviction: als vol, wordt het minst gebruikte token vervangen
//! - Elke token heeft een geleerde 64-dimensionale embedding
//!
//! ## Embeddings
//!
//! - Geïnitialiseerd met deterministische pseudo-random waarden (gebaseerd op token ID)
//! - L2-genormaliseerd voor stabiele cosine similarity
//! - Worden geleerd via equilibrium propagation (prediction error past gewichten aan)
//!
//! ## Positional Encoding
//!
//! Sinusoïdale positie-signalen (Vaswani et al., 2017):
//! - `v[d] += 0.1 * sin(pos * freq)` voor even dimensies
//! - `v[d] += 0.1 * cos(pos * freq)` voor oneven dimensies
//! - freq = 1/10000^(d/DIM) — lage dimensies = hoge frequentie (lokale positie),
//!   hoge dimensies = lage frequentie (globale positie)
//! - Schaalfactor 0.1 voorkomt dat positie het semantische signaal overschrijft

use lucy_state::DIM;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::HashMap;

/// Maximale vocabulaire grootte. Bij 8192 tokens × 64 floats = 2MB geheugen.
/// LRU eviction houdt dit beheersbaar op een low-resource server.
const MAX_VOCAB: usize = 8192;

/// Lucy's vocabulaire — groeit incrementeel vanuit wat ze leest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    /// Geleerde embeddings — elke token heeft een DIM-vector
    pub embeddings: Vec<Embedding>,
    /// Hoe vaak elk token gezien is (voor LRU eviction)
    pub counts: Vec<u32>,
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
        };
        // Speciale tokens
        vocab.add_token("<unk>");  // 0
        vocab.add_token("<pad>");  // 1
        vocab.add_token("<eos>");  // 2
        vocab
    }

    /// Voeg een token toe met random embedding
    fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }

        // Als vol, verwijder het minst gebruikte token (niet speciale tokens)
        if self.id_to_token.len() >= MAX_VOCAB {
            self.evict_least_used();
        }

        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        self.embeddings.push(Embedding { v: random_embedding(id) });
        self.counts.push(0);
        id
    }

    /// Verwijder het minst gebruikte token
    fn evict_least_used(&mut self) {
        // Vind het minst gebruikte token (skip speciale tokens 0,1,2)
        if self.counts.len() <= 3 { return; }

        let min_idx = self.counts.iter().enumerate()
            .skip(3)
            .min_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(3);

        let token = self.id_to_token[min_idx].clone();
        self.token_to_id.remove(&token);

        // Swap-remove is O(1) maar verandert IDs — gebruik gewoon count reset
        // We markeren de slot als herbruikbaar door de token te vervangen
        self.id_to_token[min_idx] = "<evicted>".to_string();
        self.counts[min_idx] = 0;
    }

    /// Tokenize tekst en registreer nieuwe tokens
    pub fn tokenize(&mut self, text: &str) -> Vec<u32> {
        let lower = text.to_lowercase();
        let mut ids = Vec::new();

        for word in tokenize_words(&lower) {
            let id = self.add_token(word);
            self.counts[id as usize] = self.counts[id as usize].saturating_add(1);
            ids.push(id);
        }

        ids
    }

    /// Tokenize zonder nieuwe tokens toe te voegen (voor inferentie)
    pub fn tokenize_readonly(&self, text: &str) -> Vec<u32> {
        let lower = text.to_lowercase();
        tokenize_words(&lower)
            .map(|word| {
                *self.token_to_id.get(word).unwrap_or(&0) // 0 = <unk>
            })
            .collect()
    }

    /// Haal embedding op voor een token ID
    pub fn embed(&self, id: u32) -> &[f32; DIM] {
        if (id as usize) < self.embeddings.len() {
            &self.embeddings[id as usize].v
        } else {
            &self.embeddings[0].v // <unk>
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
            "<unk>"
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        tracing::info!(vocab_size = self.id_to_token.len(), "Vocabulary saved");
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let vocab: Self = bincode::deserialize(&bytes)?;
        tracing::info!(vocab_size = vocab.id_to_token.len(), "Vocabulary loaded");
        Ok(vocab)
    }
}

/// Encodeer tekst als sequentie van DIM-vectoren met positional encoding
pub fn encode_text(vocab: &mut Vocabulary, text: &str) -> Vec<[f32; DIM]> {
    let ids = vocab.tokenize(text);
    ids.iter().enumerate().map(|(pos, &id)| {
        let mut v = *vocab.embed(id);
        // Voeg positional encoding toe (sinusoïdaal)
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

/// Sinusoïdale positional encoding
fn add_positional_encoding(v: &mut [f32; DIM], pos: usize) {
    let pos = pos as f32;
    for d in (0..DIM).step_by(2) {
        let freq = 1.0 / (10000.0f32).powf(d as f32 / DIM as f32);
        v[d] += 0.1 * (pos * freq).sin();
        if d + 1 < DIM {
            v[d + 1] += 0.1 * (pos * freq).cos();
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

/// Split tekst in woorden (eenvoudige tokenizer)
fn tokenize_words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
}
