//! # retrieval — Lucy's Episodisch Geheugen
//!
//! Menselijk declaratief geheugen: Lucy slaat feiten op als (key, value) paren
//! en kan ze later opzoeken via cosine similarity.
//!
//! Dit is NIET het gewicht-geheugen (dat zit in couplings).
//! Dit is expliciet, doorzoekbaar geheugen — zoals een mens die zich
//! een feit herinnert dat ze ooit heeft gelezen.
//!
//! ## Design
//!
//! - Keys: DIM-dimensionale embeddings (gemiddelde van token embeddings)
//! - Values: ruwe tekst (de oorspronkelijke zin/paragraaf)
//! - Lookup: cosine similarity met bucket-prefiltering
//! - Max 10K entries (~12MB)

use lucy_state::DIM;
use serde::{Deserialize, Serialize};

const MAX_ENTRIES: usize = 100_000; // v0.9+: 10× meer geheugen (~120MB)
const N_BUCKETS: usize = 256; // v0.9+: meer buckets voor snellere lookup

/// Lucy's episodisch geheugen — feiten die ze heeft gelezen
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalStore {
    /// Embedding keys (gemiddelde van token embeddings per entry)
    keys: Vec<Vec<f32>>, // Vec<f32> i.p.v. [f32; DIM] voor serde compat
    /// Ruwe tekst values
    values: Vec<String>,
    /// Bucket index: hash van eerste dims → indices in keys/values
    #[serde(skip)]
    buckets: Vec<Vec<usize>>,
}

impl RetrievalStore {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            buckets: vec![Vec::new(); N_BUCKETS],
        }
    }

    /// Sla een feit op: embedding + tekst
    pub fn insert(&mut self, embedding: &[f32; DIM], text: &str) {
        // Skip te korte teksten
        if text.len() < 10 { return; }

        // FIFO: als vol, verwijder oudste
        if self.keys.len() >= MAX_ENTRIES {
            self.keys.remove(0);
            self.values.remove(0);
            // Rebuild buckets (goedkoop bij FIFO)
            self.rebuild_buckets();
        }

        let idx = self.keys.len();
        self.keys.push(embedding.to_vec());
        self.values.push(text.to_string());

        // Voeg toe aan bucket
        let bucket = self.hash_to_bucket(embedding);
        if bucket < self.buckets.len() {
            self.buckets[bucket].push(idx);
        }
    }

    /// Zoek top-K meest relevante feiten voor een query
    pub fn retrieve(&self, query: &[f32; DIM], top_k: usize) -> Vec<(f32, &str)> {
        if self.keys.is_empty() { return Vec::new(); }

        // Bucket prefiltering: zoek alleen in nabije buckets
        let query_bucket = self.hash_to_bucket(query);
        let search_buckets = [
            query_bucket,
            (query_bucket + 1) % N_BUCKETS,
            (query_bucket + N_BUCKETS - 1) % N_BUCKETS,
            (query_bucket + 2) % N_BUCKETS,
            (query_bucket + N_BUCKETS - 2) % N_BUCKETS,
        ];

        let mut candidates: Vec<usize> = Vec::new();
        for &b in &search_buckets {
            if b < self.buckets.len() {
                candidates.extend_from_slice(&self.buckets[b]);
            }
        }

        // Als weinig candidates, zoek breder
        if candidates.len() < top_k * 2 {
            candidates = (0..self.keys.len()).collect();
        }

        // Score candidates via cosine similarity
        let mut scored: Vec<(f32, usize)> = candidates.iter()
            .filter(|&&idx| idx < self.keys.len())
            .map(|&idx| {
                let sim = cosine_sim(query, &self.keys[idx]);
                (sim, idx)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored.into_iter()
            .map(|(sim, idx)| (sim, self.values[idx].as_str()))
            .collect()
    }

    /// Aantal opgeslagen feiten
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    fn hash_to_bucket(&self, embedding: &[f32]) -> usize {
        // Simpele hash: som van eerste 8 dimensies → bucket
        let sum: f32 = embedding.iter().take(8).sum();
        ((sum.abs() * 100.0) as usize) % N_BUCKETS
    }

    fn rebuild_buckets(&mut self) {
        self.buckets = vec![Vec::new(); N_BUCKETS];
        for (i, key) in self.keys.iter().enumerate() {
            let bucket = self.hash_to_bucket(key);
            if bucket < self.buckets.len() {
                self.buckets[bucket].push(i);
            }
        }
    }

    /// Atomic save: schrijf naar .tmp, dan rename
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let tmp = path.with_extension("tmp");
        let bytes = bincode::serialize(self)?;
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        tracing::info!(entries = self.keys.len(), "Retrieval store saved (atomic)");
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let mut store: Self = bincode::deserialize(&bytes)?;
        store.rebuild_buckets();
        tracing::info!(entries = store.keys.len(), "Retrieval store loaded");
        Ok(store)
    }
}

/// Cosine similarity tussen query en stored key
fn cosine_sim(a: &[f32; DIM], b: &[f32]) -> f32 {
    let n = DIM.min(b.len());
    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;

    for i in 0..n {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    let denom = (mag_a.sqrt() * mag_b.sqrt()).max(1e-8);
    dot / denom
}
