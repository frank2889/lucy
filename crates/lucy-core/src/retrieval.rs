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

// v2.6.7 E4: cap verlaagd van 100K → 15K. Intelligentie zit in weights, niet in entries.
const MAX_ENTRIES: usize = 15_000;
const N_BUCKETS: usize = 256; // v0.9+: meer buckets voor snellere lookup
/// v2.6.7 E4: value-truncate op inslag — reduceert ruis, bespaart geheugen.
const MAX_VALUE_LEN: usize = 500;

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
    /// v2.6.7 E4: hoe vaak deze entry is opgehaald. Hoog = waardevol.
    #[serde(default)]
    hits: Vec<u32>,
    /// v2.6.7 E4: tick van laatste access. Voor recency in eviction-score.
    #[serde(default)]
    last_access_tick: Vec<u64>,
    /// v2.6.7 E4: reward-signaal bij inslag (satisfaction_delta). Goed → blijft.
    #[serde(default)]
    learning_value: Vec<f32>,
    /// v2.6.8 P1.1: bron-classificatie (primary truth voor provenance, geen heuristiek).
    /// 0 = external (webpagina, unknown legacy), 1 = self (reflect snapshot),
    /// 2 = teacher (AI-chat antwoord), 3 = interaction (user /ask + Lucy response).
    /// Oude snapshot-entries krijgen default 0 (external) → conservatief.
    #[serde(default)]
    kind: Vec<u8>,
}

/// v2.6.8 P1.1: retrieval entry bron-classificatie
pub const KIND_EXTERNAL: u8 = 0;
pub const KIND_SELF: u8 = 1;
pub const KIND_TEACHER: u8 = 2;
pub const KIND_INTERACTION: u8 = 3;

impl RetrievalStore {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            buckets: vec![Vec::new(); N_BUCKETS],
            hits: Vec::new(),
            last_access_tick: Vec::new(),
            learning_value: Vec::new(),
            kind: Vec::new(),
        }
    }

    /// v2.6.7 E4 + v2.6.8 P1.1: repair parallel-vectoren bij load.
    /// Oude snapshots hebben `kind` niet → default 0 (external, conservatief).
    pub fn ensure_parallel_vectors(&mut self, current_tick: u64) {
        let n = self.keys.len();
        if self.hits.len() != n { self.hits = vec![1; n]; }
        if self.last_access_tick.len() != n { self.last_access_tick = vec![current_tick; n]; }
        if self.learning_value.len() != n { self.learning_value = vec![0.0; n]; }
        if self.kind.len() != n { self.kind = vec![KIND_EXTERNAL; n]; }
    }

    /// v2.6.8 P1.1: per-kind distributie (voor retrieval metrics)
    pub fn kind_distribution(&self) -> [u32; 4] {
        let mut counts = [0u32; 4];
        for &k in &self.kind {
            if (k as usize) < 4 {
                counts[k as usize] = counts[k as usize].saturating_add(1);
            }
        }
        counts
    }

    /// v2.6.8 P1.3: Geef alle self-entries (kind=1) met key + hits + last_access.
    /// Gebruikt voor self_cluster_density / drift / centroid_strength berekening.
    pub fn self_entries(&self) -> Vec<(&Vec<f32>, u32, u64)> {
        let mut out = Vec::new();
        for i in 0..self.keys.len() {
            if self.kind.get(i).copied() == Some(KIND_SELF) {
                out.push((
                    &self.keys[i],
                    self.hits.get(i).copied().unwrap_or(1),
                    self.last_access_tick.get(i).copied().unwrap_or(0),
                ));
            }
        }
        out
    }

    /// v2.6.8 P1.3: Gemiddelde onderlinge cosine van recente self-entries (max 50).
    /// Meet of er een kern ontstaat (hoog = convergerend zelfmodel).
    pub fn self_cluster_density(&self, max_recent: usize) -> f32 {
        let mut entries: Vec<(usize, u64)> = self.kind.iter().enumerate()
            .filter(|(_, k)| **k == KIND_SELF)
            .map(|(i, _)| (i, self.last_access_tick.get(i).copied().unwrap_or(0)))
            .collect();
        entries.sort_by_key(|(_, tick)| std::cmp::Reverse(*tick));
        entries.truncate(max_recent);
        let n = entries.len();
        if n < 2 { return 0.0; }
        let mut sum = 0.0f32;
        let mut pairs = 0u32;
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.keys[entries[i].0];
                let b = &self.keys[entries[j].0];
                sum += cosine_sim_vec(a, b);
                pairs += 1;
            }
        }
        if pairs > 0 { sum / pairs as f32 } else { 0.0 }
    }

    /// v2.6.8 P1.3: Gemiddelde cosine van recente self-entries naar centroid van alle self-entries.
    /// Meet "hoe sterk is de zelf-attractor" (kern vs ruis).
    pub fn self_centroid_and_strength(&self, max_recent: usize) -> (Vec<f32>, f32) {
        let mut entries: Vec<(usize, u64)> = self.kind.iter().enumerate()
            .filter(|(_, k)| **k == KIND_SELF)
            .map(|(i, _)| (i, self.last_access_tick.get(i).copied().unwrap_or(0)))
            .collect();
        entries.sort_by_key(|(_, tick)| std::cmp::Reverse(*tick));
        entries.truncate(max_recent);
        if entries.is_empty() { return (vec![0.0; DIM], 0.0); }

        // Bereken centroid
        let mut centroid = vec![0.0f32; DIM];
        for (idx, _) in &entries {
            let k = &self.keys[*idx];
            for d in 0..DIM.min(k.len()) {
                centroid[d] += k[d];
            }
        }
        let n = entries.len() as f32;
        for d in 0..DIM { centroid[d] /= n; }

        // Gemiddelde cosine naar centroid
        let mut sum_sim = 0.0f32;
        for (idx, _) in &entries {
            sum_sim += cosine_sim_vec(&self.keys[*idx], &centroid);
        }
        let strength = sum_sim / n;
        (centroid, strength)
    }

    /// v2.6.8 P1.4: aantal teacher-entries met hits > 1 (reuse proxy voor effect-eval)
    pub fn count_teacher_reused(&self) -> u32 {
        let mut n = 0u32;
        for i in 0..self.keys.len() {
            if self.kind.get(i).copied() == Some(KIND_TEACHER)
                && self.hits.get(i).copied().unwrap_or(1) > 1
            {
                n = n.saturating_add(1);
            }
        }
        n
    }

    /// v2.6.8 P1.3: fractie self-entries die hits > 1 hebben (reuse-rate proxy).
    pub fn self_reuse_rate(&self) -> f32 {
        let mut total = 0u32;
        let mut reused = 0u32;
        for i in 0..self.keys.len() {
            if self.kind.get(i).copied() == Some(KIND_SELF) {
                total += 1;
                if self.hits.get(i).copied().unwrap_or(1) > 1 {
                    reused += 1;
                }
            }
        }
        if total > 0 { reused as f32 / total as f32 } else { 0.0 }
    }

    /// Sla een feit op: embedding + tekst (met optioneel learning_value)
    pub fn insert(&mut self, embedding: &[f32; DIM], text: &str) {
        self.insert_full(embedding, text, 0.0, 0, KIND_EXTERNAL);
    }

    /// v2.6.7 E4: insert met learning_value + tick voor score-based eviction
    pub fn insert_with_value(&mut self, embedding: &[f32; DIM], text: &str, learning_value: f32, current_tick: u64) {
        self.insert_full(embedding, text, learning_value, current_tick, KIND_EXTERNAL);
    }

    /// v2.6.8 P1.1: insert met expliciete bron-classificatie (primary truth voor provenance)
    pub fn insert_full(&mut self, embedding: &[f32; DIM], text: &str, learning_value: f32, current_tick: u64, kind: u8) {
        // Skip te korte teksten
        if text.len() < 10 { return; }

        // Zorg dat parallel-vectoren synchroon zijn
        self.ensure_parallel_vectors(current_tick);

        // v2.6.7 E4: score-based eviction (vervangt FIFO). Laagste score verdwijnt.
        // score = hits / (tick - last_access + 1) + learning_value
        if self.keys.len() >= MAX_ENTRIES {
            let mut worst_idx = 0usize;
            let mut worst_score = f32::MAX;
            for i in 0..self.keys.len() {
                let recency = (current_tick.saturating_sub(self.last_access_tick[i]) + 1) as f32;
                let score = (self.hits[i] as f32) / recency + self.learning_value[i];
                if score < worst_score {
                    worst_score = score;
                    worst_idx = i;
                }
            }
            self.keys.remove(worst_idx);
            self.values.remove(worst_idx);
            self.hits.remove(worst_idx);
            self.last_access_tick.remove(worst_idx);
            self.learning_value.remove(worst_idx);
            if worst_idx < self.kind.len() { self.kind.remove(worst_idx); }
            self.rebuild_buckets();
        }

        // Truncate value naar MAX_VALUE_LEN chars (reduceert ruis)
        let stored_text = if text.len() > MAX_VALUE_LEN {
            // Neem eerste 500 chars maar kap op woord-grens
            let cutoff = text.char_indices().take(MAX_VALUE_LEN).last().map(|(i, _)| i).unwrap_or(MAX_VALUE_LEN);
            text[..cutoff].to_string()
        } else {
            text.to_string()
        };

        let idx = self.keys.len();
        self.keys.push(embedding.to_vec());
        self.values.push(stored_text);
        self.hits.push(1);
        self.last_access_tick.push(current_tick);
        self.learning_value.push(learning_value.clamp(-1.0, 1.0));
        self.kind.push(kind);

        // Voeg toe aan bucket
        let bucket = self.hash_to_bucket(embedding);
        if bucket < self.buckets.len() {
            self.buckets[bucket].push(idx);
        }
    }

    /// v2.6.8 P1.2: retrieve_with_kinds — retourneert (sim, text, kind) zodat de caller
    /// provenance kan classificeren zonder string-heuristiek. Updatet hits + last_access.
    pub fn retrieve_with_kinds(&mut self, query: &[f32; DIM], top_k: usize, current_tick: u64) -> Vec<(f32, String, u8)> {
        if self.keys.is_empty() { return Vec::new(); }
        self.ensure_parallel_vectors(current_tick);

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
        if candidates.len() < top_k * 2 {
            candidates = (0..self.keys.len()).collect();
        }
        let mut scored: Vec<(f32, usize)> = candidates.iter()
            .filter(|&&idx| idx < self.keys.len())
            .map(|&idx| (cosine_sim(query, &self.keys[idx]), idx))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored.into_iter()
            .map(|(sim, idx)| {
                self.hits[idx] = self.hits[idx].saturating_add(1);
                self.last_access_tick[idx] = current_tick;
                let k = self.kind.get(idx).copied().unwrap_or(KIND_EXTERNAL);
                (sim, self.values[idx].clone(), k)
            })
            .collect()
    }

    /// v2.6.8 P1.2: retrieve zonder hits-update (voor audit-only self-tests)
    pub fn retrieve_with_kinds_readonly(&self, query: &[f32; DIM], top_k: usize) -> Vec<(f32, String, u8)> {
        if self.keys.is_empty() { return Vec::new(); }

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
        if candidates.len() < top_k * 2 {
            candidates = (0..self.keys.len()).collect();
        }
        let mut scored: Vec<(f32, usize)> = candidates.iter()
            .filter(|&&idx| idx < self.keys.len())
            .map(|&idx| (cosine_sim(query, &self.keys[idx]), idx))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored.into_iter()
            .map(|(sim, idx)| {
                let k = self.kind.get(idx).copied().unwrap_or(KIND_EXTERNAL);
                (sim, self.values[idx].clone(), k)
            })
            .collect()
    }

    /// v2.6.7 E4: retrieve_mut — update hits/last_access voor gebruikte entries.
    /// Hoger-gebruikte entries overleven eviction langer.
    pub fn retrieve_mut(&mut self, query: &[f32; DIM], top_k: usize, current_tick: u64) -> Vec<(f32, String)> {
        if self.keys.is_empty() { return Vec::new(); }
        self.ensure_parallel_vectors(current_tick);

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
        if candidates.len() < top_k * 2 {
            candidates = (0..self.keys.len()).collect();
        }
        let mut scored: Vec<(f32, usize)> = candidates.iter()
            .filter(|&&idx| idx < self.keys.len())
            .map(|&idx| (cosine_sim(query, &self.keys[idx]), idx))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Update hits/last_access voor gekozen entries
        let result: Vec<(f32, String)> = scored.into_iter()
            .map(|(sim, idx)| {
                self.hits[idx] = self.hits[idx].saturating_add(1);
                self.last_access_tick[idx] = current_tick;
                (sim, self.values[idx].clone())
            })
            .collect();
        result
    }

    /// Zoek top-K meest relevante feiten voor een query (read-only, legacy API)
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
        // CRC32 checksum + backup rotation
        let checksum = crc32fast::hash(&bytes);
        let _ = std::fs::write(path.with_extension("crc32"), checksum.to_le_bytes());
        lucy_io::fs::rotate_backups(path, 5);
        tracing::info!(entries = self.keys.len(), "Retrieval store saved (atomic)");
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let (bytes, _src) = lucy_io::fs::load_with_fallback(path).ok_or("no primary or backup found")?;
        let mut store: Self = bincode::deserialize(&bytes)?;
        store.rebuild_buckets();
        tracing::info!(entries = store.keys.len(), "Retrieval store loaded");
        Ok(store)
    }
}

/// Cosine similarity tussen query en stored key
/// v2.6.8 P1.3: Vec<f32> × Vec<f32> cosine (voor self-metrics berekeningen op keys)
fn cosine_sim_vec(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom > 1e-8 { dot / denom } else { 0.0 }
}

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
