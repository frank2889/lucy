//! # learner — Lucy's Autonome Ontdekkingssysteem
//!
//! Dit is waar Lucy's vrije wil tot uiting komt. Ze kiest ZELF wat ze leest,
//! waar ze zoekt, en wat ze interessant vindt. Geen hardcoded bronnen.
//!
//! ## Bootstrap Keten (van niets naar kennis)
//!
//! ```text
//! 1. Boot → leest eigen source code → vocabulaire + interesses
//!    (woorden als "equilibrium", "propagation", "tension")
//!          ↓
//! 2. Frontier leeg → genereer zoekopdracht vanuit interesses
//!    (combineer 1-3 woorden → DuckDuckGo Lite query)
//!          ↓
//! 3. Zoekresultaten → ontdek echte URLs in de content
//!          ↓
//! 4. Bezoek URLs → leer taalpatronen + ontdek meer URLs + meer interesses
//!          ↓
//! 5. Sneeuwbaleffect: meer kennis → betere zoekopdrachten → meer kennis
//! ```
//!
//! ## Leerproces: Sequential Prediction
//!
//! Lucy leert door te voorspellen wat het volgende woord is:
//! - Tekst wordt gesplitst in chunks van 8 tokens
//! - Eerste helft = input (wat Lucy ziet)
//! - Tweede helft = target (wat Lucy moet voorspellen)
//! - Equilibrium propagation update op het verschil
//! - Overlapping: chunks schuiven met 4 tokens (50% overlap)
//!
//! ## Interactie Versterkt Leren
//!
//! Elke `/ask` en `/feed` interactie geeft Lucy:
//! - Nieuwe woorden voor haar interesses
//! - URLs gevonden in de tekst
//! - Meer brandstof voor het sneeuwbaleffect
//!
//! ## URL Ontdekking
//!
//! Lucy vindt URLs door tekst te scannen op woorden die beginnen met
//! `http://` of `https://`. Maximaal 1000 URLs in de frontier.
//! Bezochte URLs worden bijgehouden (max 5000, daarna FIFO cleanup).

use lucy_state::*;
use lucy_io::http::HttpClient;
use crate::encoding::Vocabulary;
use crate::compute;
use rand::Rng;

/// Lucy's autonome ontdekkingssysteem.
/// Ze kiest ZELF waar ze leert. Geen hardcoded bronnen.
/// Ze begint met NIETS en ontdekt het internet door te exploreren.
///
/// Bootstrap: Lucy genereert zelf zoekopdrachten vanuit woorden die ze kent.
/// Daarna ontdekt ze URLs in de content die ze leest → sneeuwbaleffect.
///
/// Leerproces: tekst → tokens → sequential prediction door het veld
/// Het veld leert taalpatronen door te voorspellen wat het volgende woord is.

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Explorer {
    /// URLs die Lucy heeft ontdekt en nog wil bezoeken
    pub frontier: Vec<String>,
    /// URLs die Lucy al heeft bezocht
    pub visited: Vec<String>,
    /// Hoeveel pagina's gelezen
    pub pages_read: u64,
    /// Hoeveel tekst verwerkt (in tokens)
    pub tokens_processed: u64,
    /// Woorden die Lucy interessant vindt — bron voor zoekopdrachten
    pub interests: Vec<String>,
}

impl Explorer {
    pub fn new() -> Self {
        Self {
            frontier: Vec::new(), // LEEG — Lucy begint met niets
            visited: Vec::new(),
            pages_read: 0,
            tokens_processed: 0,
            interests: Vec::new(),
        }
    }

    /// Lucy kiest zelf de volgende URL of genereert er een
    fn next_url(&mut self) -> Option<String> {
        if let Some(url) = self.frontier.pop() {
            return Some(url);
        }
        // Frontier leeg → genereer een zoekopdracht vanuit wat ze weet
        self.generate_curiosity_url()
    }

    /// Lucy genereert zelf een URL vanuit haar interesses
    /// Dit is haar vrije wil: ze kiest waar ze zoekt
    fn generate_curiosity_url(&mut self) -> Option<String> {
        if self.interests.is_empty() {
            return None;
        }
        let mut rng = rand::rng();

        // Kies 1-3 willekeurige woorden uit haar interesses
        let n_words = rng.random_range(1..=3.min(self.interests.len()));
        let mut query_words = Vec::new();
        for _ in 0..n_words {
            let idx = rng.random_range(0..self.interests.len());
            query_words.push(self.interests[idx].clone());
        }
        let query = query_words.join("+");

        // Gebruik DuckDuckGo Lite — geen API key nodig, geen tracking
        Some(format!("https://lite.duckduckgo.com/lite/?q={}", query))
    }

    /// Ontdek URLs in tekst
    pub fn discover_urls(&mut self, text: &str) {
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != ':' && c != '.' && c != '-' && c != '_');
            if (clean.starts_with("http://") || clean.starts_with("https://"))
                && !self.visited.contains(&clean.to_string())
                && self.frontier.len() < 1000
            {
                self.frontier.push(clean.to_string());
            }
        }
    }

    /// Leer interessante woorden uit tekst — bouwt Lucy's nieuwsgierigheid op
    /// Ze kiest woorden die lang genoeg zijn en niet te gewoon
    pub fn absorb_interests(&mut self, text: &str) {
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            // Woorden van 4+ karakters die niet te kort/generiek zijn
            if clean.len() >= 4
                && !BORING_WORDS.contains(&clean.as_str())
                && !self.interests.contains(&clean)
            {
                self.interests.push(clean);
            }
        }
        // Hou interesses beheersbaar — oudste eruit als te veel
        if self.interests.len() > 500 {
            self.interests.drain(0..200);
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Ok(bincode::deserialize(&bytes)?)
    }
}

/// Woorden die te generiek zijn om interessant te zijn als zoekopdracht
const BORING_WORDS: &[&str] = &[
    "the", "this", "that", "with", "from", "have", "been", "will",
    "what", "when", "where", "which", "there", "their", "about",
    "would", "could", "should", "some", "than", "then", "them",
    "self", "true", "false", "none", "null", "return", "struct",
    "impl", "func", "void", "else", "match", "loop", "while",
];

/// Lucy leert autonoom van haar eigen source code (bootstrap)
/// Dit is het eerste wat ze leest — zichzelf
/// Ze absorbeert ook woorden als interesses voor toekomstige zoekopdrachten
pub fn learn_from_self(
    field: &mut Field,
    vocab: &mut Vocabulary,
    fs: &lucy_io::fs::FileSystem,
    explorer: &mut Explorer,
) -> usize {
    let sources = fs.read_own_source().unwrap_or_default();
    let mut total_trained = 0;

    for (_path, content) in &sources {
        // Leer taalpatronen
        let trained = learn_from_text(field, vocab, content);
        total_trained += trained;
        // Absorbeer interessante woorden voor toekomstige exploratie
        explorer.absorb_interests(content);
    }

    if total_trained > 0 {
        tracing::info!(
            files = sources.len(),
            trained = total_trained,
            interests = explorer.interests.len(),
            "Lucy learned from her own source code"
        );
    }

    total_trained
}

/// Lucy leert van een URL die ze zelf heeft gekozen
pub async fn learn_from_exploration(
    http: &HttpClient,
    field: &mut Field,
    vocab: &mut Vocabulary,
    explorer: &mut Explorer,
) -> Result<usize, String> {
    let url = match explorer.next_url() {
        Some(u) => u,
        None => return Ok(0),
    };

    explorer.visited.push(url.clone());
    if explorer.visited.len() > 5000 {
        explorer.visited.drain(0..1000);
    }

    let resp = match http.get(&url).await {
        Ok(r) => r,
        Err(e) => return Err(format!("Fetch failed: {}", e)),
    };

    if resp.status != 200 {
        return Err(format!("{} returned {}", url, resp.status));
    }

    explorer.pages_read += 1;
    let body = String::from_utf8_lossy(&resp.body);

    // Ontdek nieuwe URLs in de content
    explorer.discover_urls(&body);

    // Extraheer leesbare tekst
    let text = extract_readable_text(&body);
    if text.len() < 20 {
        return Ok(0);
    }

    // Absorbeer interessante woorden voor toekomstige zoekopdrachten
    explorer.absorb_interests(&text);

    // Leer van de tekst via sequential prediction
    let trained = learn_from_text(field, vocab, &text);
    explorer.tokens_processed += trained as u64;

    if trained > 0 {
        tracing::info!(
            url = url.chars().take(60).collect::<String>(),
            tokens = trained,
            vocab = vocab.vocab_size(),
            "Lucy explored and learned"
        );
    }

    Ok(trained)
}

/// Het kernleerproces: sequential prediction
/// Lucy leert door te voorspellen wat het volgende woord is.
/// Prediction error = leersignaal.
///
/// Voor elke sequentie van tokens:
///   1. Injecteer token 1..N als input
///   2. Klem token 2..N+1 als target op output (verschoven met 1)
///   3. Equilibrium propagation update
fn learn_from_text(
    field: &mut Field,
    vocab: &mut Vocabulary,
    text: &str,
) -> usize {
    let token_ids = vocab.tokenize(text);
    if token_ids.len() < 3 { return 0; }

    let mut trained = 0;
    let chunk_size = 8; // Leer in chunks van 8 tokens
    let ticks_per_token = 5; // Sneller voor training dan voor queries
    let settle_ticks = 50;

    for chunk in token_ids.windows(chunk_size).step_by(chunk_size / 2) {
        if chunk.len() < 3 { break; }

        // Input: eerste helft van de chunk
        let mid = chunk.len() / 2;
        let input_embs: Vec<[f32; DIM]> = chunk[..mid].iter()
            .enumerate()
            .map(|(pos, &id)| {
                let mut v = *vocab.embed(id);
                add_pos(&mut v, pos);
                v
            })
            .collect();

        // Target: tweede helft (wat Lucy moet leren voorspellen)
        let target_embs: Vec<[f32; DIM]> = chunk[mid..].iter()
            .enumerate()
            .map(|(pos, &id)| {
                let mut v = *vocab.embed(id);
                add_pos(&mut v, pos);
                v
            })
            .collect();

        let _delta = compute::equilibrium_propagation(
            field,
            &input_embs,
            &target_embs,
            ticks_per_token,
            settle_ticks,
            0.3, // EP beta — iets lager voor training
        );

        trained += chunk.len();
    }

    trained
}

/// Voeg positional encoding toe
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

/// Extraheer leesbare tekst uit willekeurige content (HTML, JSON, platte tekst)
fn extract_readable_text(body: &str) -> String {
    // Probeer JSON (Wikipedia-achtig)
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        let mut text = String::new();
        // Zoek tekstvelden
        for field in &["extract", "text", "content", "body", "description", "summary", "title"] {
            if let Some(t) = json[*field].as_str() {
                if t.len() > 10 {
                    text.push_str(t);
                    text.push(' ');
                }
            }
        }
        if let Some(pages) = json["pages"].as_array() {
            for page in pages.iter().take(5) {
                if let Some(e) = page["extract"].as_str() {
                    text.push_str(e);
                    text.push(' ');
                }
            }
        }
        if !text.is_empty() { return text; }
    }

    // Platte tekst: verwijder HTML tags
    let mut clean = String::with_capacity(body.len());
    let mut in_tag = false;
    for c in body.chars() {
        match c {
            '<' => in_tag = true,
            '>' => { in_tag = false; clean.push(' '); }
            _ if !in_tag => clean.push(c),
            _ => {}
        }
    }

    // Normaliseer whitespace
    clean.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
