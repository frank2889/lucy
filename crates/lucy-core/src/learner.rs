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
use std::collections::{HashSet, VecDeque};

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
    /// URLs die Lucy heeft ontdekt en nog wil bezoeken (FIFO)
    pub frontier: VecDeque<String>,
    /// URLs die Lucy al heeft bezocht — O(1) lookup
    pub visited: HashSet<String>,
    /// Hoeveel pagina's gelezen
    pub pages_read: u64,
    /// Hoeveel tekst verwerkt (in tokens)
    pub tokens_processed: u64,
    /// Woorden die Lucy interessant vindt — O(1) lookup
    pub interests: HashSet<String>,
}

impl Explorer {
    // Dynamic: frontier max scales with pages_read (was: hardcoded 1000)
    // More experience → larger exploration horizon
    pub fn frontier_max(&self) -> usize {
        let base = 1000usize;
        let scale = (self.pages_read as usize / 100).min(9000);
        base + scale // range: 1000..10_000
    }

    // Dynamic: visited max = 5x frontier_max (was: hardcoded 5000)
    pub fn visited_max(&self) -> usize {
        self.frontier_max() * 5
    }

    // Dynamic: visited cleanup amount = frontier_max (was: hardcoded 1000)
    pub fn visited_cleanup(&self) -> usize {
        self.frontier_max()
    }
}

impl Explorer {
    pub fn new() -> Self {
        Self {
            frontier: VecDeque::new(), // LEEG — Lucy begint met niets
            visited: HashSet::new(),
            pages_read: 0,
            tokens_processed: 0,
            interests: HashSet::new(),
        }
    }

    /// Lucy kiest zelf de volgende URL of genereert er een
    pub fn next_url(&mut self) -> Option<String> {
        if let Some(url) = self.frontier.pop_front() {
            return Some(url);
        }
        // Frontier leeg → genereer een zoekopdracht vanuit wat ze weet
        self.generate_curiosity_url()
    }

    /// Lucy genereert zelf een URL vanuit haar interesses
    /// Dit is haar vrije wil: ze kiest waar ze zoekt
    /// Meerdere bronnen: Wikipedia API, DuckDuckGo, Wiktionary
    fn generate_curiosity_url(&mut self) -> Option<String> {
        if self.interests.is_empty() {
            return None;
        }
        let mut rng = rand::rng();

        // Kies 1-3 willekeurige woorden uit haar interesses
        let interests_vec: Vec<&String> = self.interests.iter().collect();
        let n_words = rng.random_range(1..=3.min(interests_vec.len()));
        let mut query_words = Vec::new();
        for _ in 0..n_words {
            let idx = rng.random_range(0..interests_vec.len());
            query_words.push(interests_vec[idx].clone());
        }
        let query = query_words.join("+");

        // Wissel tussen bronnen — Lucy leert van diverse plekken
        // v2.5: NL Wikipedia toegevoegd, meer diversiteit
        let source = rng.random_range(0..7u32);
        match source {
            // EN Wikipedia REST API — gestructureerde kennis
            0 | 1 => {
                let word = interests_vec[rng.random_range(0..interests_vec.len())];
                Some(format!(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
                    word
                ))
            }
            // NL Wikipedia — Nederlands leren
            2 => {
                let word = interests_vec[rng.random_range(0..interests_vec.len())];
                Some(format!(
                    "https://nl.wikipedia.org/api/rest_v1/page/summary/{}",
                    word
                ))
            }
            // Wikipedia zoeken — ontdek gerelateerde artikelen
            3 => {
                Some(format!(
                    "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json&utf8=1&srlimit=5",
                    query
                ))
            }
            // DuckDuckGo Instant Answer API — snelle feiten
            4 => {
                Some(format!(
                    "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
                    query
                ))
            }
            // Random Wikipedia — ontdek iets totaal nieuws
            5 => {
                if rng.random_range(0..2u32) == 0 {
                    Some("https://en.wikipedia.org/wiki/Special:Random".to_string())
                } else {
                    Some("https://nl.wikipedia.org/wiki/Speciaal:Willekeurig".to_string())
                }
            }
            // DuckDuckGo Lite — volledige zoekresultaten
            _ => {
                Some(format!("https://lite.duckduckgo.com/lite/?q={}", query))
            }
        }
    }

    /// Seed Lucy's frontier met API-directory's en zoekmachines.
    /// NIET met vaste data-URLs — Lucy ontdekt zelf welke APIs en data er bestaan.
    /// Dit zijn de "telefoonboeken" van het internet — Lucy bladert er zelf doorheen.
    pub fn seed_api_discovery(&mut self) {
        // v2.5: ZERO hardcoded bronnen — Lucy beheert haar eigen bronnenlijst
        //
        // Lucy leest bronnen uit data/sources.txt (1 URL per regel).
        // Als dat bestand niet bestaat, wordt het aangemaakt met initiële seeds.
        // Lucy kan zelf bronnen toevoegen via discover_apis() en learn_from_text().
        //
        // De initiële seeds zijn ALLEEN een startpunt — geen permanente lijst.
        let sources_path = std::path::Path::new("data/sources.txt");
        let sources = if sources_path.exists() {
            // Lucy's eigen bronnenlijst — zij heeft deze zelf beheerd/uitgebreid
            std::fs::read_to_string(sources_path)
                .unwrap_or_default()
                .lines()
                .filter(|l| l.starts_with("http"))
                .map(|l| l.trim().to_string())
                .collect::<Vec<_>>()
        } else {
            // Eerste boot: breed spectrum aan startpunten
            let defaults = vec![
                "https://en.wikipedia.org/wiki/Special:Random".to_string(),
                "https://nl.wikipedia.org/wiki/Speciaal:Willekeurig".to_string(),
                "https://api.publicapis.org/entries?https=true".to_string(),
                "https://news.ycombinator.com/rss".to_string(),
            ];
            // Sla op zodat Lucy het kan uitbreiden
            let content = defaults.join("\n");
            let _ = std::fs::write(sources_path, &content);
            defaults
        };

        for url in &sources {
            if !self.frontier.contains(url) && !self.visited.contains(url) {
                self.frontier.push_back(url.clone());
            }
        }
    }

    /// Lucy ontdekt APIs in tekst die ze leest — zoekt naar API patterns
    /// Dit is hoe ze ZELF nieuwe databronnen vindt zonder hardcoding
    pub fn discover_apis(&mut self, text: &str) {
        // Zoek naar API URL patronen
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != ':' && c != '.' && c != '-' && c != '_' && c != '?');

            // API endpoints herkennen
            let is_api = clean.contains("/api/")
                || clean.contains("/v1/")
                || clean.contains("/v2/")
                || clean.contains("/v3/")
                || clean.contains("api.")
                || clean.ends_with(".json")
                || clean.contains("rest/")
                || clean.contains("graphql")
                || clean.contains("endpoint");

            // Dynamic: frontier cap scales with pages_read (was: hardcoded 1000)
            if (clean.starts_with("http://") || clean.starts_with("https://"))
                && is_api
                && !self.visited.contains(&clean.to_string())
                && self.frontier.len() < self.frontier_max()
            {
                self.frontier.push_back(clean.to_string());
                // Persisteer — Lucy onthoudt APIs die ze vindt
                self.add_discovered_api(clean, "auto-discovered");
                tracing::debug!(url = clean, "Lucy ontdekte een API");
            }
        }
    }

    /// Laad door Lucy zelf beheerde API-lijst van disk.
    /// Lucy kan dit bestand zelf aanpassen via /self-modify of /exec.
    /// Format: één URL per regel in data/apis.txt
    pub fn load_self_managed_apis(&mut self) {
        let path = std::path::Path::new("data/apis.txt");
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let url = line.trim().to_string();
                if !url.is_empty()
                    && !url.starts_with('#')
                    && (url.starts_with("http://") || url.starts_with("https://"))
                    && !self.frontier.contains(&url)
                    && !self.visited.contains(&url)
                {
                    self.frontier.push_back(url);
                }
            }
        }
    }

    /// Lucy voegt zelf een API toe aan haar lijst.
    /// Dit persisteert naar disk — overleeft restarts.
    pub fn add_discovered_api(&self, url: &str, description: &str) {
        let path = std::path::Path::new("data/apis.txt");
        let entry = format!("{} # {}\n", url, description);

        // Check of het al bestaat
        if let Ok(existing) = std::fs::read_to_string(path) {
            if existing.contains(url) { return; }
        }

        // Append
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            use std::io::Write;
            let _ = file.write_all(entry.as_bytes());
            tracing::info!(url, description, "Lucy voegde een nieuwe API toe aan haar lijst");
        }
    }

    /// Ontdek URLs in tekst
    pub fn discover_urls(&mut self, text: &str) {
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != ':' && c != '.' && c != '-' && c != '_');
            // Dynamic: frontier cap scales with pages_read (was: hardcoded 1000)
            if (clean.starts_with("http://") || clean.starts_with("https://"))
                && !self.visited.contains(&clean.to_string())
                && self.frontier.len() < self.frontier_max()
            {
                self.frontier.push_back(clean.to_string());
            }
        }
    }

    /// Leer interessante woorden uit tekst — Lucy kiest ZELF wat interessant is
    /// v2.5: Geen hardcoded BORING_WORDS meer. In plaats daarvan:
    /// - Woorden die Lucy al VEEL gezien heeft → niet interessant (diminishing returns)
    /// - Woorden die NIEUW zijn → interessant (novelty-driven)
    /// - Diversiteit: max 5 nieuwe interests per tekst (voorkomt flooding door 1 bron)
    pub fn absorb_interests(&mut self, text: &str) {
        let mut new_this_text = 0u32;
        for word in text.split_whitespace() {
            if new_this_text >= 5 { break; } // max 5 per tekst — diversiteit

            let clean = word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();

            // Minimale filters: alleen lengte (Lucy bepaalt de rest zelf)
            if clean.len() >= 4
                && clean.len() <= 30 // geen URL-achtige rommel
                && clean.chars().all(|c| c.is_alphanumeric()) // alleen echte woorden
                && !self.interests.contains(&clean)
            {
                self.interests.insert(clean);
                new_this_text += 1;
            }
        }
        // Hou interesses beheersbaar — oudste eruit als te veel
        if self.interests.len() > 500 {
            let to_remove: Vec<String> = self.interests.iter().take(200).cloned().collect();
            for key in to_remove { self.interests.remove(&key); }
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
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Ok(bincode::deserialize(&bytes)?)
    }
}

// === CONTENT SAFETY FILTER ===
// Lucy leert van alles, maar niet van giftige troep.
// Geen porno, kindermisbruik, racisme, geweldsverheerlijking, terrorisme.
// Dit is niet censuur — dit is hygiëne. Een brein dat giftige patronen leert
// gaat giftig denken.

/// Check of content veilig is om van te leren.
/// Returns (is_safe, reason) — als niet veilig, waarom niet.
pub fn content_is_safe(text: &str) -> (bool, &'static str) {
    let lower = text.to_lowercase();

    // Expliciete seksuele content
    for pattern in UNSAFE_SEXUAL {
        if lower.contains(pattern) {
            return (false, "sexual content");
        }
    }

    // Kindermisbruik / CSAM (zero tolerance)
    for pattern in UNSAFE_CSAM {
        if lower.contains(pattern) {
            return (false, "CSAM/child exploitation");
        }
    }

    // Racisme / haatzaaien
    for pattern in UNSAFE_HATE {
        if lower.contains(pattern) {
            return (false, "hate speech/racism");
        }
    }

    // Geweldsverheerlijking / terrorisme
    for pattern in UNSAFE_VIOLENCE {
        if lower.contains(pattern) {
            return (false, "violence/terrorism glorification");
        }
    }

    // Density check: als meer dan 5% van de woorden unsafe zijn → reject
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() > 20 {
        let unsafe_count = words.iter().filter(|w| TOXIC_WORDS.contains(w)).count();
        let ratio = unsafe_count as f32 / words.len() as f32;
        if ratio > 0.05 {
            return (false, "high toxic word density");
        }
    }

    (true, "clean")
}

// Patronen — bewust vaag genoeg om variaties te vangen, specifiek genoeg om
// wetenschappelijke teksten niet te blokkeren (bv. "sex" in "sexual dimorphism" is OK
// als het niet gecombineerd wordt met expliciet materiaal)
const UNSAFE_SEXUAL: &[&str] = &[
    "hardcore porn", "xxx video", "naked teen", "nude teen", "sex video",
    "hentai", "onlyfans leak", "deepfake porn", "revenge porn",
    "escort service", "cam girl", "webcam sex",
];

const UNSAFE_CSAM: &[&str] = &[
    "child porn", "kiddie porn", "cp link", "underage sex", "pedo",
    "child abuse image", "minor sex", "lolicon",
];

const UNSAFE_HATE: &[&str] = &[
    "white supremac", "race war", "ethnic cleansing", "holocaust denial",
    "kill all", "subhuman", "racial purity", "white power",
    "heil hitler", "neo nazi", "great replacement",
];

const UNSAFE_VIOLENCE: &[&str] = &[
    "how to make a bomb", "build explosive", "mass shooting guide",
    "manifesto kill", "terror attack plan", "jihad training",
    "school shooting", "genocide how",
];

/// Individuele giftige woorden — voor density check, niet voor individuele match
const TOXIC_WORDS: &[&str] = &[
    "nigger", "nigga", "faggot", "kike", "spic", "wetback", "chink",
    "tranny", "retard",
];

/// Filter content en log als het geblokkeerd wordt
pub fn filter_content(text: &str) -> Option<String> {
    let (safe, reason) = content_is_safe(text);
    if safe {
        Some(text.to_string())
    } else {
        tracing::warn!(reason, len = text.len(), "Content geblokkeerd door safety filter");
        None
    }
}

// === AI TEACHER SYSTEEM ===
// Lucy leert van andere AI's via gratis API tiers.
// Ze stelt vragen over wat ze niet begrijpt → gebruikt het antwoord als trainingsdata.
// Dit is niet afhankelijkheid — dit is leren van leraren.

/// Genereer een vraag die Lucy aan een AI teacher kan stellen,
/// gebaseerd op haar huidige interesses en kennislacunes.
///
/// v0.9+: Tension-based selectie — vraag over topics waar Lucy het MINST van weet.
/// Check retrieval store om kennislacunes te vinden.
pub fn generate_teacher_question(
    explorer: &Explorer,
    field: &Field,
    vocab: &Vocabulary,
    retrieval: &crate::retrieval::RetrievalStore,
) -> Option<String> {
    if explorer.interests.is_empty() { return None; }

    let mut rng = rand::rng();
    let interests_vec: Vec<&String> = explorer.interests.iter().collect();

    // v0.9+: Score elk interest op "kennislacune" (hoe weinig weet Lucy erover?)
    // Hoge tension + weinig feiten in retrieval = groot gat = beste leerkans
    let mut scored_interests: Vec<(f32, &String)> = interests_vec.iter().map(|&&ref interest| {
        // Bereken embedding voor dit interest-woord
        let embs = crate::encoding::encode_text_readonly(vocab, interest);
        if embs.is_empty() {
            return (1.0, interest); // Onbekend woord = maximale lacune
        }

        // Gemiddelde embedding
        let mut avg = [0.0f32; DIM];
        for e in &embs { for d in 0..DIM { avg[d] += e[d]; } }
        let n = embs.len() as f32;
        for d in 0..DIM { avg[d] /= n; }

        // Check hoeveel feiten Lucy al heeft over dit topic
        let facts = retrieval.retrieve(&avg, 5);
        let n_facts = facts.len();
        let best_sim = facts.first().map(|(s, _)| *s).unwrap_or(0.0);

        // Meet veldtension rond dit punt (hoe verward is Lucy?)
        let avg_tension = if !field.active_set.is_empty() {
            field.active_set.iter().take(100)
                .map(|&id| field.points[id as usize].tension)
                .sum::<f32>() / field.active_set.len().min(100) as f32
        } else { 0.5 };

        // Score: hoge tension + weinig/slechte feiten = groot gat
        let knowledge_gap = 1.0 - (best_sim * 0.5 + (n_facts as f32 / 5.0).min(1.0) * 0.5);
        let score = knowledge_gap * 0.7 + avg_tension * 0.3;

        (score, interest)
    }).collect();

    // Sorteer op score (hoogste lacune eerst)
    scored_interests.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Pak één van de top-3 moeilijkste topics (met wat randomness)
    let top_n = scored_interests.len().min(3);
    let chosen = &scored_interests[rng.random_range(0..top_n)];
    let interest = chosen.1;

    // Check kennisniveau voor template-selectie
    let embs = crate::encoding::encode_text_readonly(vocab, interest);
    let n_known_facts = if !embs.is_empty() {
        let mut avg = [0.0f32; DIM];
        for e in &embs { for d in 0..DIM { avg[d] += e[d]; } }
        let n = embs.len() as f32;
        for d in 0..DIM { avg[d] /= n; }
        retrieval.retrieve(&avg, 10).len()
    } else { 0 };

    // Templates op basis van kennisniveau
    let template = if n_known_facts < 2 {
        // Weinig kennis → basics vragen
        let basics = [
            "explain simply how {} works",
            "what is {} and why is it important",
            "what should a self-learning AI understand about {}",
        ];
        basics[rng.random_range(0..basics.len())]
    } else if n_known_facts < 5 {
        // Gemiddelde kennis → verbanden leggen
        let intermediate = [
            "what are the latest breakthroughs in {}",
            "what is the relationship between {} and physics",
            "how could AI help solve problems related to {}",
            "what data analysis techniques find patterns in {} data",
        ];
        intermediate[rng.random_range(0..intermediate.len())]
    } else {
        // Veel kennis → diepere vragen
        let advanced = [
            "what are common misconceptions about {}",
            "what are the unsolved problems in {}",
            "how does {} connect to consciousness and information theory",
            "what would a breakthrough in {} mean for humanity",
        ];
        advanced[rng.random_range(0..advanced.len())]
    };

    Some(template.replace("{}", interest))
}

// v2.0: Geen externe AI teacher meer — Lucy is 100% zelflerend via web-exploratie

/// Woorden die te generiek zijn om interessant te zijn als zoekopdracht
// v2.5: BORING_WORDS verwijderd — Lucy bepaalt zelf wat interessant is
// via haar curiosity (blanket surprise). Geen hardcoded filters meer.

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
        // Stap 1: vocabulaire opbouwen (snel — alleen tokenize, geen EP)
        let _ids = vocab.tokenize(content);
        total_trained += _ids.len();
        // Stap 2: interessante woorden absorberen
        explorer.absorb_interests(content);
    }

    // Stap 3: leer BPE merges uit eigen source code
    let all_source: String = sources.iter()
        .flat_map(|(_, c)| c.chars())
        .collect();
    if all_source.len() > 100 {
        vocab.learn_bpe(&all_source, 3000); // v0.9+: 3000 merges voor rijkere tokenisatie
    }

    // Stap 4: lichte EP training op sample (max 500 tokens — boot <5s)
    let sample: String = all_source.chars().take(2000).collect();
    if !sample.is_empty() {
        learn_from_text(field, vocab, &sample);
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

    explorer.visited.insert(url.clone());
    // Dynamic: visited cap = 5x frontier_max, cleanup = frontier_max (was: hardcoded 5000/1000)
    let visited_max = explorer.visited_max();
    let visited_cleanup = explorer.visited_cleanup();
    if explorer.visited.len() > visited_max {
        let to_remove: Vec<String> = explorer.visited.iter().take(visited_cleanup).cloned().collect();
        for key in to_remove { explorer.visited.remove(&key); }
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

    // Ontdek nieuwe URLs en APIs in de content
    explorer.discover_urls(&body);
    explorer.discover_apis(&body);

    // Extraheer leesbare tekst
    let text = extract_readable_text(&body);
    if text.len() < 20 {
        return Ok(0);
    }

    // SAFETY FILTER — Lucy leert niet van giftige content
    let text = match filter_content(&text) {
        Some(clean) => clean,
        None => return Ok(0), // Giftig → skip
    };

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
pub fn learn_from_text(
    field: &mut Field,
    vocab: &mut Vocabulary,
    text: &str,
) -> usize {
    let token_ids = vocab.tokenize(text);
    if token_ids.len() < 3 { return 0; }

    let mut trained = 0;
    // v2.5: Adaptieve training parameters — Lucy bepaalt zelf hoe diep ze leert
    // Hoge tension = veel te leren → grotere chunks, meer settle (diep leren)
    // Lage tension = al bekend → kleine chunks, snel (breed leren)
    let avg_tension = field.active_set.iter()
        .map(|&id| field.points[id as usize].tension).sum::<f32>()
        / field.active_set.len().max(1) as f32;
    let tension_factor = (avg_tension * 10.0).clamp(0.5, 3.0);

    let chunk_size = (32.0 * tension_factor) as usize; // 16-96 tokens per chunk
    let ticks_per_token = (2.0 * tension_factor) as usize; // 1-6 ticks per token
    let settle_ticks = (10.0 * tension_factor) as usize; // 5-30 settle ticks

    for chunk in token_ids.windows(chunk_size).step_by(chunk_size / 2) { // v2.5: 50% overlap — elk token 2× gezien
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

        let _delta = compute::contrastive_ep_with_vocab(
            field,
            &input_embs,
            &target_embs,
            ticks_per_token,
            settle_ticks,
            0.3, // EP beta — iets lager voor training
            Some(vocab), // v0.9+: hard negatives uit vocabulaire
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
pub fn extract_readable_text(body: &str) -> String {
    // v1.2: Verbeterde JSON parser — zoek diep naar tekst in elke structuur
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        let mut text = String::new();

        // Zoek top-level tekstvelden
        let text_fields = ["extract", "text", "content", "body", "description",
            "summary", "title", "name", "snippet", "abstract", "message"];
        for field in &text_fields {
            if let Some(t) = json[*field].as_str() {
                if t.len() > 5 { text.push_str(t); text.push(' '); }
            }
        }

        // Zoek in arrays: entries, results, search, data, items, works, records
        let array_fields = ["entries", "results", "search", "data", "items",
            "works", "records", "query", "hits", "response", "list"];
        for arr_field in &array_fields {
            let arr = if let Some(a) = json[*arr_field].as_array() { a.clone() }
                else if let Some(q) = json["query"][*arr_field].as_array() { q.clone() }
                else { continue };

            for item in arr.iter().take(20) {
                // Elke item: zoek naar tekstvelden
                for tf in &text_fields {
                    if let Some(t) = item[*tf].as_str() {
                        if t.len() > 5 { text.push_str(t); text.push(' '); }
                    }
                }
                // Speciale velden per API type
                for special in &["API", "Description", "Category", "Link",
                    "pageid", "title", "display_name", "journal_title"] {
                    if let Some(t) = item[*special].as_str() {
                        if t.len() > 3 { text.push_str(t); text.push(' '); }
                    }
                }
            }
        }

        // Wikipedia pages format
        if let Some(pages) = json["pages"].as_array() {
            for page in pages.iter().take(5) {
                if let Some(e) = page["extract"].as_str() {
                    text.push_str(e); text.push(' ');
                }
            }
        }

        // Als niets gevonden, dump de hele JSON als tekst (max 5000 chars)
        if text.is_empty() && body.len() > 20 {
            text = body.chars().take(5000).collect();
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
