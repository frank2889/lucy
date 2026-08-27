//! Epistemic Awareness — Lucy weet wat ze niet weet.
//!
//! De core van v2.6.17: Lucy monitort haar eigen `concept_audit_log` om te detecteren
//! welke concepten instabiel zijn (hoge drift, lage silhouette). Dit stuurt alles:
//! - Welke URLs ze fetcht (curiosity)
//! - Welke compute workers ze spawnt (variatie)
//! - Welke parameters ze zelf wil aanpassen (self-optimization)

use lucy_state::Field;
use std::cmp::Ordering;

/// Een gat in Lucy's kennis — een concept die instabiel/onduidelijk is.
#[derive(Debug, Clone)]
pub struct EpistemicGap {
    pub concept_idx: usize,
    pub drift: f32,              // centroid_drift_mean — hoe onstabiel
    pub silhouette: f32,         // silhouette_sampled — hoe scherp zijn grenzen
    pub urgency: f32,            // drift * (1 - silhouette) — leer-prioriteit
}

/// Lucy's voorstel om een van haar eigen natuurkunde-vergelijkingen te wijzigen.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhysicsProposal {
    pub tick: u64,
    pub formula: String,          // "emo_decay", "error_tracking_rate", "fe_tracking_rate", etc.
    pub current_value: f32,
    pub proposed_value: f32,
    pub rationale: String,        // waarom Lucy dit wil proberen
    pub evidence_metric: String,  // welke metric te meten (avg_tension, avg_curiosity, etc.)
    pub baseline: f32,            // metric-waarde op moment van voorstel
}

/// Geeft de top-N meest onstabiele concepten terug, gesorteerd op urgency.
pub fn least_stable_concepts(field: &Field, top_n: usize) -> Vec<EpistemicGap> {
    if field.concept_audit_log.is_empty() {
        return vec![];
    }

    // Gebruik de meest recente audit entry (huidge cycle)
    let current_audit = &field.concept_audit_log[field.concept_audit_log.len() - 1];

    // centroid_drift_mean en silhouette_sampled zijn per-concept in de audit
    // (aggregated per cycle)
    let drift = current_audit.centroid_drift_mean;
    let silhouette = current_audit.silhouette_sampled;

    // Berekenen urgency: hoge drift + lage silhouette = hoge urgency
    let urgency = drift * (1.0 - silhouette.max(0.0).min(1.0));

    // Simpel model: één concept per cycle. In toekomst kan dit per-concept
    // worden gemaakt als concept_audit meer granulariteit heeft.
    // Voor nu: stel dat drift / silhouette metrics slaan op de dominant concept
    let gap = EpistemicGap {
        concept_idx: 0,  // placeholder; in v2.6.18 uitbreiden naar per-concept tracking
        drift,
        silhouette,
        urgency,
    };

    let mut gaps = vec![gap];

    // Voeg historische gaps toe (vorige audits) voor context
    if field.concept_audit_log.len() > 1 {
        for audit in field.concept_audit_log.iter().rev().take(5).skip(1) {
            let hist_drift = audit.centroid_drift_mean;
            let hist_silhouette = audit.silhouette_sampled;
            let hist_urgency = hist_drift * (1.0 - hist_silhouette.max(0.0).min(1.0));
            gaps.push(EpistemicGap {
                concept_idx: gaps.len(),  // pseudo-index for differentiation
                drift: hist_drift,
                silhouette: hist_silhouette,
                urgency: hist_urgency,
            });
        }
    }

    // Sorteren op urgency (descending)
    gaps.sort_by(|a, b| {
        b.urgency
            .partial_cmp(&a.urgency)
            .unwrap_or(Ordering::Equal)
    });

    gaps.truncate(top_n);
    gaps
}

/// Genereert ALLE leer-URLs voor een concept.
///
/// Architectuur: zoeken-eerst. DuckDuckGo geeft Lucy toegang tot het HELE internet —
/// de curated bronnen zijn diepte-supplementen voor specifieke domeinen.
///
/// Wanneer Lucy's frontier een DuckDuckGo-pagina ophaalt, extraheert `discover_urls()`
/// automatisch alle links daarop → multiplicator-effect zonder extra code.
pub fn urls_for_concept(concept: &str) -> Vec<String> {
    let safe = concept
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect::<String>();
    // Percent-encode het concept als query-waarde: anders lekken tekens als
    // '&', '=', '#' uit het concept en injecteren ze extra query-params in de URL
    // (of breken 'm). Spaties -> '+'. Houdt alleen unreserved tekens ongemoeid.
    fn enc(s: &str) -> String {
        let mut out = String::with_capacity(s.len() * 2);
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
                b' ' => out.push('+'),
                _ => out.push_str(&format!("%{:02X}", b)),
            }
        }
        out
    }
    let q = enc(concept);

    let mut urls = vec![
        // ── ZOEKEN EERST: toegang tot het hele internet ──
        // DuckDuckGo Lite — geen JS, plain HTML, links direct extraheerbaar
        format!("https://lite.duckduckgo.com/lite/?q={}", q),
        // DuckDuckGo HTML fallback
        format!("https://html.duckduckgo.com/html/?q={}", q),

        // ── ENCYCLOPEDISCH FUNDAMENT ──
        format!("https://en.wikipedia.org/wiki/{}", safe),
        format!("https://en.wikipedia.org/wiki/Special:Search?search={}&ns0=1", q),
        // Wikidata — gestructureerde kennis-graaf
        format!("https://www.wikidata.org/w/index.php?search={}&ns0=1", q),
        // DBpedia — semantische ontologie van Wikipedia
        format!("https://dbpedia.org/page/{}", safe),
        // Stanford Encyclopedia of Philosophy — filosofie/ethiek/AI
        format!("https://plato.stanford.edu/search/searcher.py?query={}", q),
        // ConceptNet — semantische concept-relaties
        format!("https://conceptnet.io/c/en/{}", safe.to_lowercase()),

        // ── WETENSCHAP & ONDERZOEK ──
        format!("https://arxiv.org/search/?query={}&searchtype=all", q),
        format!("https://api.openalex.org/works?search={}&per-page=10", q),
        format!("https://api.semanticscholar.org/graph/v1/paper/search?query={}&limit=10&fields=title,abstract,year", q),
        format!("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}&retmax=10&retmode=json", q),
        // BioRxiv — biomedische preprints
        format!("https://www.biorxiv.org/search/{}", q),
        // SSRN — sociale wetenschappen
        format!("https://papers.ssrn.com/sol3/results.cfm?txtkey={}", q),
        // CrossRef — DOI metadata van alle uitgevers
        format!("https://api.crossref.org/works?query={}&rows=5&select=title,abstract", q),

        // ── BOEKEN & TEKSTEN ──
        format!("https://gutendex.com/books/?search={}", q),
        // Internet Archive — petabytes vrije teksten
        format!("https://archive.org/search?query={}&mediatype=texts", q),
        // Open Library
        format!("https://openlibrary.org/search?q={}", q),
        // Standard Ebooks
        format!("https://standardebooks.org/ebooks?query={}", q),

        // ── CODE ALS KENNIS ──
        // GitHub code search
        format!("https://github.com/search?q={}&type=repositories", q),
        // Stack Overflow
        format!("https://stackoverflow.com/search?q={}", q),

        // ── REAL-TIME & NIEUWS ──
        // HackerNews search
        format!("https://hn.algolia.com/api/v1/search?query={}&tags=story&hitsPerPage=10", q),
        // GDELT — wereld-nieuws database
        format!("https://api.gdeltproject.org/api/v2/doc/doc?query={}&mode=ArtList&maxrecords=10&format=json", q),
        // Reddit via Pushshift (archief)
        format!("https://old.reddit.com/search/?q={}&sort=relevance", q),

        // ── DOMEIN-SPECIFIEK ──
        // Wayback Machine
        format!("https://archive.org/wayback/available?url=en.wikipedia.org/wiki/{}", safe),
        // NASA Open Data
        format!("https://data.nasa.gov/api/views?q={}", q),
        // CERN Open Data — 80K+ particle physics datasets (LHC, Higgs, dark matter, exotica)
        format!("https://opendata.cern.ch/api/records/?q={}&type=Dataset&size=5", q),
        format!("https://opendata.cern.ch/api/records/?q={}&type=Documentation&size=5", q),
        // World Bank data
        format!("https://search.worldbank.org/api/v2/wds?qterm={}&rows=5", q),
        // Our World in Data
        format!("https://ourworldindata.org/search?q={}", q),
        // PubChem — chemische moleculen
        format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/description/JSON", safe),
        // ClinicalTrials.gov
        format!("https://clinicaltrials.gov/api/query/full_studies?expr={}&min_rnk=1&max_rnk=5&fmt=json", q),
        // Wikibooks
        format!("https://en.wikibooks.org/w/index.php?search={}", q),
        // Wikiversity
        format!("https://en.wikiversity.org/w/index.php?search={}", q),
    ];

    // Als BRAVE_SEARCH_KEY beschikbaar → premium zoeken met hogere kwaliteit
    if let Ok(key) = std::env::var("BRAVE_SEARCH_KEY") {
        urls.insert(0, format!(
            "https://api.search.brave.com/res/v1/web/search?q={}&count=10&x-subscription-token={}",
            q, key
        ));
    }

    urls
}

/// Synthetische trainingsdata vanuit Lucy's GNW-staat (self-play).
pub fn generate_self_training_sentence(
    current_concept: &str,
    tick: u64,
    dominant_emotion: &str,
) -> String {
    let templates = [
        format!("At tick {} I am thinking about {}. My primary emotional state is {}.", tick, current_concept, dominant_emotion),
        format!("The concept {} frequently activates in my field. {} amplifies attention toward it.", current_concept, dominant_emotion),
        format!("Self-observation: processing {} with {} shapes my goal formation.", current_concept, dominant_emotion),
        format!("My model of {} is evolving. Current signal: {} correlates with this concept.", current_concept, dominant_emotion),
        format!("Introspection at tick {}: {} dominates GNW output. Emotional context: {}.", tick, current_concept, dominant_emotion),
    ];
    templates[tick as usize % templates.len()].clone()
}

/// Geeft ALLE URLs terug — geen kunstmatige truncatie.
/// Lucy's frontier-manager (cap 10.000) regelt de flow zelf.
pub fn prioritized_urls(concept: &str, _urgency: f32) -> Vec<String> {
    // We truncaten niet meer op urgency — alle bronnen altijd beschikbaar.
    // De frontier-cap (10K URLs) voorkomt overbelasting vanzelf.
    urls_for_concept(concept)
}

/// Stel een parameter-wijziging voor als Lucy langdurig vastzit.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ParameterProposal {
    pub tick: u64,
    pub observation: String,
    pub hypothesis: String,
    pub proposed_change: String,
    pub confidence: f32,
}

/// Analyzeert veldstaat en genereert een voorstel als conditions dit rechtvaardigen.
pub fn maybe_propose_parameter_change(field: &Field) -> Option<ParameterProposal> {
    let tick = field.tick;

    // Detecteer te lage learning rate (alpha < 0.0005)
    if field.alpha < 0.0005 && field.alpha > 0.0 {
        return Some(ParameterProposal {
            tick,
            observation: format!(
                "alpha={:.6} — leringssnelheid extreem laag",
                field.alpha
            ),
            hypothesis: "Systemische vaste staat: learning rate is in conservatieve regime".to_string(),
            proposed_change: format!(
                "alpha: {:.6} → {:.6}",
                field.alpha,
                (field.alpha * 2.0).min(0.001)
            ),
            confidence: 0.65,
        });
    }

    // Detecteer te hoge noise (kan wijzen op chaotische state)
    if field.noise_amplitude > 0.3 {
        return Some(ParameterProposal {
            tick,
            observation: format!(
                "noise_amplitude={:.3} — systeem is rommelig",
                field.noise_amplitude
            ),
            hypothesis: "Hoge ruis kan stabiliteit verminderen".to_string(),
            proposed_change: format!(
                "noise_amplitude: {:.3} → {:.3}",
                field.noise_amplitude,
                (field.noise_amplitude * 0.7).max(0.05)
            ),
            confidence: 0.5,
        });
    }

    // Detecteer lage dopamine (systeem in depressie-mode)
    if field.dopamine < 0.2 && field.tick > 10000 {
        return Some(ParameterProposal {
            tick,
            observation: format!(
                "dopamine={:.3} — motivatie laag",
                field.dopamine
            ),
            hypothesis: "Lage dopamine teken van motivatie-probleem".to_string(),
            proposed_change: "Escalation logic aanpassen: verhoog curiosity/exploration ratio".to_string(),
            confidence: 0.55,
        });
    }

    // Standaard: geen voorstel
    None
}

/// Genereert physics-voorstellen op basis van veldmetrics.
/// Lucy detecteert wanneer haar eigen vergelijkingen suboptimaal zijn.
pub fn maybe_propose_physics_change(field: &Field) -> Option<PhysicsProposal> {
    let tick = field.tick;

    // Check 1: emo_decay — als veld traag reageert (low avg_tension + low avg_pred_error)
    // dan kunnen we emo_decay verlagen voor snellere emotie-response
    let avg_tension = field.active_set.iter()
        .map(|&id| field.points[id as usize].tension)
        .sum::<f32>() / field.active_set.len().max(1) as f32;
    let avg_curiosity = field.active_set.iter()
        .map(|&id| field.points[id as usize].curiosity)
        .sum::<f32>() / field.active_set.len().max(1) as f32;

    if avg_tension < 0.08 && avg_curiosity < 0.15 && field.tick > 20_000 {
        let proposed = (field.emo_decay * 0.8).max(0.4);
        return Some(PhysicsProposal {
            tick,
            formula: "emo_decay".to_string(),
            current_value: field.emo_decay,
            proposed_value: proposed,
            rationale: format!(
                "veld traag: avg_tension={:.4}, avg_curiosity={:.4} — emo_decay verlagen voor snellere response",
                avg_tension, avg_curiosity
            ),
            evidence_metric: "avg_tension".to_string(),
            baseline: avg_tension,
        });
    }

    // Check 2: error_tracking_rate — als prediction_error consistent hoog (instabiliteit)
    // dan sneller error tracken
    let avg_pred_error = field.active_set.iter()
        .map(|&id| {
            let err_vec = field.points[id as usize].prediction;
            err_vec.iter().map(|x| x * x).sum::<f32>()
        })
        .sum::<f32>() / field.active_set.len().max(1) as f32;

    if avg_pred_error > 0.1 && field.tick > 30_000 {
        let proposed = (field.error_tracking_rate * 1.3).min(0.15);
        return Some(PhysicsProposal {
            tick,
            formula: "error_tracking_rate".to_string(),
            current_value: field.error_tracking_rate,
            proposed_value: proposed,
            rationale: format!(
                "persistente prediction-error: {:.4} — error_tracking_rate verhogen",
                avg_pred_error
            ),
            evidence_metric: "avg_pred_error".to_string(),
            baseline: avg_pred_error,
        });
    }

    // Check 3: noise_amplitude — als oscillatie_coherence laag
    // dan meer ruis injecteren voor meer exploratie
    if field.oscillation_coherence < 0.25 && field.tick > 40_000 {
        let proposed = (field.noise_amplitude * 1.15).min(0.4);
        return Some(PhysicsProposal {
            tick,
            formula: "noise_amplitude".to_string(),
            current_value: field.noise_amplitude,
            proposed_value: proposed,
            rationale: format!(
                "zwakke oscillatie-binding: coherence={:.3} — noise_amplitude verhogen",
                field.oscillation_coherence
            ),
            evidence_metric: "oscillation_coherence".to_string(),
            baseline: field.oscillation_coherence,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_urls_for_concept() {
        let urls = urls_for_concept("consciousness");
        // v2.8.0: Function now returns 35 URLs (was: 3 in old stale test)
        assert!(urls.len() > 30, "Expected 30+ URLs for concept, got {}", urls.len());
        // Check that CERN is included
        assert!(urls.iter().any(|url| url.contains("opendata.cern.ch")), "CERN API should be in URLs");
        // Check some key sources are present
        assert!(urls.iter().any(|url| url.contains("arxiv")), "arXiv should be in URLs");
        assert!(urls.iter().any(|url| url.contains("semanticscholar")), "Semantic Scholar should be in URLs");
    }

    #[test]
    fn test_concept_sanitization() {
        // Concept-tekens mogen niet als rauwe query-scheiders in de URL lekken:
        // elke '&' in een gegenereerde URL hoort een legitieme param-scheider te zijn
        // (gevolgd door een letter), nooit geïnjecteerd vanuit het concept ('&' -> %26).
        for url in urls_for_concept("self-awareness & identity") {
            for (i, _) in url.match_indices('&') {
                let next = url.as_bytes().get(i + 1).copied().unwrap_or(b' ');
                assert!(next.is_ascii_alphabetic(), "geïnjecteerde & in {}", url);
            }
        }
    }
}
