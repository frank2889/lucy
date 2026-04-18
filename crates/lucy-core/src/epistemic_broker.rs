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
    let q = concept.split_whitespace().collect::<Vec<_>>().join("+");

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_urls_for_concept() {
        let urls = urls_for_concept("consciousness");
        assert_eq!(urls.len(), 3);
        assert!(urls[0].contains("wikipedia"));
        assert!(urls[1].contains("arxiv"));
        assert!(urls[2].contains("semanticscholar"));
    }

    #[test]
    fn test_concept_sanitization() {
        let urls = urls_for_concept("self-awareness & identity");
        for url in urls {
            assert!(!url.contains("&") || url.contains("&amp;") || url.contains("&query"));
        }
    }
}
