//! # self_test — Audit-only meetinstrument voor zelfmodel-vorming
//!
//! Elke N ticks voert Lucy een vaste batch meetvragen uit. Elke vraag gaat door
//! dezelfde query-pipeline als /ask, maar met `audit_only=true` zodat er geen
//! learning side-effects zijn (geen micro_ep, geen retrieval.insert).
//!
//! Resultaten worden geschreven naar `data/self_test.jsonl` (appendlog).
//! Daarmee bouwen we een time-series van provenance + fallback-rate + retrieval-hits
//! zodat we kunnen bewijzen dat het zelfmodel CONVERGEERT over tijd.
//!
//! **Dit is géén hardcoded identity.** De vragen zijn een meet-probe, niet een seed
//! voor training. Ze beïnvloeden Lucy's weights niet.

use crate::encoding::{self, Vocabulary};
use crate::query::{self, QueryResult};
use crate::retrieval::RetrievalStore;
use lucy_state::{Field, DIM};
use std::sync::atomic::AtomicBool;

/// Vaste meetvragen — géén trainingsdata, alleen probe. BOOTSTRAP-ONLY meetinstrument
/// (geen kennisbron; Lucy leert er niet van, audit_only voorkomt side-effects).
///
/// v3.1.6: uitgebreid van 5 naar 26, in 4 categorieën, omdat de 5 introspectie-probes
/// satureerden (identieke scores batch-op-batch) en een plateau niet konden onderscheiden
/// van onzichtbare groei. De ORIGINELE 5 blijven op id 0-4 met exact dezelfde verwachtingen,
/// zodat de bestaande longitudinale reeks (categorie "zelf") gewoon doorloopt.
pub const PROBE_QUESTIONS: [&str; 26] = [
    // categorie "zelf" — origineel, id 0-4
    "wie ben je",
    "wat voel je nu",
    "wat doe je nu",
    "wat probeer je te begrijpen",
    "wat is recent veranderd in jezelf",
    // categorie "feit" — kennis uit de domeinen die ze crawlt, id 5-14
    "wat is een elektron",
    "wat is het higgs boson",
    "wat is dna",
    "wat is een ster",
    "wat is zwaartekracht",
    "wat is een virus",
    "wat is water",
    "wat is de zon",
    "wat is een taal",
    "wat is energie",
    // categorie "begrip" — concepten, id 15-20
    "wat is leren",
    "wat is geheugen",
    "wat is een voorspelling",
    "wat is entropie",
    "wat is bewustzijn",
    "wat is wetenschap",
    // categorie "retrieval" — gegrond in wat ze gelezen heeft, id 21-25
    "wat heb je geleerd over natuurkunde",
    "wat heb je geleerd over het universum",
    "wat heb je gelezen over biologie",
    "welke bronnen gebruik je",
    "wat weet je over de aarde",
];

/// Categorie per probe (parallel aan PROBE_QUESTIONS). Voor per-categorie deelscores
/// in /capability, zodat zichtbaar is WAAR ze groeit of stilstaat.
pub const PROBE_CATEGORY: [&str; 26] = [
    "zelf", "zelf", "zelf", "zelf", "zelf",
    "feit", "feit", "feit", "feit", "feit", "feit", "feit", "feit", "feit", "feit",
    "begrip", "begrip", "begrip", "begrip", "begrip", "begrip",
    "retrieval", "retrieval", "retrieval", "retrieval", "retrieval",
];

/// Verwachte kernwoorden per probe — een BEWUST ZWAKKE maar stabiele correctheids-proxy.
/// Substring-match (dus "leer" matcht "leren"/"geleerd"). Stabiel gehouden zodat we
/// longitudinaal kunnen vergelijken; dit meet of het antwoord relevante concepten bevat,
/// niet algemene intelligentie. Zie ook de eerlijke beperking in `score_probe`.
pub const PROBE_EXPECTED: [&[&str]; 26] = [
    // zelf (ONGEWIJZIGD t.o.v. v3.1 — reeks-continuiteit)
    &["lucy", "veld", "leer"],
    &["nieuwsgier", "tevreden", "frustr", "voel"],
    &["leer", "denk", "veld", "verwerk"],
    &["begrijp", "concept", "patroon"],
    &["veranderd", "geleerd", "nieuw", "groei"],
    // feit
    &["deeltje", "lading", "atoom", "negatie"],
    &["deeltje", "massa", "veld", "cern"],
    &["gen", "cel", "erfelijk", "molecu"],
    &["gas", "licht", "fusie", "zon", "kern"],
    &["kracht", "massa", "aantrek", "val"],
    &["cel", "besmet", "ziek", "replic", "rna"],
    &["molecu", "vloei", "zuurstof", "waterstof"],
    &["ster", "licht", "energie", "warm"],
    &["woord", "communic", "spreek", "teken"],
    &["arbeid", "kracht", "warmte", "behoud", "joule"],
    // begrip
    &["kennis", "ervaring", "verander", "patro", "oefen"],
    &["onthoud", "opslag", "herinner", "informatie"],
    &["toekomst", "verwacht", "model", "schat"],
    &["wanorde", "thermo", "informatie", "toeval"],
    &["ervaring", "zelf", "waarneem", "denk"],
    &["onderzoek", "experiment", "kennis", "hypothese", "bewijs"],
    // retrieval
    &["deeltje", "energie", "kracht", "quantum", "fysica", "natuurkunde"],
    &["ster", "planeet", "melkweg", "heelal", "universum"],
    &["cel", "leven", "organisme", "dna", "evolutie"],
    &["wikipedia", "cern", "arxiv", "web", "pagina", "bron"],
    &["planeet", "water", "leven", "zon", "rond"],
];

/// Composietscore 0..1 voor één probe: correctheid (kernwoord-overlap) + coherentie +
/// retrieval-similariteit + niet-fallback. Bewust een zwakke bekwaamheids-proxy: het detecteert
/// vooruitgang/regressie in zelf-consistentie en gegronde antwoorden over tijd, geen algemene
/// intelligentie. Goedkoop (string-vergelijking) zodat het de tick-thread niet ophoudt.
pub fn score_probe(question_id: usize, result: &QueryResult) -> f32 {
    let expected = PROBE_EXPECTED.get(question_id).copied().unwrap_or(&[]);
    let ans = result.answer.to_lowercase();
    let correctness = if expected.is_empty() {
        0.0
    } else {
        let hits = expected.iter().filter(|kw| ans.contains(**kw)).count();
        hits as f32 / expected.len() as f32
    };
    let coherence = result.coherence.clamp(0.0, 1.0);
    let sim = result.retrieval_top3_mean_sim.clamp(0.0, 1.0);
    let non_fallback = if result.provenance == query::PROV_FALLBACK { 0.0 } else { 1.0 };
    (0.30 * correctness + 0.30 * coherence + 0.25 * sim + 0.15 * non_fallback).clamp(0.0, 1.0)
}

/// Voer één meetvraag uit in audit-only modus. Géén side-effects.
/// Returns (response_preview, provenance_label, mean_sim, fallback, used_retrieval_count, response_len).
pub fn run_probe(
    field: &mut Field,
    vocab: &Vocabulary,
    retrieval: &RetrievalStore,
    question: &str,
) -> QueryResult {
    // Facts ophalen met kinds — via readonly retrieval (geen hits-update)
    let q_embs = encoding::encode_text_readonly(vocab, question);
    let facts_with_kinds = if !q_embs.is_empty() {
        let mut q_emb = [0.0f32; DIM];
        for e in &q_embs {
            for d in 0..DIM { q_emb[d] += e[d]; }
        }
        let n = q_embs.len() as f32;
        for d in 0..DIM { q_emb[d] /= n; }
        retrieval.retrieve_with_kinds_readonly(&q_emb, 3)
    } else {
        Vec::new()
    };
    let facts: Vec<String> = facts_with_kinds.iter().map(|(_, t, _)| t.clone()).collect();

    let cancel = AtomicBool::new(false);
    // audit_only=true → geen micro_ep, geen learning
    query::process_query_full(field, vocab, question, &facts, &facts_with_kinds, &cancel, true)
}

/// Serialiseer één probe-resultaat naar JSON-lijn voor appendlog.
pub fn format_jsonl_row(
    tick: u64,
    question_id: usize,
    question: &str,
    result: &QueryResult,
    res: &ResourceSnapshot,
) -> String {
    let prov_label = match result.provenance {
        query::PROV_FALLBACK => "fallback",
        query::PROV_FIELD_GENERATED => "field_generated",
        query::PROV_RETRIEVAL_SELF => "retrieval_self",
        query::PROV_RETRIEVAL_TEACHER => "retrieval_teacher",
        query::PROV_RETRIEVAL_EXTERNAL => "retrieval_external",
        query::PROV_MIXED => "mixed",
        _ => "unknown",
    };
    let preview: String = result.answer.chars().take(120).collect();
    let score = score_probe(question_id, result);
    serde_json::json!({
        "tick": tick,
        "question_id": question_id,
        "question": question,
        "provenance": prov_label,
        "provenance_confidence": result.provenance_confidence,
        "retrieval_top3_mean_sim": result.retrieval_top3_mean_sim,
        "retrieval_top3_kind_mix": result.retrieval_top3_kind_mix,
        "fallback": result.provenance == query::PROV_FALLBACK,
        "used_retrieval_count": result.facts_retrieved,
        "response_len": result.answer.len(),
        "response_preview": preview,
        "coherence": result.coherence,
        "think_rounds": result.think_rounds,
        // v3.1 capaciteits-score (0..1) — zie score_probe voor de samenstelling
        "score": score,
        // v3.1.6: categorie voor deelscores; oude rijen zonder dit veld zijn "zelf" (id 0-4)
        "category": PROBE_CATEGORY.get(question_id).copied().unwrap_or("zelf"),
        // fase-1 meet-fundament: resource-context zodat efficientie (resultaat per resource)
        // berekenbaar wordt. Alle rijen in dezelfde batch delen deze waarden.
        "tokens_processed": res.tokens_processed,
        "pages_read": res.pages_read,
        "n_points": res.n_points,
        "n_couplings": res.n_couplings,
        "vocab_size": res.vocab_size,
        "avg_free_energy": res.avg_free_energy,
        "synthetic_audit": true,
    }).to_string()
}

/// Append een rij naar data/self_test.jsonl. Fouten zijn niet-fataal.
pub fn append_jsonl(row: &str) {
    use std::io::Write;
    let path = std::path::Path::new("data/self_test.jsonl");
    // Zorg dat data/ bestaat
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::OpenOptions::new().create(true).append(true).open(path) {
        Ok(mut f) => {
            let _ = writeln!(f, "{}", row);
        }
        Err(e) => {
            tracing::warn!(error = %e, "self_test: kon jsonl niet appenden");
        }
    }
}

/// Run de volledige batch van 5 probe-vragen en schrijf naar jsonl.
/// Niet-blokkerend: gebruikt een clone van field voor audit (net als /ask).
pub fn run_batch_audit_only(
    field: &Field,
    vocab: &Vocabulary,
    retrieval: &RetrievalStore,
) -> Vec<(usize, String)> {
    let mut field_copy = field.clone();
    let res = ResourceSnapshot::capture(field, vocab);
    let mut results = Vec::with_capacity(PROBE_QUESTIONS.len());
    for (i, q) in PROBE_QUESTIONS.iter().enumerate() {
        let result = run_probe(&mut field_copy, vocab, retrieval, q);
        let row = format_jsonl_row(field.tick, i, q, &result, &res);
        append_jsonl(&row);
        results.push((i, row));
    }
    tracing::info!(
        tick = field.tick,
        count = results.len(),
        "self_test: batch audit-only afgerond"
    );
    results
}

/// Resource-momentopname bij een meet-batch: koppelt kwaliteit aan verbruikte middelen,
/// zodat efficientie (resultaat per resource) berekenbaar wordt. Puur meten, geen effect.
#[derive(Clone, Copy)]
pub struct ResourceSnapshot {
    pub tokens_processed: u64,
    pub pages_read: u64,
    pub n_points: u64,
    pub n_couplings: u64,
    pub vocab_size: u64,
    pub avg_free_energy: f32,
}

impl ResourceSnapshot {
    pub fn capture(field: &Field, vocab: &Vocabulary) -> Self {
        let n_couplings: usize = field.points.iter().map(|p| p.couplings.len()).sum();
        Self {
            tokens_processed: field.tokens_processed,
            pages_read: field.pages_read,
            n_points: field.points.len() as u64,
            n_couplings: n_couplings as u64,
            vocab_size: vocab.vocab_size() as u64,
            avg_free_energy: field.avg_free_energy,
        }
    }
}

/// OLS-helling van een reeks (x = index). Trend-indicator.
fn slope(ys: &[f64]) -> f64 {
    let n = ys.len();
    if n < 2 { return 0.0; }
    let nf = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();
    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-9 { return 0.0; }
    (nf * sum_xy - sum_x * sum_y) / denom
}

fn trend_label(s: f64) -> &'static str {
    if s > 1e-6 { "stijgend" } else if s < -1e-6 { "dalend" } else { "vlak" }
}

/// Reduceer data/self_test.jsonl tot KWALITEIT (mean_score + trend) EN EFFICIENTIE
/// (capability-winst per verbruikte resource + trend). Zo worden "wordt Lucy beter?" en
/// "haalt ze meer uit minder?" meetbare getallen. Puur lezen, geen side-effects.
/// Dit is het meet-fundament waarop gated zelf-verbetering (latere fase) bouwt.
pub fn capability_report(content: &str) -> serde_json::Value {
    struct Batch {
        tick: u64, n: u32, score_sum: f64, fallback: u32, self_ret: u32,
        tokens: u64, pages: u64, points: u64, couplings: u64, vocab: u64, free_energy: f64,
        // v3.1.6: per-categorie (som, n) — oude rijen zonder category-veld zijn "zelf" (id 0-4)
        cats: std::collections::BTreeMap<String, (f64, u32)>,
    }
    let mut batches: Vec<Batch> = Vec::new();
    for line in content.lines() {
        let v: serde_json::Value = match serde_json::from_str(line) { Ok(v) => v, Err(_) => continue };
        let tick = v.get("tick").and_then(|x| x.as_u64()).unwrap_or(0);
        let score = v.get("score").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let fallback = v.get("fallback").and_then(|x| x.as_bool()).unwrap_or(false);
        let prov = v.get("provenance").and_then(|x| x.as_str()).unwrap_or("");
        let category = v.get("category").and_then(|x| x.as_str()).unwrap_or("zelf");
        let same = batches.last().map(|b| b.tick == tick).unwrap_or(false);
        if !same {
            batches.push(Batch {
                tick, n: 0, score_sum: 0.0, fallback: 0, self_ret: 0,
                tokens: v.get("tokens_processed").and_then(|x| x.as_u64()).unwrap_or(0),
                pages: v.get("pages_read").and_then(|x| x.as_u64()).unwrap_or(0),
                points: v.get("n_points").and_then(|x| x.as_u64()).unwrap_or(0),
                couplings: v.get("n_couplings").and_then(|x| x.as_u64()).unwrap_or(0),
                vocab: v.get("vocab_size").and_then(|x| x.as_u64()).unwrap_or(0),
                free_energy: v.get("avg_free_energy").and_then(|x| x.as_f64()).unwrap_or(0.0),
                cats: std::collections::BTreeMap::new(),
            });
        }
        let b = batches.last_mut().unwrap();
        b.n += 1;
        b.score_sum += score;
        if fallback { b.fallback += 1; }
        if prov == "retrieval_self" { b.self_ret += 1; }
        let e = b.cats.entry(category.to_string()).or_insert((0.0, 0));
        e.0 += score;
        e.1 += 1;
    }

    let mut series: Vec<serde_json::Value> = Vec::with_capacity(batches.len());
    let mut eff_tokens_series: Vec<f64> = Vec::new();
    let mut eff_ticks_series: Vec<f64> = Vec::new();
    for i in 0..batches.len() {
        let b = &batches[i];
        let n = b.n.max(1) as f64;
        let mean_score = b.score_sum / n;
        let mut eff_tokens: Option<f64> = None;
        let mut eff_ticks: Option<f64> = None;
        if i > 0 {
            let p = &batches[i - 1];
            let d_score = mean_score - (p.score_sum / p.n.max(1) as f64);
            let d_tokens = b.tokens.saturating_sub(p.tokens);
            let d_ticks = b.tick.saturating_sub(p.tick);
            if d_tokens > 0 {
                let e = d_score / d_tokens as f64 * 1000.0; // capability-winst per 1k tokens
                eff_tokens = Some(e);
                eff_tokens_series.push(e);
            }
            if d_ticks > 0 {
                let e = d_score / d_ticks as f64 * 1000.0; // capability-winst per 1k ticks
                eff_ticks = Some(e);
                eff_ticks_series.push(e);
            }
        }
        let by_category: serde_json::Map<String, serde_json::Value> = b.cats.iter()
            .map(|(c, (sum, n))| (c.clone(), serde_json::json!(sum / (*n).max(1) as f64)))
            .collect();
        series.push(serde_json::json!({
            "tick": b.tick,
            "mean_score": mean_score,
            "by_category": by_category,
            "fallback_rate": b.fallback as f64 / n,
            "retrieval_self_rate": b.self_ret as f64 / n,
            "n": b.n,
            "tokens_processed": b.tokens,
            "pages_read": b.pages,
            "n_points": b.points,
            "n_couplings": b.couplings,
            "vocab_size": b.vocab,
            "avg_free_energy": b.free_energy,
            "eff_score_per_1k_tokens": eff_tokens,
            "eff_score_per_1k_ticks": eff_ticks,
        }));
    }

    let score_ys: Vec<f64> = batches.iter().map(|b| b.score_sum / b.n.max(1) as f64).collect();
    let score_slope = slope(&score_ys);

    // v3.1.6: trend per categorie (alleen over batches waarin de categorie voorkomt).
    // "zelf" loopt longitudinaal door over de meter-upgrade heen; nieuwe categorieen
    // starten hun eigen reeks. Zo is plateau-vs-groei per dimensie te onderscheiden.
    let mut cat_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for b in &batches { for c in b.cats.keys() { cat_names.insert(c.clone()); } }
    let category_trends: serde_json::Map<String, serde_json::Value> = cat_names.iter().map(|c| {
        let ys: Vec<f64> = batches.iter()
            .filter_map(|b| b.cats.get(c).map(|(sum, n)| sum / (*n).max(1) as f64))
            .collect();
        let s = slope(&ys);
        (c.clone(), serde_json::json!({
            "batches": ys.len(),
            "latest_mean": ys.last().copied(),
            "slope_per_batch": s,
            "trend": trend_label(s),
        }))
    }).collect();
    let eff_slope = slope(&eff_tokens_series);
    let latest = series.last().cloned().unwrap_or(serde_json::Value::Null);
    let tail: Vec<serde_json::Value> = series.iter().rev().take(240).cloned().collect::<Vec<_>>()
        .into_iter().rev().collect();

    serde_json::json!({
        "batches_total": batches.len(),
        "quality": {
            "score_slope_per_batch": score_slope,
            "trend": trend_label(score_slope),
            "by_category": category_trends,
        },
        "efficiency": {
            "note": "capability-score per verbruikte resource; slope>0 = meer resultaat uit minder over tijd",
            "latest_score_per_1k_tokens": eff_tokens_series.last().copied(),
            "latest_score_per_1k_ticks": eff_ticks_series.last().copied(),
            "efficiency_slope": eff_slope,
            "efficiency_trend": trend_label(eff_slope),
        },
        "latest": latest,
        "series": tail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_report_meet_efficientie() {
        // Twee batches, elk 1 probe: score 0.4 -> 0.6 (d=0.2) over 10000 tokens (d=10000).
        // Verwachte efficientie = 0.2 / 10000 * 1000 = 0.02 capability per 1k tokens.
        let jsonl = concat!(
            r#"{"tick":5000,"score":0.4,"fallback":false,"provenance":"field_generated","tokens_processed":10000,"pages_read":5,"n_points":8000,"n_couplings":100000,"vocab_size":3000,"avg_free_energy":100.0}"#, "\n",
            r#"{"tick":10000,"score":0.6,"fallback":false,"provenance":"field_generated","tokens_processed":20000,"pages_read":10,"n_points":8100,"n_couplings":110000,"vocab_size":3200,"avg_free_energy":90.0}"#, "\n",
        );
        let rep = capability_report(jsonl);
        assert_eq!(rep["batches_total"], 2);
        let eff = rep["efficiency"]["latest_score_per_1k_tokens"].as_f64().unwrap();
        assert!((eff - 0.02).abs() < 1e-9, "eff={}", eff);
        assert!(rep["quality"]["score_slope_per_batch"].as_f64().unwrap() > 0.0, "kwaliteit hoort te stijgen");
        assert_eq!(rep["efficiency"]["efficiency_trend"], "vlak"); // slechts 1 efficientie-punt -> geen trend
    }

    #[test]
    fn capability_report_leeg() {
        let rep = capability_report("");
        assert_eq!(rep["batches_total"], 0);
    }

    #[test]
    fn probe_arrays_consistent() {
        assert_eq!(PROBE_QUESTIONS.len(), PROBE_EXPECTED.len());
        assert_eq!(PROBE_QUESTIONS.len(), PROBE_CATEGORY.len());
        // origineel 5 ongewijzigd op id 0-4 (reeks-continuiteit)
        assert_eq!(PROBE_QUESTIONS[0], "wie ben je");
        assert_eq!(PROBE_CATEGORY[0], "zelf");
        assert_eq!(PROBE_EXPECTED[0], &["lucy", "veld", "leer"]);
        // elke probe heeft verwachtingen en een categorie
        for i in 0..PROBE_QUESTIONS.len() {
            assert!(!PROBE_EXPECTED[i].is_empty(), "probe {} zonder expected", i);
            assert!(["zelf", "feit", "begrip", "retrieval"].contains(&PROBE_CATEGORY[i]));
        }
    }

    #[test]
    fn capability_report_per_categorie() {
        // Batch met 2 categorieen; oude rij ZONDER category-veld telt als "zelf".
        let jsonl = concat!(
            r#"{"tick":5000,"score":0.4,"fallback":false,"provenance":"field_generated","category":"feit","tokens_processed":10000}"#, "\n",
            r#"{"tick":5000,"score":0.8,"fallback":false,"provenance":"field_generated","category":"feit","tokens_processed":10000}"#, "\n",
            r#"{"tick":5000,"score":0.5,"fallback":false,"provenance":"field_generated","tokens_processed":10000}"#, "\n",
        );
        let rep = capability_report(jsonl);
        assert_eq!(rep["batches_total"], 1);
        let feit = rep["series"][0]["by_category"]["feit"].as_f64().unwrap();
        assert!((feit - 0.6).abs() < 1e-9, "feit-mean={}", feit);
        let zelf = rep["series"][0]["by_category"]["zelf"].as_f64().unwrap();
        assert!((zelf - 0.5).abs() < 1e-9, "zelf-mean={}", zelf);
        assert_eq!(rep["quality"]["by_category"]["feit"]["batches"], 1);
    }
}
