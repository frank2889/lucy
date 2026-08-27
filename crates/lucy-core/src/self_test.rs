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

/// Vaste meetvragen — géén trainingsdata, alleen probe.
/// Deze lijst is intentioneel klein en stabiel: we willen longitudinaal vergelijken.
pub const PROBE_QUESTIONS: [&str; 5] = [
    "wie ben je",
    "wat voel je nu",
    "wat doe je nu",
    "wat probeer je te begrijpen",
    "wat is recent veranderd in jezelf",
];

/// Verwachte kernwoorden per probe — een BEWUST ZWAKKE maar stabiele correctheids-proxy.
/// Substring-match (dus "leer" matcht "leren"/"geleerd"). Klein en stabiel gehouden zodat we
/// longitudinaal kunnen vergelijken; dit meet of het antwoord relevante concepten bevat,
/// niet algemene intelligentie. Zie ook de eerlijke beperking in `score_probe`.
pub const PROBE_EXPECTED: [&[&str]; 5] = [
    &["lucy", "veld", "leer"],
    &["nieuwsgier", "tevreden", "frustr", "voel"],
    &["leer", "denk", "veld", "verwerk"],
    &["begrijp", "concept", "patroon"],
    &["veranderd", "geleerd", "nieuw", "groei"],
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
    let mut results = Vec::with_capacity(PROBE_QUESTIONS.len());
    for (i, q) in PROBE_QUESTIONS.iter().enumerate() {
        let result = run_probe(&mut field_copy, vocab, retrieval, q);
        let row = format_jsonl_row(field.tick, i, q, &result);
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
