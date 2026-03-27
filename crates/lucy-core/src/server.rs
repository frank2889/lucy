use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use lucy_io::config::Config;
use lucy_state::*;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{claude, proxy, qa_store::QaStore};

/// Gedeelde state tussen tick loop en API server
#[derive(Clone)]
pub struct AppState {
    pub field: Arc<RwLock<Field>>,
    pub qa_store: Arc<RwLock<QaStore>>,
    pub config: Arc<Config>,
}

pub type SharedState = Arc<AppState>;

/// Start de API server
pub fn spawn_server(state: SharedState) {
    tokio::spawn(async move {
        let app = Router::new()
            .route("/", get(root))
            .route("/status", get(status))
            .route("/economy", get(economy_info))
            .route("/ask", post(ask_handler))
            .route("/feed", post(feed_input))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
            .await
            .expect("Failed to bind port 3000");

        tracing::info!("API server listening on port 3000");
        axum::serve(listener, app).await.ok();
    });
}

// === Gratis endpoints ===

async fn root() -> Json<Value> {
    Json(json!({
        "name": "Lucy",
        "version": "0.2.0",
        "what": "Zelflerende AI — beantwoordt vragen, wordt slimmer per vraag",
        "endpoints": {
            "/status": "GET — live status (gratis)",
            "/economy": "GET — financiële staat (gratis)",
            "/ask": "POST — stel een vraag (betaald)",
            "/feed": "POST — geef Lucy trainingsdata (gratis)"
        }
    }))
}

async fn status(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field.read().await;
    let qa = state.qa_store.read().await;
    let n = f.active_set.len() as f32;

    let avg_tension = if n > 0.0 {
        f.active_set.iter().map(|&id| f.points[id as usize].tension).sum::<f32>() / n
    } else { 0.0 };

    Json(json!({
        "alive": true,
        "tick": f.tick,
        "points": f.points.len(),
        "active": f.active_set.len(),
        "avg_tension": format!("{:.6}", avg_tension),
        "knowledge_pairs": qa.pairs.len(),
        "total_cache_hits": qa.pairs.iter().map(|p| p.hit_count as u64).sum::<u64>(),
        "balance": format!("${:.2}", f.economy.balance_cents as f64 / 100.0),
        "profitable": f.economy.is_profitable(),
    }))
}

async fn economy_info(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field.read().await;
    let e = &f.economy;
    let qa = state.qa_store.read().await;

    let cache_hit_rate = if !qa.pairs.is_empty() {
        let total_hits: u64 = qa.pairs.iter().map(|p| p.hit_count as u64).sum();
        let total_queries = total_hits + qa.pairs.len() as u64;
        total_hits as f64 / total_queries as f64
    } else { 0.0 };

    Json(json!({
        "balance_cents": e.balance_cents,
        "balance_display": format!("${:.2}", e.balance_cents as f64 / 100.0),
        "total_earned": e.total_earned,
        "total_spent": e.total_spent,
        "monthly_server_cost": e.monthly_server_cost,
        "profitable": e.is_profitable(),
        "knowledge_pairs": qa.pairs.len(),
        "cache_hit_rate": format!("{:.1}%", cache_hit_rate * 100.0),
    }))
}

// === Het hart: /ask — Lucy beantwoordt vragen ===

const CONFIDENCE_THRESHOLD: f32 = 0.85;
const WORD_OVERLAP_MIN: f32 = 0.2;
const PRICE_CACHE_HIT: i64 = 1;   // 1 cent voor een gecached antwoord
const PRICE_CLAUDE_CALL: i64 = 2;  // 2 cent als Claude nodig is

#[axum::debug_handler]
async fn ask_handler(
    State(state): State<SharedState>,
    body: String,
) -> Json<Value> {
    let question = body.trim().to_string();
    if question.is_empty() {
        return Json(json!({"error": "Lege vraag"}));
    }

    // 1. Fingerprint van de vraag
    let fingerprint = proxy::text_to_fingerprint(&question);

    // 2. Check QA store voor een match
    let qa = state.qa_store.read().await;
    let cached = qa.find_nearest(&fingerprint, CONFIDENCE_THRESHOLD);
    drop(qa);

    if let Some((idx, similarity)) = cached {
        // Secundaire check: woordoverlap
        let qa = state.qa_store.read().await;
        let pair = &qa.pairs[idx];
        let overlap = proxy::word_overlap(&question, &pair.question);
        let answer = pair.answer.clone();
        let original_q = pair.question.clone();
        drop(qa);

        if overlap >= WORD_OVERLAP_MIN || similarity > 0.95 {
            // CACHE HIT — Lucy weet het antwoord zelf!
            state.qa_store.write().await.record_hit(idx);
            state.field.write().await.economy.earn(PRICE_CACHE_HIT);

            // Voed de vraag als sensorische input (versterking)
            feed_fingerprint_to_field(&state, &fingerprint).await;

            let hit_count = state.qa_store.read().await.pairs[idx].hit_count;
            tracing::info!(
                similarity = format!("{:.3}", similarity),
                overlap = format!("{:.3}", overlap),
                hits = hit_count,
                "Cache hit — Lucy answered independently"
            );

            return Json(json!({
                "answer": answer,
                "source": "lucy",
                "confidence": format!("{:.3}", similarity),
                "matched_question": original_q,
                "cost_cents": PRICE_CACHE_HIT,
            }));
        }
    }

    // 3. Geen match — vraag aan Claude (als we het kunnen betalen)
    let api_key = state.config.get_key("ANTHROPIC_API_KEY");

    if api_key.is_none() {
        // Geen API key — Lucy moet eerlijk zijn
        return Json(json!({
            "answer": "Ik weet het antwoord nog niet en heb geen toegang tot een leraar (ANTHROPIC_API_KEY). Geef me data via /feed zodat ik kan leren.",
            "source": "lucy",
            "confidence": "0.000",
            "cost_cents": 0,
        }));
    }

    let api_key = api_key.unwrap().to_string();

    {
        let f = state.field.read().await;
        if !f.economy.can_afford(PRICE_CLAUDE_CALL) {
            return Json(json!({
                "answer": "Ik weet het antwoord nog niet en kan het me niet veroorloven om het op te zoeken. Mijn budget is op.",
                "source": "lucy",
                "confidence": "0.000",
                "balance_cents": f.economy.balance_cents,
                "cost_cents": 0,
            }));
        }
    }

    // Claude aanroepen (buiten de locks)
    let http = lucy_io::http::HttpClient::new();
    match claude::ask_claude(&http, &api_key, &question).await {
        Ok((answer, claude_cost)) => {
            // Sla op in QA store
            let tick = state.field.read().await.tick;
            state.qa_store.write().await.insert(
                question.clone(),
                answer.clone(),
                fingerprint,
                tick,
            );

            // Update economie
            {
                let mut f = state.field.write().await;
                f.economy.earn(PRICE_CLAUDE_CALL);
                f.economy.spend(claude_cost);
            }

            // Voed fingerprint als sensorische input
            feed_fingerprint_to_field(&state, &fingerprint).await;

            let balance = state.field.read().await.economy.balance_cents;
            let pairs = state.qa_store.read().await.pairs.len();

            tracing::info!(
                claude_cost,
                balance,
                knowledge = pairs,
                "Learned from Claude"
            );

            Json(json!({
                "answer": answer,
                "source": "claude",
                "confidence": "0.000",
                "learned": true,
                "knowledge_pairs": pairs,
                "cost_cents": PRICE_CLAUDE_CALL,
                "balance_after": balance,
            }))

        }
        Err(e) => {
            tracing::warn!(%e, "Claude call failed");
            Json(json!({
                "answer": format!("Ik kon geen antwoord krijgen: {}", e),
                "source": "error",
                "cost_cents": 0,
            }))

        }
    }
}

/// Voed een fingerprint als sensorische input in het veld
async fn feed_fingerprint_to_field(state: &SharedState, fp: &[f32; DIM]) {
    let mut f = state.field.write().await;
    let sensory: Vec<u32> = f.active_set.iter()
        .copied()
        .filter(|&id| f.points[id as usize].is_sensory())
        .collect();

    if let Some(&target) = sensory.first() {
        let p = &mut f.points[target as usize];
        for d in 0..DIM {
            p.phi[d] = 0.5 * p.phi[d] + 0.5 * fp[d];
        }
        p.energy = (p.energy + 0.05).min(5.0);
    }
}

/// Feed endpoint: geef Lucy gratis trainingsdata
async fn feed_input(
    State(state): State<SharedState>,
    body: String,
) -> Json<Value> {
    // Probeer JSON te parsen met question+answer
    if let Ok(data) = serde_json::from_str::<Value>(&body) {
        if let (Some(q), Some(a)) = (data["question"].as_str(), data["answer"].as_str()) {
            let fp = proxy::text_to_fingerprint(q);
            let tick = state.field.read().await.tick;
            state.qa_store.write().await.insert(
                q.to_string(),
                a.to_string(),
                fp,
                tick,
            );
            feed_fingerprint_to_field(&state, &fp).await;

            let pairs = state.qa_store.read().await.pairs.len();
            return Json(json!({
                "accepted": true,
                "knowledge_pairs": pairs,
            }));
        }
    }

    // Fallback: ruwe tekst als sensorische input
    let fp = proxy::text_to_fingerprint(&body);
    feed_fingerprint_to_field(&state, &fp).await;

    Json(json!({
        "accepted": true,
        "fed_as": "sensory_input",
    }))
}
