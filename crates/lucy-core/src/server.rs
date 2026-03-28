//! # server — Lucy's HTTP API
//!
//! De interface waarmee de buitenwereld met Lucy communiceert.
//! Draait op port 3000 via Axum (async, performant).
//!
//! ## Endpoints
//!
//! | Route | Methode | Functie |
//! |-------|---------|---------|
//! | `/` | GET | Versie-info en beschikbare endpoints |
//! | `/status` | GET | Live veld-statistieken (punten, spanning, vocab, explorer) |
//! | `/ask` | POST | Stel een vraag — veld berekent het antwoord |
//! | `/feed` | POST | Geef Lucy trainingsdata (JSON of platte tekst) |
//!
//! ## /ask — Vraagverwerking
//!
//! Body: platte tekst (de vraag).
//! Lucy encodeert de vraag → injecteert in het veld → laat settlen → decodeert output.
//! Elke vraag verrijkt ook Lucy's explorer: URLs en interessante woorden worden geabsorbeerd.
//!
//! Response: `{ answer, source: "field", settle_ticks, tension, tokens_generated }`
//!
//! ## /feed — Training
//!
//! Twee formaten:
//! 1. **JSON**: `{ "input": "vraag", "target": "antwoord" }` → equilibrium propagation training
//! 2. **Platte tekst**: → sequential prediction (tekst in twee helften splitsen)
//!
//! Beide formaten slaan het voorbeeld op in de replay buffer voor consolidatie.
//! URLs en woorden uit de tekst worden geabsorbeerd voor exploratie.
//!
//! ## Gedeelde State
//!
//! Alle state is thread-safe via ``Arc<RwLock<T>>``:
//! - `field` — het samenveld (write lock voor berekeningen)
//! - `vocab` — vocabulaire (write lock voor nieuwe tokens)
//! - `explorer` — URL frontier en interesses (write lock voor ontdekking)
//! - `replay` — consolidatie buffer (write lock voor opslag)

use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use lucy_state::*;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::encoding::Vocabulary;
use crate::consolidation::ReplayBuffer;
use crate::learner::Explorer;
use crate::query;

/// Gedeelde state — thread-safe via `Arc<RwLock<T>>`
#[derive(Clone)]
pub struct AppState {
    pub field: Arc<RwLock<Field>>,
    pub vocab: Arc<RwLock<Vocabulary>>,
    pub explorer: Arc<RwLock<Explorer>>,
    pub replay: Arc<RwLock<ReplayBuffer>>,
}

pub type SharedState = Arc<AppState>;

pub fn spawn_server(state: SharedState) {
    tokio::spawn(async move {
        let app = Router::new()
            .route("/", get(root))
            .route("/status", get(status))
            .route("/ask", post(ask_handler))
            .route("/feed", post(feed_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
            .await
            .expect("Failed to bind port 3000");

        tracing::info!("API server listening on port 3000");
        axum::serve(listener, app).await.ok();
    });
}

async fn root() -> Json<Value> {
    Json(json!({
        "name": "Lucy",
        "version": "0.5.0",
        "what": "Zelflerend veld-systeem — echte AI, geen lookup cache",
        "how": "Equilibrium propagation op een small-world netwerk",
        "endpoints": {
            "/status": "GET — live status",
            "/ask": "POST — stel een vraag (veld berekent het antwoord)",
            "/feed": "POST — geef Lucy trainingsdata"
        }
    }))
}

async fn status(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field.read().await;
    let v = state.vocab.read().await;
    let e = state.explorer.read().await;
    let r = state.replay.read().await;

    let n = f.active_set.len() as f32;
    let avg_tension = if n > 0.0 {
        f.active_set.iter().map(|&id| f.points[id as usize].tension).sum::<f32>() / n
    } else { 0.0 };

    let n_couplings: usize = f.points.iter().map(|p| p.couplings.len()).sum();

    Json(json!({
        "alive": true,
        "tick": f.tick,
        "phase": format!("{:?}", f.phase),
        "field": {
            "points": f.points.len(),
            "active": f.active_set.len(),
            "input_ports": f.input_ids.len(),
            "output_ports": f.output_ids.len(),
            "couplings": n_couplings,
            "avg_tension": format!("{:.6}", avg_tension),
        },
        "vocabulary": {
            "size": v.vocab_size(),
        },
        "explorer": {
            "pages_read": e.pages_read,
            "tokens_processed": e.tokens_processed,
            "frontier_size": e.frontier.len(),
        },
        "replay_buffer": r.len(),
        "balance": format!("${:.2}", f.economy.balance_cents as f64 / 100.0),
    }))
}

/// Het hart: /ask — Lucy beantwoordt vragen met haar veld
#[axum::debug_handler]
async fn ask_handler(
    State(state): State<SharedState>,
    body: String,
) -> Json<Value> {
    let question = body.trim().to_string();
    if question.is_empty() {
        return Json(json!({"error": "Lege vraag"}));
    }

    // Lucy leert van elke interactie: URLs en interessante woorden
    {
        let mut explorer = state.explorer.write().await;
        explorer.discover_urls(&question);
        explorer.absorb_interests(&question);
    }

    let mut field = state.field.write().await;
    let vocab = state.vocab.read().await;

    let result = query::process_query(&mut field, &vocab, &question);

    if result.answer.is_empty() {
        return Json(json!({
            "answer": "...",
            "source": "field",
            "note": "Lucy denkt nog na — ze heeft meer training nodig",
            "settle_ticks": result.settle_ticks,
            "tension": format!("{:.4}", result.final_tension),
        }));
    }

    tracing::info!(
        tokens = result.tokens_generated,
        ticks = result.settle_ticks,
        tension = format!("{:.4}", result.final_tension),
        "Lucy answered from field dynamics"
    );

    Json(json!({
        "answer": result.answer,
        "source": "field",
        "settle_ticks": result.settle_ticks,
        "tension": format!("{:.4}", result.final_tension),
        "tokens_generated": result.tokens_generated,
    }))
}

/// /feed — geef Lucy trainingsdata, ze leert via equilibrium propagation
async fn feed_handler(
    State(state): State<SharedState>,
    body: String,
) -> Json<Value> {
    // Probeer JSON met input/target
    if let Ok(data) = serde_json::from_str::<Value>(&body) {
        if let (Some(input), Some(target)) = (data["input"].as_str(), data["target"].as_str()) {
            let mut field = state.field.write().await;
            let mut vocab = state.vocab.write().await;

            let delta = query::train_pair(&mut field, &mut vocab, input, target);

            // Sla op voor consolidation replay
            let input_embs = crate::encoding::encode_text_readonly(&vocab, input);
            let target_embs = crate::encoding::encode_text_readonly(&vocab, target);
            if !input_embs.is_empty() && !target_embs.is_empty() {
                let mut replay = state.replay.write().await;
                replay.store(input_embs, target_embs);
            }

            return Json(json!({
                "learned": true,
                "weight_delta": format!("{:.4}", delta),
                "vocab_size": vocab.vocab_size(),
            }));
        }
    }

    // Lucy leert van elke interactie: URLs en interessante woorden
    {
        let mut explorer = state.explorer.write().await;
        explorer.discover_urls(&body);
        explorer.absorb_interests(&body);
    }

    // Fallback: ruwe tekst als leesmateriaal
    let mut field = state.field.write().await;
    let mut vocab = state.vocab.write().await;

    // Tokenize en leer via sequential prediction
    let token_ids = vocab.tokenize(&body);
    let n_tokens = token_ids.len();

    // Train het veld op deze tekst
    if n_tokens >= 3 {
        let embeddings: Vec<[f32; DIM]> = token_ids.iter().enumerate().map(|(pos, &id)| {
            let mut v = *vocab.embed(id);
            // Positional encoding
            let p = pos as f32;
            for d in (0..DIM).step_by(2) {
                let freq = 1.0 / (10000.0f32).powf(d as f32 / DIM as f32);
                v[d] += 0.1 * (p * freq).sin();
                if d + 1 < DIM { v[d + 1] += 0.1 * (p * freq).cos(); }
            }
            v
        }).collect();

        let mid = embeddings.len() / 2;
        let input_embs = &embeddings[..mid];
        let target_embs = &embeddings[mid..];

        let delta = crate::compute::equilibrium_propagation(
            &mut field,
            input_embs,
            target_embs,
            5,
            100,
            0.3,
        );

        // Sla op voor consolidation replay
        {
            let mut replay = state.replay.write().await;
            replay.store(input_embs.to_vec(), target_embs.to_vec());
        }

        return Json(json!({
            "learned": true,
            "fed_as": "sequential_prediction",
            "tokens": n_tokens,
            "weight_delta": format!("{:.4}", delta),
            "vocab_size": vocab.vocab_size(),
        }));
    }

    Json(json!({
        "learned": false,
        "reason": "tekst te kort",
    }))
}
