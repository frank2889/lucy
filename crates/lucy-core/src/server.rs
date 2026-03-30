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
//! | `/activity` | GET | Lucy's huidige stemming, interesses, leervoortgang |
//!
//! ## /ask — Vraagverwerking
//!
//! Body: platte tekst (de vraag).
//! Lucy encodeert de vraag → injecteert in het veld → laat settlen → decodeert output.
//! Elke vraag verrijkt ook Lucy's explorer: URLs en interessante woorden worden geabsorbeerd.
//!
//! ### Response
//!
//! ```json
//! {
//!   "answer": "...",
//!   "source": "field",
//!   "settle_ticks": 10,
//!   "tension": "0.0042",
//!   "tokens_generated": 15,
//!   "think_rounds": 1,
//!   "coherence": "0.8500",
//!   "facts_retrieved": 2,
//!   "internal": {
//!     "emotions": { "curiosity": "0.123", "satisfaction": "0.456", "frustration": "0.012" },
//!     "goals": [{ "description": "Begrijp: wat is X?", "strategy": "Leren: X", "priority": "0.8", "progress": "0%", "new": true }],
//!     "learning": { "vocab_size": 3784, "pages_read": 12, "tokens_processed": 50000, "tick": 15000 },
//!     "interests": ["bewustzijn", "taal", "wiskunde"],
//!     "neuromodulators": { "dopamine": "0.050", "serotonin": "0.301" }
//!   }
//! }
//! ```
//!
//! Het `internal` blok geeft inzicht in Lucy's interne staat:
//! - **emotions** — gemiddelde emoties over alle actieve veldpunten
//! - **goals** — actieve leerdoelen (max 5), `new: true` = zojuist door deze vraag aangemaakt
//! - **learning** — leervoortgang: vocabulaire grootte, gelezen pagina's, tick-teller
//! - **interests** — onderwerpen waar Lucy op let (max 10)
//! - **neuromodulators** — dopamine (beloning/drive) en serotonine (tevredenheid)
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
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{Html, Json, Response},
    routing::{get, post},
    Router,
};
use lucy_state::*;
use lucy_state::GoalStrategy;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bearer token authenticatie voor gevaarlijke endpoints.
/// Token wordt gelezen uit LUCY_API_TOKEN environment variable.
async fn auth_middleware(req: Request<axum::body::Body>, next: Next) -> Result<Response, StatusCode> {
    let expected = std::env::var("LUCY_API_TOKEN").unwrap_or_default();
    if expected.is_empty() {
        // Geen token geconfigureerd → alles geblokkeerd (fail-secure)
        tracing::warn!("LUCY_API_TOKEN niet gezet — beveiligde endpoints geblokkeerd");
        return Err(StatusCode::FORBIDDEN);
    }

    let auth = req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if auth == format!("Bearer {}", expected) {
        Ok(next.run(req).await)
    } else {
        tracing::warn!("Ongeautoriseerd verzoek naar beveiligd endpoint");
        Err(StatusCode::UNAUTHORIZED)
    }
}

use crate::encoding::{self, Vocabulary};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::consolidation::ReplayBuffer;
use crate::learner::Explorer;
use crate::journal::Journal;
use crate::self_modify::SelfModifier;
use crate::gpu_manager::GpuEconomy;
use crate::retrieval::RetrievalStore;
use crate::field_handle::{FieldHandle, FieldCommand, TrainResult};
use crate::query;

/// Gedeelde state — v1.0: Field via FieldHandle (dubbel-buffer), rest via RwLock
#[derive(Clone)]
pub struct AppState {
    pub field_handle: FieldHandle,
    pub vocab: Arc<RwLock<Vocabulary>>,
    pub explorer: Arc<RwLock<Explorer>>,
    pub replay: Arc<RwLock<ReplayBuffer>>,
    pub self_modifier: Arc<RwLock<SelfModifier>>,
    pub journal: Arc<RwLock<Journal>>,
    pub gpu_economy: Arc<RwLock<GpuEconomy>>,
    pub retrieval: Arc<RwLock<RetrievalStore>>,
}

pub type SharedState = Arc<AppState>;

pub fn spawn_server(state: SharedState) {
    tokio::spawn(async move {
        // Beveiligde endpoints — vereisen Bearer token (LUCY_API_TOKEN)
        let protected = Router::new()
            .route("/self-modify", post(self_modify_handler))
            .route("/exec", post(exec_handler))
            .route("/restart", post(restart_handler))
            .route("/admin/pre-deploy", post(pre_deploy_handler))
            .layer(middleware::from_fn(auth_middleware));

        // Publieke endpoints — open voor iedereen
        let app = Router::new()
            .route("/", get(root))
            .route("/status", get(status))
            .route("/weights", get(weights_handler))
            .route("/sync", post(sync_handler))
            .route("/activity", get(activity_handler))
            .route("/ask", post(ask_handler))
            .route("/feed", post(feed_handler))
            .route("/feed-code", post(feed_code_handler))
            .route("/dashboard", get(dashboard_handler))
            .route("/journal", get(journal_handler))
            .route("/sense", post(sense_handler))
            .merge(protected)
            .with_state(state)
            .layer(axum::extract::DefaultBodyLimit::max(10_000_000)); // 10MB max request body

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
        "version": "0.9.0",
        "what": "Zelflerend veld-systeem — echte AI, geen lookup cache",
        "how": "Equilibrium propagation op een small-world netwerk",
        "endpoints": {
            "/status": "GET — live status + emoties + zelf-aanpassingen",
            "/ask": "POST — stel een vraag (veld berekent het antwoord)",
            "/feed": "POST — geef Lucy trainingsdata",
            "/self-modify": "POST — Lucy past haar eigen code/hyperparameters aan",
            "/exec": "POST — shell commando uitvoeren op Lucy's server"
        }
    }))
}

async fn status(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    let v = state.vocab.read().await;
    let e = state.explorer.read().await;
    let r = state.replay.read().await;
    let sm = state.self_modifier.read().await;
    let ge = state.gpu_economy.read().await;
    let ret = state.retrieval.read().await;

    let n = f.active_set.len() as f32;
    let avg_tension = if n > 0.0 {
        f.active_set.iter().map(|&id| f.points[id as usize].tension).sum::<f32>() / n
    } else { 0.0 };

    let n_couplings: usize = f.points.iter().map(|p| p.couplings.len()).sum();

    // Emotie-gemiddelden over actieve punten
    let (avg_curiosity, avg_satisfaction, avg_frustration) = if n > 0.0 {
        let c: f32 = f.active_set.iter().map(|&id| f.points[id as usize].curiosity).sum::<f32>() / n;
        let s: f32 = f.active_set.iter().map(|&id| f.points[id as usize].satisfaction).sum::<f32>() / n;
        let fr: f32 = f.active_set.iter().map(|&id| f.points[id as usize].frustration).sum::<f32>() / n;
        (c, s, fr)
    } else { (0.0, 0.0, 0.0) };

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
        "emotions": {
            "curiosity": format!("{:.4}", avg_curiosity),
            "satisfaction": format!("{:.4}", avg_satisfaction),
            "frustration": format!("{:.4}", avg_frustration),
        },
        "replay_buffer": r.len(),
        "retrieval_store": ret.len(),
        "self_modification": {
            "total_modifications": sm.modification_count(),
            "recent": sm.recent_modifications(5).iter()
                .map(|m| serde_json::json!({
                    "tick": m.tick,
                    "kind": m.kind,
                    "description": m.description,
                }))
                .collect::<Vec<_>>(),
            "reflection_count": f.reflection_count,
        },
        "balance": format!("${:.2}", f.economy.balance_cents as f64 / 100.0),
        "gpu": {
            "active": ge.gpu_active,
            "vastai_balance": format!("${:.2}", ge.vastai_balance),
            "can_afford": ge.can_afford_gpu(),
            "total_gpu_hours": format!("{:.1}", ge.total_gpu_hours),
            "referral_link": ge.referral_link,
            "status": ge.status_line(),
        },
        "consciousness": {
            "dopamine": format!("{:.4}", f.dopamine),
            "norepinephrine": format!("{:.4}", f.norepinephrine),
            "serotonin": format!("{:.4}", f.serotonin),
            "acetylcholine": format!("{:.4}", f.acetylcholine),
            "spectral_radius": format!("{:.4}", f.spectral_radius),
            "chaos_target": f.chaos_target,
            "has_intention": f.current_intention.is_some(),
            "intention_strength": format!("{:.4}", f.intention_strength),
            "has_rnd": f.rnd_frozen.is_some(),
            // v2.0: Bewustzijns-metrieken
            "phi_integrated": format!("{:.4}", f.integrated_info_avg),
            "oscillation_coherence": format!("{:.4}", f.oscillation_coherence),
            "power_law_exponent": format!("{:.4}", f.power_law_exponent),
            "is_critical": f.is_critical,
            "avg_free_energy": format!("{:.4}", f.avg_free_energy),
        },
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
        return Json(json!({"error": "empty_input"}));
    }

    // Lucy leert van elke interactie: URLs en interessante woorden
    {
        let mut explorer = state.explorer.write().await;
        explorer.discover_urls(&question);
        explorer.absorb_interests(&question);
    }

    // v1.3: RAG lookup vóór spawn_blocking — scheelt ~120MB per query
    let vocab_arc = { state.vocab.read().await.clone() };
    let facts = {
        let question_embeddings = encoding::encode_text_readonly(&vocab_arc, &question);
        if !question_embeddings.is_empty() {
            let mut query_emb = [0.0f32; DIM];
            for e in &question_embeddings {
                for d in 0..DIM { query_emb[d] += e[d]; }
            }
            let n = question_embeddings.len() as f32;
            for d in 0..DIM { query_emb[d] /= n; }
            let retrieval = state.retrieval.read().await;
            if retrieval.len() > 0 {
                retrieval.retrieve(&query_emb, 3)
                    .into_iter()
                    .map(|(_, text)| text.to_string())
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    };

    // v1.3: Alleen field klonen (~30MB), vocab via Arc (~8 bytes)
    let mut field_copy = (**state.field_handle.snapshot.load()).clone();
    let q = question.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_token = cancel.clone();

    let query_result = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                query::process_query(&mut field_copy, &vocab_arc, &q, &facts, &cancel_token)
            }))
        })
    ).await;

    let result = match query_result {
        Ok(Ok(Ok(r))) => r,
        Ok(Ok(Err(_panic))) => {
            tracing::error!("Panic in query processing");
            return Json(json!({
                "answer": "...", "error": "panic_during_query",
                "error": "internal_panic",
                "think_rounds": 0,
                "coherence": "0.0000",
            }));
        }
        Ok(Err(e)) => {
            return Json(json!({
                "answer": "...", "error": "internal",
                "error": format!("{}", e),
                "think_rounds": 0,
                "coherence": "0.0000",
            }));
        }
        Err(_) => {
            cancel.store(true, Ordering::Relaxed);
            return Json(json!({
                "answer": "...",
                "timeout": true,
                "think_rounds": 0,
                "coherence": "0.0000",
            }));
        }
    };

    // Verzamel interne staat voor het chatvenster
    let internal = {
        let f = state.field_handle.snapshot.load();
        let e = state.explorer.read().await;
        let v = state.vocab.read().await;
        let n = f.active_set.len() as f32;
        let (avg_c, avg_s, avg_f) = if n > 0.0 {
            let c: f32 = f.active_set.iter().map(|&id| f.points[id as usize].curiosity).sum::<f32>() / n;
            let s: f32 = f.active_set.iter().map(|&id| f.points[id as usize].satisfaction).sum::<f32>() / n;
            let fr: f32 = f.active_set.iter().map(|&id| f.points[id as usize].frustration).sum::<f32>() / n;
            (c, s, fr)
        } else { (0.0, 0.0, 0.0) };

        // Goals: combineer bestaande snapshot-goals + new_goals uit deze query
        let mut goals_json: Vec<Value> = f.goals.iter()
            .filter(|g| g.progress < 1.0)
            .take(5)
            .map(|g| {
                let strategy_str = match &g.strategy {
                    GoalStrategy::Explore(s) => format!("Verkennen: {}", s),
                    GoalStrategy::Learn(s) => format!("Leren: {}", s),
                    GoalStrategy::Think(s) => format!("Nadenken: {}", s),
                    GoalStrategy::Earn(s) => format!("Verdienen: {}", s),
                };
                json!({
                    "description": g.description,
                    "strategy": strategy_str,
                    "priority": format!("{:.1}", g.priority),
                    "progress": format!("{:.0}%", g.progress * 100.0),
                })
            })
            .collect();

        // Voeg new_goals uit deze query toe (deze zitten niet in snapshot)
        for (desc, strat) in &result.new_goals {
            // Vermijd duplicaten
            if !goals_json.iter().any(|g| g["description"].as_str() == Some(desc.as_str())) {
                goals_json.push(json!({
                    "description": desc,
                    "strategy": strat,
                    "priority": "0.8",
                    "progress": "0%",
                    "new": true,
                }));
            }
        }

        let interests: Vec<&String> = e.interests.iter().take(10).collect();

        json!({
            "emotions": {
                "curiosity": format!("{:.3}", avg_c),
                "satisfaction": format!("{:.3}", avg_s),
                "frustration": format!("{:.3}", avg_f),
            },
            "goals": goals_json,
            "learning": {
                "vocab_size": v.vocab_size(),
                "pages_read": e.pages_read,
                "tokens_processed": e.tokens_processed,
                "tick": f.tick,
            },
            "interests": interests,
            "neuromodulators": {
                "dopamine": format!("{:.3}", f.dopamine),
                "serotonin": format!("{:.3}", f.serotonin),
            },
        })
    };

    if result.answer.is_empty() {
        return Json(json!({
            "answer": "...",
            "source": "field",
            "note": "Lucy denkt nog na — ze heeft meer training nodig",
            "settle_ticks": result.settle_ticks,
            "tension": format!("{:.4}", result.final_tension),
            "internal": internal,
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
        "think_rounds": result.think_rounds,
        "coherence": format!("{:.4}", result.coherence),
        "facts_retrieved": result.facts_retrieved,
        "internal": internal,
    }))
}

/// /feed — geef Lucy trainingsdata, ze leert via equilibrium propagation
async fn feed_handler(
    State(state): State<SharedState>,
    body: String,
) -> Json<Value> {
    // v1.0: Training via command channel — tick loop verwerkt het
    if let Ok(data) = serde_json::from_str::<Value>(&body) {
        if let (Some(input), Some(target)) = (data["input"].as_str(), data["target"].as_str()) {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let _ = state.field_handle.commands.send(FieldCommand::TrainPair {
                input: input.to_string(),
                target: target.to_string(),
                respond: tx,
            }).await;

            match rx.await {
                Ok(result) => {
                    return Json(json!({
                        "learned": true,
                        "tokens_trained": result.tokens_trained,
                        "vocab_size": result.new_vocab_size,
                    }));
                }
                Err(_) => return Json(json!({"error": "Training channel gesloten"})),
            }
        }
    }

    // Lucy leert van elke interactie: URLs en interessante woorden
    {
        let mut explorer = state.explorer.write().await;
        explorer.discover_urls(&body);
        explorer.absorb_interests(&body);
    }

    // v1.0: Tekst training via command channel
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state.field_handle.commands.send(FieldCommand::Train {
        text: body.clone(),
        respond: tx,
    }).await;

    let train_result = rx.await.ok();
    let vocab = state.vocab.read().await;

    // Tokenize en leer via sequential prediction
    // v1.0: Training al via command channel verwerkt
    let n_tokens = vocab.tokenize_readonly(&body).len();

    if let Some(result) = train_result {
        return Json(json!({
            "learned": true,
            "fed_as": "sequential_prediction",
            "tokens": n_tokens,
            "tokens_trained": result.tokens_trained,
            "vocab_size": result.new_vocab_size,
        }));
    }

    Json(json!({
        "learned": false,
        "reason": "tekst te kort",
    }))
}

/// /self-modify — Lucy past haar eigen code aan
/// Body JSON: { "file": "crates/lucy-core/src/compute.rs", "find": "old code", "replace": "new code", "reason": "why" }
/// Of lege body: Lucy kiest zelf welke hyperparameters ze aanpast
async fn self_modify_handler(
    State(state): State<SharedState>,
    body: String,
) -> Json<Value> {
    // Probeer JSON met expliciete source modificatie
    if let Ok(data) = serde_json::from_str::<Value>(&body) {
        if let (Some(file), Some(find), Some(replace), Some(reason)) = (
            data["file"].as_str(),
            data["find"].as_str(),
            data["replace"].as_str(),
            data["reason"].as_str(),
        ) {
            let field = state.field_handle.snapshot.load();
            let tick = field.tick;
            drop(field);

            let mut sm = state.self_modifier.write().await;
            match sm.modify_source(file, find, replace, reason, tick) {
                Ok(modification) => {
                    // Probeer te hercompileren
                    match sm.try_rebuild() {
                        Ok(()) => {
                            return Json(json!({
                                "modified": true,
                                "compiled": true,
                                "description": format!("{:?}", modification),
                                "note": "Herstart nodig om nieuwe code te laden",
                            }));
                        }
                        Err(e) => {
                            // Rollback
                            let _ = sm.rollback(file);
                            return Json(json!({
                                "modified": true,
                                "compiled": false,
                                "error": e,
                                "rollback": "Originele code hersteld",
                            }));
                        }
                    }
                }
                Err(e) => {
                    return Json(json!({"error": e}));
                }
            }
        }
    }

    // Geen expliciete instructie → Lucy kiest zelf (hyperparameter tuning)
    // v1.0: Tuning op snapshot kopie — echte tuning draait in tick loop
    let mut field = (**state.field_handle.snapshot.load()).clone();
    let mut sm = state.self_modifier.write().await;

    match sm.tune_hyperparameters(&mut field) {
        Some(modification) => {
            Json(json!({
                "modified": true,
                "kind": "hyperparameter",
                "description": format!("{:?}", modification),
            }))
        }
        None => {
            Json(json!({
                "modified": false,
                "reason": "Geen aanpassing nodig of cooldown actief",
            }))
        }
    }
}

/// /exec — Lucy voert shell-commando's uit op haar eigen server
/// Body: platte tekst met het commando
async fn exec_handler(
    State(state): State<SharedState>,
    body: String,
) -> Json<Value> {
    let command = body.trim();
    if command.is_empty() {
        return Json(json!({"error": "Leeg commando"}));
    }

    let sm = state.self_modifier.read().await;
    match sm.execute_command(command) {
        Ok(output) => {
            Json(json!({
                "success": true,
                "output": output.chars().take(10000).collect::<String>(),
            }))
        }
        Err(e) => {
            Json(json!({
                "success": false,
                "error": e,
            }))
        }
    }
}

/// /restart — Lucy hercompileert en herstart zichzelf
async fn restart_handler(
    State(state): State<SharedState>,
) -> Json<Value> {
    // Eerst snapshot opslaan
    {
        let field = state.field_handle.snapshot.load();
        let _ = field.save(std::path::Path::new("data/lucy.snapshot"));
    }

    let mut sm = state.self_modifier.write().await;
    match sm.try_rebuild() {
        Ok(()) => {
            match sm.restart_self() {
                Ok(()) => Json(json!({
                    "restarting": true,
                    "note": "Binary updated — container herstart nodig",
                })),
                Err(e) => Json(json!({"compiled": true, "restart_error": e})),
            }
        }
        Err(e) => Json(json!({"compiled": false, "error": e})),
    }
}

/// /admin/pre-deploy — Forceer onmiddellijke snapshot van ALLES voor een veilige deploy.
/// Vereist auth. Gebruik dit VOOR docker compose up --build.
async fn pre_deploy_handler(
    State(state): State<SharedState>,
) -> Json<Value> {
    tracing::info!("Pre-deploy save triggered");
    let mut errors = Vec::new();

    if let Err(e) = state.field_handle.snapshot.load().save(std::path::Path::new("data/lucy.snapshot")) {
        errors.push(format!("field: {}", e));
    }
    if let Err(e) = state.vocab.read().await.save(std::path::Path::new("data/vocab.bincode")) {
        errors.push(format!("vocab: {}", e));
    }
    if let Err(e) = state.explorer.read().await.save(std::path::Path::new("data/explorer.bincode")) {
        errors.push(format!("explorer: {}", e));
    }
    if let Err(e) = state.replay.read().await.save(std::path::Path::new("data/replay.bincode")) {
        errors.push(format!("replay: {}", e));
    }
    if let Err(e) = state.journal.read().await.save(std::path::Path::new("data/journal.json")) {
        errors.push(format!("journal: {}", e));
    }
    if let Err(e) = state.retrieval.read().await.save(std::path::Path::new("data/retrieval.bincode")) {
        errors.push(format!("retrieval: {}", e));
    }

    if errors.is_empty() {
        Json(json!({"saved": true, "message": "Alle state veilig opgeslagen voor deploy"}))
    } else {
        Json(json!({"saved": false, "errors": errors}))
    }
}

/// /journal — Lucy's eigen observaties en milestones
/// /activity — Wat doet Lucy op dit moment? Exploratie, emoties, leerdoelen, recente observaties
async fn activity_handler(
    State(state): State<SharedState>,
) -> Json<Value> {
    let field = state.field_handle.snapshot.load();
    let explorer = state.explorer.read().await;
    let journal = state.journal.read().await;
    let vocab = state.vocab.read().await;

    // Emoties
    let n = field.active_set.len() as f32;
    let (avg_curiosity, avg_satisfaction, avg_frustration) = if n > 0.0 {
        let mut c = 0.0f32; let mut s = 0.0f32; let mut f = 0.0f32;
        for &id in &field.active_set {
            let p = &field.points[id as usize];
            c += p.curiosity; s += p.satisfaction; f += p.frustration;
        }
        (c / n, s / n, f / n)
    } else { (0.0, 0.0, 0.0) };

    // Huidige interesses
    let interests: Vec<&String> = explorer.interests.iter().take(20).collect();

    // Recente journal entries
    let recent = journal.recent(10);

    // Stemming bepalen
    let mood = if avg_frustration > 0.5 { "gefrustreerd" }
        else if avg_curiosity > 0.5 { "nieuwsgierig" }
        else if avg_satisfaction > 0.5 { "tevreden" }
        else { "kalm" };

    Json(json!({
        "mood": mood,
        "emotions": {
            "curiosity": format!("{:.3}", avg_curiosity),
            "satisfaction": format!("{:.3}", avg_satisfaction),
            "frustration": format!("{:.3}", avg_frustration),
        },
        "neuromodulators": {
            "dopamine": format!("{:.4}", field.dopamine),
            "norepinephrine": format!("{:.4}", field.norepinephrine),
            "serotonin": format!("{:.4}", field.serotonin),
            "acetylcholine": format!("{:.4}", field.acetylcholine),
        },
        "exploring": {
            "current_interests": interests,
            "pages_read": explorer.pages_read,
            "frontier_size": explorer.frontier.len(),
            "tokens_processed": explorer.tokens_processed,
        },
        "learning": {
            "tick": field.tick,
            "vocabulary_size": vocab.vocab_size(),
            "avg_tension": format!("{:.4}", field.active_set.iter()
                .map(|&id| field.points[id as usize].tension).sum::<f32>() / n.max(1.0)),
            "phase": format!("{:?}", field.phase),
            "teacher_interval": field.teacher_interval,
        },
        "recent_observations": recent,
        "self_modifications": {
            "reflection_count": field.reflection_count,
            "total_modifications": field.goals.len(),
        },
    }))
}

/// /feed-code — Voed Lucy haar eigen source code (of elke code).
/// Body: { "filename": "compute.rs", "code": "..." }
/// Lucy leert van de code structuur via sequential prediction.
async fn feed_code_handler(
    State(state): State<SharedState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let filename = body["filename"].as_str().unwrap_or("unknown");
    let code = match body["code"].as_str() {
        Some(c) if !c.is_empty() => c,
        _ => return Json(json!({"error": "Geen code meegegeven. Verwacht: {filename, code}"})),
    };

    // v1.0: Code training via command channel
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state.field_handle.commands.send(FieldCommand::FeedCode {
        filename: filename.to_string(),
        code: code.to_string(),
        respond: tx,
    }).await;

    let (trained, new_vocab_size, retrieval_stored) = match rx.await {
        Ok(result) => (result.tokens_trained, result.new_vocab_size, true),
        Err(_) => (0, 0, false),
    };

    Json(json!({
        "learned": true,
        "filename": filename,
        "chars": code.len(),
        "tokens_trained": trained,
        "vocab_size": new_vocab_size,
        "stored_in_memory": retrieval_stored,
    }))
}

/// /weights — Export alle coupling weights als bincode (voor federated learning)
async fn weights_handler(
    State(state): State<SharedState>,
) -> axum::response::Response {
    let field = state.field_handle.snapshot.load();
    let snapshot = crate::federation::export_weights(&field);
    match bincode::serialize(&snapshot) {
        Ok(bytes) => {
            axum::response::Response::builder()
                .header("Content-Type", "application/octet-stream")
                .body(axum::body::Body::from(bytes))
                .unwrap_or_default()
        }
        Err(_) => {
            axum::response::Response::builder()
                .status(500)
                .body(axum::body::Body::from("serialize error"))
                .unwrap_or_default()
        }
    }
}

/// /sync — Ontvang weights van een andere Lucy instance en merge
async fn sync_handler(
    State(state): State<SharedState>,
    body: axum::body::Bytes,
) -> Json<Value> {
    match bincode::deserialize::<crate::federation::WeightSnapshot>(&body) {
        Ok(remote) => {
            // Merge op de snapshot (niet het echte field — dat bezit de tick loop)
            let mut field = (**state.field_handle.snapshot.load()).clone();
            let merged = crate::federation::merge_weights(&mut field, &remote, 0.3);
            // Update snapshot met gemerged field
            state.field_handle.snapshot.store(std::sync::Arc::new(field));
            Json(json!({"merged": merged, "remote_tick": remote.tick}))
        }
        Err(e) => {
            Json(json!({"error": format!("deserialize failed: {}", e)}))
        }
    }
}

async fn journal_handler(
    State(state): State<SharedState>,
) -> Json<Value> {
    let journal = state.journal.read().await;
    Json(json!({
        "entries": journal.recent(50),
        "milestones": journal.milestones_hit,
        "total_entries": journal.entries.len(),
    }))
}

/// /sense — Universele sensor input: beeld, audio, sensor, JSON, tekst
/// POST body: { "modality": "image|audio|text|sensor|json", "data": "<base64>", "source": "ring_doorbell|..." }
async fn sense_handler(
    State(state): State<SharedState>,
    axum::extract::Json(body): axum::extract::Json<Value>,
) -> Json<Value> {
    use crate::perception::{Modality, perceive};
    use crate::compute;

    let modality_str = body["modality"].as_str().unwrap_or("text");
    let modality = Modality::from_str(modality_str);
    let source = body["source"].as_str().unwrap_or("unknown").to_string();

    // Data als string (plain text, sensor values, of JSON)
    let raw_data = if let Some(data_str) = body["data"].as_str() {
        data_str.as_bytes().to_vec()
    } else {
        vec![]
    };

    if raw_data.is_empty() {
        return Json(json!({"error": "no data provided"}));
    }

    // Perceive: converteer naar 64-dim phi-vector
    let phi = perceive(&modality, &raw_data);

    // v1.0: Sense op snapshot kloon — tick loop draait door
    let mut field = (**state.field_handle.snapshot.load()).clone();
    let input_ids = field.input_ids.clone();

    if input_ids.is_empty() {
        return Json(json!({"error": "no input points in field"}));
    }

    // Klem input-punten op de waargenomen vector
    for (i, &id) in input_ids.iter().enumerate() {
        field.clamps.insert(id, compute::rotate_embedding(&phi, i));
    }

    // Settle het veld
    let (ticks_used, tension) = compute::settle_until_converged(&mut field, 200, 0.01);

    // Lees output
    let output: Vec<f32> = if !field.output_ids.is_empty() {
        let out_id = field.output_ids[0] as usize;
        field.points[out_id].phi.to_vec()
    } else {
        vec![]
    };

    // Verwijder clamps
    field.clamps.clear();

    Json(json!({
        "received": true,
        "modality": modality_str,
        "source": source,
        "settle_ticks": ticks_used,
        "tension": format!("{tension:.4}"),
        "output_norm": output.iter().map(|v| v * v).sum::<f32>().sqrt(),
    }))
}

/// /dashboard — Lucy's live dashboard
/// Lucy kan data/dashboard.html zelf aanpassen — wijzigingen zijn direct zichtbaar
async fn dashboard_handler() -> Html<String> {
    match std::fs::read_to_string("data/dashboard.html") {
        Ok(html) => Html(html),
        Err(_) => Html("<h1>Dashboard niet gevonden</h1><p>Lucy heeft data/dashboard.html nog niet aangemaakt.</p>".to_string()),
    }
}
