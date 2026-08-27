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
    http::{header, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use lucy_state::*;
use lucy_state::GoalStrategy;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::epistemic_broker;

/// Cache-Control headers voor dashboard static assets.
/// Gehashte assets (in /brain/assets/) krijgen immutable caching,
/// index.html krijgt no-cache zodat de browser altijd de laatste versie laadt.
async fn cache_headers(req: Request<axum::body::Body>, next: Next) -> Response {
    let path = req.uri().path().to_owned();
    let mut resp = next.run(req).await;
    if path.starts_with("/assets/") || path.starts_with("/brain/assets/") {
        resp.headers_mut().insert(
            header::CACHE_CONTROL,
            header::HeaderValue::from_static("public, max-age=31536000, immutable"),
        );
    } else if path == "/" || path.starts_with("/brain") {
        resp.headers_mut().insert(
            header::CACHE_CONTROL,
            header::HeaderValue::from_static("no-cache"),
        );
    }
    resp
}

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
use crate::retrieval::{self, RetrievalStore};
use crate::field_handle::{FieldHandle, FieldCommand};
use crate::query;

/// Gedeelde state — v1.0: Field via FieldHandle (dubbel-buffer), rest via RwLock
pub struct AppState {
    pub field_handle: FieldHandle,
    pub vocab: Arc<RwLock<Vocabulary>>,
    pub explorer: Arc<RwLock<Explorer>>,
    pub replay: Arc<RwLock<ReplayBuffer>>,
    pub self_modifier: Arc<RwLock<SelfModifier>>,
    pub journal: Arc<RwLock<Journal>>,
    pub gpu_economy: Arc<RwLock<GpuEconomy>>,
    pub retrieval: Arc<RwLock<RetrievalStore>>,
    /// v2.6.8 P1.2/P1.3: cumulatieve counters voor provenance + self-metrics.
    /// AtomicU64 = lock-free updates vanuit /ask handler.
    pub total_queries: std::sync::atomic::AtomicU64,
    pub fallback_queries: std::sync::atomic::AtomicU64,
    pub queries_with_self_hit: std::sync::atomic::AtomicU64,
    pub provenance_counts: [std::sync::atomic::AtomicU64; 6],
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
            .route("/sync", post(sync_handler))
            .route("/feed-code", post(feed_code_handler))
            .route("/debug/toggle_priors", post(debug_toggle_priors_handler))
            .route("/admin/grow", post(admin_grow_handler))
            .route("/admin/self-fix", post(self_fix_handler))
            // v3.1: /weights exporteert het model → achter auth (was publiek: exfiltratie-risico)
            .route("/weights", get(weights_handler))
            .layer(middleware::from_fn(auth_middleware));

        // Publieke endpoints — open voor iedereen (Lucy is headless: JSON-API, geen frontend)
        let app = Router::new()
            .route("/", get(root))
            .route("/api", get(root))
            .route("/status", get(status))
            .route("/activity", get(activity_handler))
            .route("/ask", post(ask_handler))
            .route("/feed", post(feed_handler))
            .route("/journal", get(journal_handler))
            .route("/sense", post(sense_handler))
            .route("/topology", get(topology_handler))
            .route("/telemetry", get(telemetry_handler))
            // v2.6.8 bewijs-architectuur endpoints
            .route("/self_metrics", get(self_metrics_handler))
            .route("/teacher_metrics", get(teacher_metrics_handler))
            .route("/concept_audit", get(concept_audit_handler))
            .route("/self_ep_audit", get(self_ep_audit_handler))
            .route("/self_history", get(self_history_handler))
            .route("/capability", get(capability_handler))
            .route("/retrieval_metrics", get(retrieval_metrics_handler))
            .route("/plugins", get(plugins_list_handler))
            .route("/version", get(version_handler))
            // v2.6.17: Epistemic awareness + autonomy endpoints
            .route("/admin/epistemic", get(epistemic_handler))
            .route("/admin/proposals", get(proposals_handler))
            .route("/admin/physics-proposals", get(physics_proposals_handler))
            .merge(protected)
            .with_state(state)
            .layer(CorsLayer::permissive())
            .layer(middleware::from_fn(cache_headers))
            .layer(axum::extract::DefaultBodyLimit::max(10_000_000)); // 10MB max request body

        // v3.1 "overal draaien, nooit lokaal": poort ALTIJD uit PORT-env (Render/Cloud Run/
        // Fly zetten die), gebonden op 0.0.0.0 — nooit hardcoded 3000, nooit 127.0.0.1.
        // Fallback 8080 (conventie) alleen als PORT ontbreekt op een zelf-hostende node.
        let port: u16 = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080);
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::UNSPECIFIED, port))
            .await
            .unwrap_or_else(|e| panic!("Failed to bind port {}: {}", port, e));

        tracing::info!(port, "API server listening");
        axum::serve(listener, app).await.ok();
    });
}

async fn root() -> Json<Value> {
    Json(json!({
        "name": "Lucy",
        "version": "2.0.0",
        "what": "Zelflerend veld-systeem — echte AI, geen lookup cache",
        "how": "Equilibrium propagation op een small-world netwerk",
        "endpoints": {
            "/status": "GET — live status + emoties + zelf-aanpassingen",
            "/ask": "POST — stel een vraag (veld berekent het antwoord)",
            "/feed": "POST — geef Lucy trainingsdata",
            "/brain": "GET — 3D brain dashboard",
            "/telemetry": "GET — compact real-time telemetry",
            "/topology": "GET — network topology for visualization",
            "/journal": "GET — journal entries",
            "/self-modify": "POST — Lucy past haar eigen code/hyperparameters aan",
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
            // v2.4: Upgraded metrieken
            "theta_gamma_coupling": format!("{:.4}", f.theta_gamma_coupling),
            "branching_ratio": format!("{:.4}", f.branching_ratio_avg),
            "ignition_active": f.ignition_active,
            "ignition_strength": format!("{:.4}", f.ignition_strength),
            "avg_expected_free_energy": format!("{:.4}", f.avg_expected_free_energy),
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
    // v2.6.8 P1.2: retrieve_with_kinds_readonly levert kind voor provenance-classificatie
    let vocab_arc = { state.vocab.read().await.clone() };
    let (facts, facts_with_kinds) = {
        let question_embeddings = encoding::encode_text_readonly(&vocab_arc, &question);
        if !question_embeddings.is_empty() {
            let mut query_emb = [0.0f32; DIM];
            for e in &question_embeddings {
                for d in 0..DIM { query_emb[d] += e[d]; }
            }
            let n = question_embeddings.len() as f32;
            for d in 0..DIM { query_emb[d] /= n; }
            let mut retrieval = state.retrieval.write().await;
            if retrieval.len() > 0 {
                let current_tick = state.field_handle.snapshot.load().tick;
                let with_kinds = retrieval.retrieve_with_kinds(&query_emb, 3, current_tick);
                let plain: Vec<String> = with_kinds.iter().map(|(_, t, _)| t.clone()).collect();
                (plain, with_kinds)
            } else {
                (Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new())
        }
    };

    // v1.3: Alleen field klonen (~30MB), vocab via Arc (~8 bytes)
    let mut field_copy = (**state.field_handle.snapshot.load()).clone();
    let q = question.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_token = cancel.clone();
    // v2.6.7 E5: bewaar vocab_arc clone voor interactie-log na spawn_blocking
    let vocab_for_log = vocab_arc.clone();

    let facts_kinds_moved = facts_with_kinds.clone();
    let query_result = tokio::time::timeout(
        std::time::Duration::from_secs(30), // v2.1: Meer tijd voor query processing
        tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // v2.6.8 P1.2: process_query_full met retrieval-kinds voor provenance
                query::process_query_full(&mut field_copy, &vocab_arc, &q, &facts, &facts_kinds_moved, &cancel_token, false)
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

    // v2.6.7 E5: Interactie-log in retrieval store.
    // (query_embedding, response_text). Future queries dichtbij deze vraag
    // kunnen Lucy's eigen eerdere antwoorden terugvinden via cosine similarity.
    // learning_value = 0.5 (neutraal-positief, interacties zijn waardevol).
    {
        let question_embs = encoding::encode_text_readonly(&vocab_for_log, &question);
        if !question_embs.is_empty() {
            let mut q_emb = [0.0f32; DIM];
            for e in &question_embs {
                for d in 0..DIM { q_emb[d] += e[d]; }
            }
            let n = question_embs.len() as f32;
            for d in 0..DIM { q_emb[d] /= n; }
            let combined = format!("V: {}\nA: {}", question, result.answer);
            let current_tick = state.field_handle.snapshot.load().tick;
            let mut ret = state.retrieval.write().await;
            // v2.6.8 P1.1: kind=INTERACTION → primary truth voor provenance
            ret.insert_full(&q_emb, &combined, 0.5, current_tick, retrieval::KIND_INTERACTION);
        }
    }

    // v2.6.8 P1.2: provenance label string + atomic counters update
    use std::sync::atomic::Ordering::Relaxed;
    let prov_idx = result.provenance.min(5) as usize;
    state.provenance_counts[prov_idx].fetch_add(1, Relaxed);
    state.total_queries.fetch_add(1, Relaxed);
    if result.provenance == query::PROV_FALLBACK {
        state.fallback_queries.fetch_add(1, Relaxed);
    }
    // Self-hit = top-3 bevatte ≥1 self-entry
    if result.retrieval_top3_kind_mix[1] > 0 {
        state.queries_with_self_hit.fetch_add(1, Relaxed);
    }

    let prov_label = match result.provenance {
        query::PROV_FALLBACK => "fallback",
        query::PROV_FIELD_GENERATED => "field_generated",
        query::PROV_RETRIEVAL_SELF => "retrieval_self",
        query::PROV_RETRIEVAL_TEACHER => "retrieval_teacher",
        query::PROV_RETRIEVAL_EXTERNAL => "retrieval_external",
        query::PROV_MIXED => "mixed",
        _ => "unknown",
    };
    tracing::info!(
        provenance = %prov_label,
        prov_conf = result.provenance_confidence,
        mean_sim = result.retrieval_top3_mean_sim,
        kind_mix = ?result.retrieval_top3_kind_mix,
        "ask: provenance"
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
        "provenance": prov_label,
        "provenance_confidence": format!("{:.3}", result.provenance_confidence),
        "retrieval_top3_mean_sim": format!("{:.3}", result.retrieval_top3_mean_sim),
        "retrieval_top3_kind_mix": result.retrieval_top3_kind_mix,
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

/// /sync — Ontvang weights van een andere Lucy instance en merge (auth-protected).
/// v3.1: routet de merge via het command-kanaal naar het BEZITTENDE veld. De oude versie
/// merge'de in een snapshot-kloon die de tick-loop elke ~50 ticks overschreef — dus geen enkel
/// geïmporteerd gewicht overleefde. De merge is compat-gated in de tick-loop (federation.rs).
async fn sync_handler(
    State(state): State<SharedState>,
    body: axum::body::Bytes,
) -> Json<Value> {
    match bincode::deserialize::<crate::federation::WeightSnapshot>(&body) {
        Ok(remote) => {
            let remote_tick = remote.tick;
            let (tx, rx) = tokio::sync::oneshot::channel();
            if state.field_handle.commands
                .send(crate::field_handle::FieldCommand::MergeWeights { snapshot: remote, respond: tx })
                .await
                .is_err()
            {
                return Json(json!({"error": "tick loop niet bereikbaar"}));
            }
            match rx.await {
                Ok(Ok(merged)) => Json(json!({"merged": merged, "remote_tick": remote_tick})),
                Ok(Err(e)) => Json(json!({"error": e, "remote_tick": remote_tick})),
                Err(_) => Json(json!({"error": "geen antwoord van tick loop"})),
            }
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

/// /admin/physics-proposals — Lucy's voorstellen voor zelf-verbetering van haar fysica
async fn physics_proposals_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let field = state.field_handle.snapshot.load();

    // Genereer huidige physics-proposal (als aanwezig)
    if let Some(proposal) = epistemic_broker::maybe_propose_physics_change(&field) {
        Json(json!({
            "proposal": {
                "tick": proposal.tick,
                "formula": proposal.formula,
                "current_value": format!("{:.6}", proposal.current_value),
                "proposed_value": format!("{:.6}", proposal.proposed_value),
                "rationale": proposal.rationale,
                "evidence_metric": proposal.evidence_metric,
                "baseline": format!("{:.6}", proposal.baseline),
            }
        }))
    } else {
        Json(json!({
            "proposal": null,
            "status": "no changes needed"
        }))
    }
}

/// /admin/grow — Voeg verborgen punten toe aan het veld
/// Trigger veldgroei, uitvoering via growth::grow_hidden()
async fn admin_grow_handler(
    State(state): State<SharedState>,
    axum::Json(payload): axum::Json<serde_json::Value>,
) -> impl IntoResponse {
    let n_new = payload
        .get("n")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;

    // Voeg punten toe via command-based interface
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let cmd = FieldCommand::GrowHidden { n: n_new, respond: resp_tx };

    if state.field_handle.commands.send(cmd).await.is_err() {
        return Json(json!({
            "error": "field command channel closed",
            "added": 0
        }));
    }

    // Wacht max 5 seconden op response
    match tokio::time::timeout(std::time::Duration::from_secs(5), resp_rx).await {
        Ok(Ok(result)) => Json(json!({
            "added": result,
            "total_points": state.field_handle.snapshot.load().points.len()
        })),
        _ => Json(json!({
            "error": "growth operation timeout",
            "added": 0
        }))
    }
}

/// /topology — Network topology for 3D brain visualization
/// Returns sampled nodes with 3D positions derived from depth + hash, and top couplings by |weight|
async fn topology_handler(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    let n_points = f.points.len();

    // Sample max 2000 nodes (all active + random sample of rest)
    let mut node_ids: Vec<usize> = f.active_set.iter().map(|&id| id as usize).collect();
    let max_nodes = 2000usize;
    if node_ids.len() < max_nodes && n_points > node_ids.len() {
        // Add random inactive nodes
        let mut rng = rand::rng();
        let remaining: Vec<usize> = (0..n_points)
            .filter(|i| !f.active_set.contains(&(*i as u32)))
            .collect();
        use rand::seq::SliceRandom;
        let mut sample = remaining;
        sample.shuffle(&mut rng);
        node_ids.extend(sample.into_iter().take(max_nodes - node_ids.len()));
    }
    node_ids.truncate(max_nodes);

    // Build node set for fast lookup
    let node_set: std::collections::HashSet<usize> = node_ids.iter().copied().collect();

    // Build nodes with 3D positions
    let nodes: Vec<Value> = node_ids.iter().map(|&id| {
        let p = &f.points[id];
        // Position from depth (y) + hash of id (x, z) — deterministic layout
        let hash = ((id as u64).wrapping_mul(2654435761)) as f32;
        let angle = (hash % 10000.0) / 10000.0 * std::f32::consts::TAU;
        let radius = (p.depth as f32 + 1.0) * 2.0 + (hash % 1000.0) / 1000.0;
        let x = angle.cos() * radius;
        let y = p.depth as f32 * 3.0 - 4.5;
        let z = angle.sin() * radius;

        let is_active = f.active_set.contains(&(id as u32));
        json!({
            "id": id,
            "pos": [x, y, z],
            "depth": p.depth,
            "active": is_active,
            "energy": p.energy,
            "tension": p.tension,
            "curiosity": p.curiosity,
            "satisfaction": p.satisfaction,
            "frustration": p.frustration,
            "phi_integrated": p.phi_integrated,
            "oscillation_phase": p.oscillation_phase,
        })
    }).collect();

    // Collect top couplings (only between sampled nodes)
    let mut edges: Vec<(usize, u32, f32)> = Vec::new();
    for &id in &node_ids {
        for c in &f.points[id].couplings {
            if node_set.contains(&(c.target as usize)) {
                edges.push((id, c.target, c.weight));
            }
        }
    }
    // Sort by |weight| descending, take top 5000
    edges.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
    edges.truncate(5000);

    let links: Vec<Value> = edges.iter().map(|(src, tgt, w)| {
        json!({"s": src, "t": tgt, "w": w})
    }).collect();

    Json(json!({
        "tick": f.tick,
        "total_points": n_points,
        "sampled": nodes.len(),
        "nodes": nodes,
        "links": links,
    }))
}

// =============================================================================
// v2.6.8 — Bewijs-architectuur endpoints
// =============================================================================

/// /self_metrics — Self-model coherentie metrics (density, drift, strength, hit-rate)
async fn self_metrics_handler(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    let retrieval = state.retrieval.read().await;

    let density = retrieval.self_cluster_density(50);
    let (centroid, strength) = retrieval.self_centroid_and_strength(50);
    let reuse = retrieval.self_reuse_rate();

    // Drift: cosine tussen huidige centroid en opgeslagen last_self_centroid
    let drift = if f.self_metrics.last_self_centroid.is_empty() {
        0.0
    } else {
        let last = &f.self_metrics.last_self_centroid;
        let mut dot = 0.0f32;
        let mut mag_a = 0.0f32;
        let mut mag_b = 0.0f32;
        let n = centroid.len().min(last.len());
        for i in 0..n {
            dot += centroid[i] * last[i];
            mag_a += centroid[i] * centroid[i];
            mag_b += last[i] * last[i];
        }
        let cos = if mag_a.sqrt() * mag_b.sqrt() > 1e-8 {
            dot / (mag_a.sqrt() * mag_b.sqrt())
        } else { 0.0 };
        1.0 - cos // drift = 1 - cosine; hoger = meer gewijzigd
    };

    let kind_dist = retrieval.kind_distribution();
    let self_entries_count = kind_dist[1];
    let total_entries = retrieval.len() as u32;

    // Hit-rate: atomic counters (lock-free) uit AppState
    use std::sync::atomic::Ordering::Relaxed;
    let total_q = state.total_queries.load(Relaxed);
    let fallback_q = state.fallback_queries.load(Relaxed);
    let self_hit_q = state.queries_with_self_hit.load(Relaxed);
    let hit_rate = if total_q > 0 { self_hit_q as f32 / total_q as f32 } else { 0.0 };
    let fallback_rate = if total_q > 0 { fallback_q as f32 / total_q as f32 } else { 0.0 };

    Json(json!({
        "tick": f.tick,
        "self_cluster_density": density,
        "self_cluster_drift": drift,
        "self_centroid_strength": strength,
        "self_query_hit_rate": hit_rate,
        "self_response_fallback_rate": fallback_rate,
        "self_state_reuse_rate": reuse,
        "self_entries_count": self_entries_count,
        "total_retrieval_entries": total_entries,
        "total_queries": total_q,
        "fallback_queries": fallback_q,
        "queries_with_self_hit": self_hit_q,
        "provenance_counts": {
            "fallback": state.provenance_counts[0].load(Relaxed),
            "field_generated": state.provenance_counts[1].load(Relaxed),
            "retrieval_self": state.provenance_counts[2].load(Relaxed),
            "retrieval_teacher": state.provenance_counts[3].load(Relaxed),
            "retrieval_external": state.provenance_counts[4].load(Relaxed),
            "mixed": state.provenance_counts[5].load(Relaxed),
        },
    }))
}

/// /teacher_metrics — AI-teacher bootstrap fade-out metrics
async fn teacher_metrics_handler(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    let explorer = state.explorer.read().await;

    let events = &explorer.teacher_audit;
    let total = events.len();
    let success_count = events.iter().filter(|e| e.sat_delta > 0.0).count();
    let pending_count = events.iter().filter(|e| e.effect_outcome.is_none()).count();
    let good_effect_count = events.iter().filter(|e| e.effect_outcome == Some(true)).count();
    let avg_sat_delta = if total > 0 {
        events.iter().map(|e| e.sat_delta).sum::<f32>() / total as f32
    } else { 0.0 };

    // Teacher events in laatste 1000 ticks
    let recent_cutoff = f.tick.saturating_sub(1000);
    let recent_events: Vec<&lucy_state::TeacherEvent> = events.iter()
        .filter(|e| e.tick >= recent_cutoff).collect();

    // Dependency ratio = teacher-provenance / total meaningful responses
    use std::sync::atomic::Ordering::Relaxed;
    let teacher_provs = state.provenance_counts[3].load(Relaxed); // retrieval_teacher
    let total_provs: u64 = (0..6).map(|i| state.provenance_counts[i].load(Relaxed)).sum();
    let dependency_ratio = if total_provs > 0 {
        teacher_provs as f64 / total_provs as f64
    } else { 0.0 };

    // Recent teacher_interval trajectory (laatste 20 unique values in events)
    let mut interval_traj: Vec<u64> = Vec::new();
    let mut last_seen = 0u64;
    for e in events.iter().rev().take(50) {
        // pending_until - tick ≈ 1000, so this is approximate; gebruik sat_delta > 0 als proxy
        if e.tick != last_seen {
            interval_traj.push(e.pending_until_tick.saturating_sub(e.tick));
            last_seen = e.tick;
        }
        if interval_traj.len() >= 20 { break; }
    }

    Json(json!({
        "tick": f.tick,
        "teacher_interval_current": f.teacher_interval,
        "teacher_success_streak": f.teacher_success_streak,
        "teacher_calls_total": total,
        "teacher_success_rate": if total > 0 { success_count as f32 / total as f32 } else { 0.0 },
        "avg_teacher_sat_delta": avg_sat_delta,
        "teacher_pending_evaluations": pending_count,
        "teacher_good_effect_count": good_effect_count,
        "teacher_dependency_ratio": dependency_ratio,
        "recent_events_1000_ticks": recent_events.len(),
        "provider_score": &explorer.provider_score,
        "interval_trajectory_approx": interval_traj,
        "recent_events_preview": events.iter().rev().take(10).map(|e| json!({
            "tick": e.tick,
            "provider": e.provider,
            "sat_delta": e.sat_delta,
            "effect_outcome": e.effect_outcome,
            "retrieval_reuse": e.retrieval_reuse_within_window,
            "preview": e.answer_preview.chars().take(80).collect::<String>(),
        })).collect::<Vec<_>>(),
    }))
}

/// /concept_audit — Per-cycle concept-clustering kwaliteit metrics
async fn concept_audit_handler(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    Json(json!({
        "tick": f.tick,
        "current_k": f.concept_centroids.len(),
        "priors_enabled": f.concept_priors_enabled,
        "priors_disabled_until_tick": f.concept_priors_disabled_until_tick,
        "audit_log": f.concept_audit_log.iter().rev().take(32).collect::<Vec<_>>(),
    }))
}

/// /self_ep_audit — Self-EP + WM-seq + dream-replay effect-metingen
async fn self_ep_audit_handler(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    let audit = &f.self_ep_audit;
    let total = audit.len();
    let avg_tension_delta = if total > 0 {
        audit.iter().map(|a| a.avg_tension_post - a.avg_tension_pre).sum::<f32>() / total as f32
    } else { 0.0 };
    let total_dream_replays: u32 = audit.iter().map(|a| a.dream_replay_count).sum();
    let total_wm_pairs: u32 = audit.iter().map(|a| a.wm_seq_pairs).sum();

    Json(json!({
        "tick": f.tick,
        "total_audit_entries": total,
        "avg_tension_delta_pre_post": avg_tension_delta,
        "total_dream_replay_events": total_dream_replays,
        "total_wm_seq_pairs": total_wm_pairs,
        "recent": audit.iter().rev().take(32).collect::<Vec<_>>(),
    }))
}

/// /self_history — Tijdreeks uit data/self_test.jsonl (laatste 240 rows ≈ 48u)
async fn self_history_handler(State(_state): State<SharedState>) -> Json<Value> {
    let path = std::path::Path::new("data/self_test.jsonl");
    if !path.exists() {
        return Json(json!({"rows": [], "note": "no self_test.jsonl yet"}));
    }
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().rev().take(240).collect();
            let rows: Vec<Value> = lines.into_iter().rev()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect();
            Json(json!({"rows": rows}))
        }
        Err(e) => Json(json!({"error": format!("{}", e)}))
    }
}

/// Ordinary-least-squares helling van y over de index 0..n. 0 bij < 2 punten.
fn ols_slope(ys: &[f64]) -> f64 {
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

/// /capability — reduceert data/self_test.jsonl tot per-batch aggregaten + trend-helling.
/// Maakt "wordt Lucy slimmer over tijd?" een meetbaar getal (score_slope > 0 = vooruitgang)
/// i.p.v. een gevoel. Score-samenstelling: zie self_test::score_probe (bewust zwakke proxy).
async fn capability_handler(State(_state): State<SharedState>) -> Json<Value> {
    let path = std::path::Path::new("data/self_test.jsonl");
    if !path.exists() {
        return Json(json!({"batches_total": 0, "series": [], "note": "no self_test.jsonl yet"}));
    }
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return Json(json!({"error": format!("{}", e)})),
    };
    // Groepeer per batch (opeenvolgende rijen met dezelfde tick). Append-log is chronologisch.
    struct Acc { tick: u64, n: u32, score_sum: f64, fallback: u32, self_ret: u32 }
    let mut batches: Vec<Acc> = Vec::new();
    for line in content.lines() {
        let v: Value = match serde_json::from_str(line) { Ok(v) => v, Err(_) => continue };
        let tick = v.get("tick").and_then(|x| x.as_u64()).unwrap_or(0);
        let score = v.get("score").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let fallback = v.get("fallback").and_then(|x| x.as_bool()).unwrap_or(false);
        let prov = v.get("provenance").and_then(|x| x.as_str()).unwrap_or("");
        let same = batches.last().map(|b| b.tick == tick).unwrap_or(false);
        if !same {
            batches.push(Acc { tick, n: 0, score_sum: 0.0, fallback: 0, self_ret: 0 });
        }
        let b = batches.last_mut().unwrap();
        b.n += 1;
        b.score_sum += score;
        if fallback { b.fallback += 1; }
        if prov == "retrieval_self" { b.self_ret += 1; }
    }
    let series: Vec<Value> = batches.iter().map(|b| {
        let n = b.n.max(1) as f64;
        json!({
            "tick": b.tick,
            "mean_score": b.score_sum / n,
            "fallback_rate": b.fallback as f64 / n,
            "retrieval_self_rate": b.self_ret as f64 / n,
            "n": b.n,
        })
    }).collect();
    let ys: Vec<f64> = batches.iter().map(|b| b.score_sum / b.n.max(1) as f64).collect();
    let slope = ols_slope(&ys);
    let latest = series.last().cloned().unwrap_or(Value::Null);
    let tail: Vec<Value> = series.iter().rev().take(240).cloned().collect::<Vec<_>>()
        .into_iter().rev().collect();
    Json(json!({
        "batches_total": batches.len(),
        "score_slope_per_batch": slope,
        "trend": if slope > 1e-4 { "stijgend" } else if slope < -1e-4 { "dalend" } else { "vlak" },
        "latest": latest,
        "series": tail,
    }))
}

/// /retrieval_metrics — cache-vs-brein bewijsmetrics
async fn retrieval_metrics_handler(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    let r = state.retrieval.read().await;
    let kind_counts = r.kind_distribution();
    let total = r.len() as u64;

    Json(json!({
        "tick": f.tick,
        "total_entries": total,
        "by_kind": {
            "external": kind_counts[0],
            "self": kind_counts[1],
            "teacher": kind_counts[2],
            "interaction": kind_counts[3],
        },
        "cap": 15000,
        "usage_fraction": if total > 0 { total as f32 / 15000.0 } else { 0.0 },
    }))
}

/// POST /debug/toggle_priors?enabled=false&ticks=1000
/// Ablation window: zet concept-priors uit voor N ticks, auto-restore.
async fn debug_toggle_priors_handler(
    State(state): State<SharedState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<Value> {
    let enabled_str = params.get("enabled").cloned().unwrap_or_else(|| "true".to_string());
    let enabled = enabled_str == "true";
    let ticks: u64 = params.get("ticks").and_then(|s| s.parse().ok()).unwrap_or(1000);

    // Command naar tick-loop: set flags
    // We gebruiken FieldCommand::SetConceptPriors (nieuw)
    let _ = state.field_handle.commands.send(FieldCommand::SetConceptPriors {
        enabled,
        until_delta_ticks: ticks,
    }).await;

    Json(json!({
        "ok": true,
        "enabled": enabled,
        "window_ticks": ticks,
        "note": "ablation toegepast op next adapt cycle",
    }))
}

/// /telemetry — Compact real-time telemetry for dashboard polling (lightweight)
async fn telemetry_handler(State(state): State<SharedState>) -> Json<Value> {
    let f = state.field_handle.snapshot.load();
    let n = f.active_set.len() as f32;

    let (avg_curiosity, avg_satisfaction, avg_frustration) = if n > 0.0 {
        let c: f32 = f.active_set.iter().map(|&id| f.points[id as usize].curiosity).sum::<f32>() / n;
        let s: f32 = f.active_set.iter().map(|&id| f.points[id as usize].satisfaction).sum::<f32>() / n;
        let fr: f32 = f.active_set.iter().map(|&id| f.points[id as usize].frustration).sum::<f32>() / n;
        (c, s, fr)
    } else { (0.0, 0.0, 0.0) };

    let avg_tension = if n > 0.0 {
        f.active_set.iter().map(|&id| f.points[id as usize].tension).sum::<f32>() / n
    } else { 0.0 };

    // v2.5: Alles uit Field snapshot (lock-free via ArcSwap)
    let vocab_size = state.vocab.try_read().map(|v| v.vocab_size()).unwrap_or(0);
    let pages_read = f.pages_read;         // gesync'd vanuit explorer elke 50 ticks
    let tokens_processed = f.tokens_processed;
    let n_couplings: usize = f.active_set.iter()
        .map(|&id| f.points[id as usize].couplings.len())
        .sum();

    // v2.6.11: GNW-stats runtime-computed zodat het dashboard live kan zien of
    // selectiecompetitie actief is. Ontwerpregel: "geen volledige activatie
    // zonder competitie" wordt hier meetbaar. Zonder deze cijfers was de
    // differentiatie-collapse van 2026-04-15 alleen zichtbaar via docker logs.
    let mut n_depth3_total = 0usize;
    let mut n_depth3_active = 0usize;
    let mut scores: Vec<f32> = Vec::with_capacity(256);
    for p in f.points.iter() {
        if p.depth != 3 { continue; }
        n_depth3_total += 1;
        // is_active: FLAG_ACTIVE = bit 0
        if (p.flags & 1) == 0 { continue; }
        n_depth3_active += 1;
        let phi_norm: f32 = p.phi.iter().map(|v| v * v).sum::<f32>().sqrt();
        let s = phi_norm * (1.0 + p.phi_integrated) * (1.0 + p.expected_free_energy * 0.5);
        if s > 0.0001 { scores.push(s); }
    }
    let d3_ratio = if n_depth3_total > 0 {
        n_depth3_active as f32 / n_depth3_total as f32
    } else { 0.0 };
    let (score_mean, score_cv) = if scores.len() > 1 {
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        (mean, if mean > 1e-6 { var.sqrt() / mean } else { 0.0 })
    } else { (0.0, 0.0) };
    // Invariant-schending (ontwerpregel): d3_ratio ≥ 0.5 én score_cv < 0.05
    // = selectie-collapse. Als dit waar is hoort de collapse detector gefuurd te
    // hebben; als we dit in telemetry zien staan we dus effectief al te wachten
    // op de volgende GNW cycle.
    let in_collapse = scores.len() >= 64 && d3_ratio > 0.5 && score_cv < 0.05;

    // v2.6.12: cap observability — atomics uitlezen vóór json! macro
    let ign_cap_hits = crate::compute::IGNITION_CAP_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let ign_samples = crate::compute::IGNITION_SAMPLES.load(std::sync::atomic::Ordering::Relaxed);
    let ign_raw = f32::from_bits(
        crate::compute::IGNITION_LAST_RAW_BITS.load(std::sync::atomic::Ordering::Relaxed)
    );
    let ign_cap_ratio: f32 = if ign_samples > 0 {
        ign_cap_hits as f32 / ign_samples as f32
    } else { 0.0 };

    // v2.6.13: winner turnover (compute.rs) + goal latency (thinker.rs)
    let win_repeats = crate::compute::WINNER_REPEATS.load(std::sync::atomic::Ordering::Relaxed);
    let win_switches = crate::compute::WINNER_SWITCHES.load(std::sync::atomic::Ordering::Relaxed);
    let win_turnover_ratio: f32 = {
        let total = win_repeats + win_switches;
        if total > 0 { win_switches as f32 / total as f32 } else { 0.0 }
    };
    let goal_complete_total = crate::thinker::GOAL_COMPLETE_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let goal_latency_sum = crate::thinker::GOAL_LATENCY_SUM_TICKS.load(std::sync::atomic::Ordering::Relaxed);
    let goal_mean_latency: f32 = if goal_complete_total > 0 {
        goal_latency_sum as f32 / goal_complete_total as f32
    } else { 0.0 };
    let ml_spawn = crate::thinker::META_LEARN_SPAWN_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let ml_complete = crate::thinker::META_LEARN_COMPLETE_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let ml_latency_sum = crate::thinker::META_LEARN_LATENCY_SUM_TICKS.load(std::sync::atomic::Ordering::Relaxed);
    let ml_mean_latency: f32 = if ml_complete > 0 {
        ml_latency_sum as f32 / ml_complete as f32
    } else { 0.0 };
    let goal_repeat_events = crate::thinker::GOAL_REPEAT_EVENTS.load(std::sync::atomic::Ordering::Relaxed);

    Json(json!({
        "tick": f.tick,
        "phase": format!("{:?}", f.phase),
        "points": f.points.len(),
        "active": f.active_set.len(),
        "tension": avg_tension,
        // v2.5: field stats direct in telemetry (geen /status nodig)
        "vocab_size": vocab_size,
        "pages_read": pages_read,
        "tokens_processed": tokens_processed,
        "couplings": n_couplings,
        "reflection_count": f.reflection_count,
        "emotions": {
            "curiosity": avg_curiosity,
            "satisfaction": avg_satisfaction,
            "frustration": avg_frustration,
        },
        "neuro": {
            "dopamine": f.dopamine,
            "norepinephrine": f.norepinephrine,
            "serotonin": f.serotonin,
            "acetylcholine": f.acetylcholine,
        },
        "consciousness": {
            "spectral_radius": f.spectral_radius,
            "phi": f.integrated_info_avg,
            "coherence": f.oscillation_coherence,
            "free_energy": f.avg_free_energy,
            "is_critical": f.is_critical,
            // v2.4
            "theta_gamma_coupling": f.theta_gamma_coupling,
            "branching_ratio": f.branching_ratio_avg,
            "ignition_active": f.ignition_active,
            "ignition_strength": f.ignition_strength,
            "expected_free_energy": f.avg_expected_free_energy,
        },
        // v2.6.11: GNW sparsity-stats — observability voor de ontwerpregel
        // "geen volledige activatie zonder competitie".
        // d3_ratio is typisch ~1.0 in gezonde operatie (FLAG_ACTIVE is persistent na
        // growth) — de échte drift-metric is `score_cv`. Collapse triggert bij
        // cv < 0.05 én ratio > 0.5 (dat laatste is in praktijk altijd waar).
        // Watch `score_cv` voor gezondheid, `ignition_strength` voor runaway.
        //
        // v2.6.12: Fase-2 cap observability. Raw (pre-clamp), clamped (=
        // field.ignition_strength want bij runaway steeds gelijk aan CAP), en
        // cumulatieve cap-hit ratio sinds proces-start. Als ratio daalt → cap
        // doet z'n werk én het systeem stabiliseert eronder.
        "gnw_stats": {
            "depth3_total": n_depth3_total,
            "depth3_active": n_depth3_active,
            "d3_ratio": d3_ratio,
            "score_mean": score_mean,
            "score_cv": score_cv,
            "n_candidates": scores.len(),
            "collapse_condition_met": in_collapse,
            "ignition_cap": crate::compute::IGNITION_CAP,
            "ignition_strength_raw": ign_raw,
            "ignition_strength_clamped": f.ignition_strength,
            "ignition_cap_hits_total": ign_cap_hits,
            "ignition_samples_total": ign_samples,
            "ignition_cap_hit_ratio": ign_cap_ratio,
            // v2.6.13: winner turnover — switches/(switches+repeats) over proces-leven.
            // Hoog = diverse winners (gezond). Laag = oligarchie (top-K dwingt sparsity
            // af maar zelfde paar nodes winnen).
            "winner_repeats": win_repeats,
            "winner_switches": win_switches,
            "winner_turnover_ratio": win_turnover_ratio,
        },
        // v2.6.13: goal completion latency — geeft aan of goals instant recursief
        // completen (=collapse-symptoom) of in gezonde tempo.
        "goal_stats": {
            "complete_total": goal_complete_total,
            "mean_latency_ticks": goal_mean_latency,
            "meta_learning_spawn_total": ml_spawn,
            "meta_learning_complete_total": ml_complete,
            "meta_learning_mean_latency_ticks": ml_mean_latency,
            "repeat_events_total": goal_repeat_events,
        },
        "goals": f.goals.iter().take(5).map(|g| {
            json!({
                "description": g.description,
                "priority": g.priority,
                "progress": g.progress,
            })
        }).collect::<Vec<_>>(),
    }))
}

/// /plugins — List available dashboard plugins (from data/plugins/)
/// v2.6.11: Unambiguous version endpoint — bouwtijd, git-sha en schema-versie.
/// Geen guesswork meer over wat er in prod draait.
async fn version_handler() -> Json<Value> {
    Json(json!({
        "version": env!("LUCY_VERSION"),
        "git_sha": env!("LUCY_GIT_SHA"),
        "built_at": env!("LUCY_BUILD_TIME"),
        "schema_version": lucy_state::snapshot_header::CURRENT_SCHEMA_VERSION,
        "supported_legacy_schemas": ["v0 (v2.6.5-era)", "v1 (v2.6.7-era)"],
        "rustc_target": std::env::consts::ARCH,
        "os": std::env::consts::OS,
    }))
}

async fn plugins_list_handler() -> Json<Value> {
    let plugin_dir = std::path::Path::new("data/plugins");
    let mut plugins = Vec::new();

    if plugin_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(plugin_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(manifest) = serde_json::from_str::<Value>(&content) {
                            plugins.push(manifest);
                        }
                    }
                }
            }
        }
    }

    Json(json!({ "plugins": plugins }))
}

// v2.6.17: Epistemic awareness endpoint
async fn epistemic_handler(State(state): State<SharedState>) -> Json<Value> {
    let field_guard = state.field_handle.snapshot.load();
    if !field_guard.points.is_empty() {
        let gaps = crate::epistemic_broker::least_stable_concepts(&field_guard, 5);

        let gaps_json: Vec<Value> = gaps
            .iter()
            .map(|g| {
                json!({
                    "concept_idx": g.concept_idx,
                    "drift": g.drift,
                    "silhouette": g.silhouette,
                    "urgency": g.urgency,
                })
            })
            .collect();

        let maybe_proposal = crate::epistemic_broker::maybe_propose_parameter_change(&field_guard);
        let proposal_json = maybe_proposal
            .map(|p| {
                json!({
                    "tick": p.tick,
                    "observation": p.observation,
                    "hypothesis": p.hypothesis,
                    "proposed_change": p.proposed_change,
                    "confidence": p.confidence,
                })
            })
            .unwrap_or(Value::Null);

        Json(json!({
            "is_stuck": gaps.first().map(|g| g.urgency > 0.7).unwrap_or(false),
            "gaps": gaps_json,
            "current_proposal": proposal_json,
            "tick": field_guard.tick,
        }))
    } else {
        Json(json!({
            "error": "field snapshot unavailable"
        }))
    }
}

// v2.6.17: Parameter proposals endpoint (stubs for now)
async fn proposals_handler() -> Json<Value> {
    // TODO: persist proposals to data/proposals.json and serve them
    Json(json!({
        "proposals": [],
        "note": "proposal persistence coming in v2.6.18"
    }))
}

// v3.0: Self-Fix Orchestrator — Full Auto-Patch Pipeline
async fn self_fix_handler(State(state): State<SharedState>) -> Json<Value> {
    use crate::diagnostics::{BottleneckDetector, TelemetrySnapshot};
    use crate::compile_sandbox::SandboxCompiler;
    use crate::deploy_coordinator::DeployCoordinator;

    let field = (*state.field_handle.snapshot.load()).clone();
    let now = std::time::Instant::now();

    // Phase 1: Diagnostics — bereken emoties als gemiddelde over active_set (zelfde als /telemetry)
    let n = field.active_set.len().max(1) as f32;
    let avg_curiosity = field.active_set.iter()
        .map(|&id| field.points[id as usize].curiosity).sum::<f32>() / n;
    let avg_satisfaction = field.active_set.iter()
        .map(|&id| field.points[id as usize].satisfaction).sum::<f32>() / n;
    let avg_frustration = field.active_set.iter()
        .map(|&id| field.points[id as usize].frustration).sum::<f32>() / n;

    let telemetry_snapshot = TelemetrySnapshot {
        tick: field.tick,
        dopamine: field.dopamine,
        frustration: avg_frustration,
        satisfaction: avg_satisfaction,
        curiosity: avg_curiosity,
        free_energy: field.avg_free_energy,
        tension: field.points.iter().map(|p| p.tension).sum::<f32>() / field.points.len().max(1) as f32,
        ignition_active: field.ignition_active,
        ignition_strength: field.ignition_strength,
        goal_progress: field.goals.first().map(|g| g.progress).unwrap_or(0.0),
        n_goals: field.goals.len(),
        reflect_interval: field.reflect_interval,
    };

    let diagnoses = BottleneckDetector::diagnose(&field, &telemetry_snapshot);
    let triage = BottleneckDetector::triage(&diagnoses);

    if diagnoses.is_empty() {
        return Json(json!({
            "status": "ok",
            "message": "No bottlenecks detected",
            "tick": field.tick,
            "triage": triage,
        }));
    }

    // Phase 2-4: sandbox setup + patch/compile + deploy zijn blokkerend
    // (synchrone cargo build + thread::sleep). Draai op de blocking pool
    // zodat de async runtime niet vastloopt.
    let diagnoses_count = diagnoses.len();
    let blocking_result = tokio::task::spawn_blocking(move || -> Value {
        // Phase 2: Sandbox Setup
        let sandbox = SandboxCompiler::new();
        if let Err(e) = sandbox.setup() {
            return json!({
                "status": "failed",
                "phase": "sandbox_setup",
                "error": e,
            });
        }

        // Phase 3: Apply Patches & Compile
        let mut patches_applied = 0;
        let mut compile_error = None;

        for diagnosis in &diagnoses {
            let patch = &diagnosis.suggested_patch;
            if sandbox.apply_patch(&patch.file, &patch.old_pattern, &patch.new_pattern).is_ok() {
                patches_applied += 1;
            }
        }

        // Try to compile
        match sandbox.compile() {
            Ok(compile_result) => {
                if !compile_result.success {
                    compile_error = Some(format!("Compilation failed: {:?}", compile_result.errors));
                    sandbox.cleanup();
                    return json!({
                        "status": "failed",
                        "phase": "compilation",
                        "error": compile_error,
                    });
                }

                // Phase 4: Deploy
                let deployer = DeployCoordinator::new();
                if let Some(binary_path) = &compile_result.binary_path {
                    match deployer.deploy(binary_path, "http://localhost:3000", "") {
                        Ok(_) => {
                            sandbox.cleanup();
                            return json!({
                                "status": "success",
                                "message": "Auto-deployed",
                                "patches_applied": patches_applied,
                                "elapsed_seconds": now.elapsed().as_secs_f32(),
                                "diagnoses_count": diagnoses_count,
                            });
                        }
                        Err(e) => {
                            let _ = deployer.rollback();
                            sandbox.cleanup();
                            return json!({
                                "status": "failed",
                                "phase": "deployment",
                                "error": e,
                            });
                        }
                    }
                }
            }
            Err(e) => {
                sandbox.cleanup();
                return json!({
                    "status": "failed",
                    "phase": "compilation",
                    "error": e,
                });
            }
        }

        json!({
            "status": "failed",
            "message": "Unexpected state in self_fix handler"
        })
    }).await;

    match blocking_result {
        Ok(v) => Json(v),
        Err(e) => Json(json!({
            "status": "failed",
            "phase": "runtime",
            "http_status": 500,
            "error": format!("Blocking self-fix taak mislukt: {}", e),
        })),
    }
}
