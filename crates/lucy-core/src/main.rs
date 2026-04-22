//! # Lucy v0.9 — Field Intelligence + RAG + Attention + Working Memory
//!
//! Lucy is een zelflerend AI-systeem gebaseerd op een samenveld (unified field)
//! dat leert via Equilibrium Propagation. Geen backprop, geen externe AI,
//! geen hardcoded bronnen. 100% autonoom.
//!
//! ## Architectuur Overzicht
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    lucy-core (dit binary)                   │
//! │                                                             │
//! │  main.rs ── Tick loop: de hartslag van Lucy                │
//! │     ├── compute.rs ── Velddynamica + Equilibrium Propagation│
//! │     ├── encoding.rs ── Tekst → 64-dim embeddings           │
//! │     ├── decoding.rs ── 64-dim output → tekst               │
//! │     ├── query.rs ── Vraag → veld settle → antwoord         │
//! │     ├── field.rs ── Small-world netwerk bootstrap           │
//! │     ├── growth.rs ── Veldgroei + energie-economie           │
//! │     ├── meta.rs ── Vastloop-detectie + escalatie            │
//! │     ├── learner.rs ── Autonome exploratie + zelf-leren      │
//! │     ├── consolidation.rs ── Replay buffer (hippocampaal)    │
//! │     ├── curriculum.rs ── HTML/CSS patroon-training          │
//! │     ├── journal.rs ── Lucy's autonome observaties           │
//! │     ├── self_modify.rs ── Zelf-aanpassing + hercompilatie   │
//! │     ├── gpu_compute.rs ── Lokale GPU via wgpu shaders       │
//! │     ├── gpu_client.rs ── Remote GPU bridge (Vast.ai)        │
//! │     ├── gpu_manager.rs ── GPU-economie + autonome promotie  │
//! │     └── server.rs ── HTTP API (axum, port 3000)             │
//! │                                                             │
//! │  lucy-gpu ── Standalone GPU service (draait op Vast.ai)     │
//! │  lucy-state ── Gedeelde datastructuren (Field, FieldPoint)  │
//! │  lucy-io ── HTTP client + filesystem                        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Tick Loop (de hartslag)
//!
//! Lucy draait in een oneindige loop waar elke tick het volgende doet:
//!
//! | Fase | Interval | Wat het doet |
//! |------|----------|-------------|
//! | 1. Velddynamica | Elke tick | Φ update, geheugen, spanning voor alle actieve punten |
//! | 2. Exploratie-ruis | Elke 10 ticks | Random perturbatie als spanning te laag |
//! | 3. Gewicht updates | Elke 100 ticks | Hebbian learning + meta-leren op η |
//! | 4. Groei | Elke 10.000 ticks | Splits vastgelopen punten, prune dode verbindingen |
//! | 5. Energie | Elke tick | Kosten, beloningen, deactivatie, reactivatie |
//! | 6. Consolidatie | Elke 500 ticks | Replay een willekeurig oud trainingsvoorbeeld |
//! | 7. Logging | Elke 1.000 ticks | Heartbeat: spanning, ticks/sec, vocab, explorer stats |
//! | 8. Exploratie | Elke 5.000 ticks | Bezoek een URL, leer van de content |
//! | 9. Zelf-lezing | Elke 5.000 ticks | Herlees eigen source code |
//! | 10. Curriculum | Elke 50.000 ticks | HTML/CSS patroon-training |
//! | 11. GPU economie | Elke 50.000 ticks | Check balance, promotie als nodig |
//! | 12. Snapshot | Elke 10.000 ticks | Sla alles op naar disk |
//!
//! ## Zelfleren
//!
//! Lucy's leerproces is volledig autonoom:
//! 1. **Boot**: leert van eigen source code → bouwt vocabulaire + interesses op
//! 2. **Curiosity**: genereert zoekopdrachten vanuit interesses → DuckDuckGo
//! 3. **Exploratie**: bezoekt gevonden URLs → leert taalpatronen + ontdekt meer URLs
//! 4. **Interactie**: elke `/ask` of `/feed` verrijkt vocabulaire en interesses
//! 5. **Consolidatie**: herhaalt oude voorbeelden om kennis te verankeren
//!
//! ## Persistentie
//!
//! Alles wordt periodiek opgeslagen in `data/`:
//! - `lucy.snapshot` — volledige veldstaat (bincode)
//! - `vocab.bincode` — vocabulaire + embeddings
//! - `explorer.bincode` — URL frontier + bezochte URLs + interesses
//! - `replay.bincode` — consolidatie replay buffer

mod compute;
mod consolidation;
mod decoding;
mod encoding;
mod field;
mod growth;
mod learner;
mod meta;
mod query;
mod curriculum;
mod journal;
mod self_modify;
mod gpu_client;
mod gpu_compute;
mod gpu_manager;
mod perception;
mod retrieval;
mod thinker;
mod field_handle;
mod backup;
mod epistemic_broker;
mod compute_broker;
mod poster;
mod ai_teacher;
mod self_test;
mod replica_manager;
mod federation;
mod diagnostics;
mod compile_sandbox;
mod simulate;
mod deploy_coordinator;
mod feedback_loop;
mod self_fix;
mod server;

use lucy_io::fs::FileSystem;
use lucy_state::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

// === Persistentie paden ===
const SNAPSHOT_PATH: &str = "data/lucy.snapshot";
const VOCAB_PATH: &str = "data/vocab.bincode";
const EXPLORER_PATH: &str = "data/explorer.bincode";
const REPLAY_PATH: &str = "data/replay.bincode";
const RETRIEVAL_PATH: &str = "data/retrieval.bincode";

// === Tick intervals ===
// v2.3: De meeste intervallen zijn nu dynamisch in field.* (bijgewerkt door adapt_dynamics).
// Alleen LOG_INTERVAL, CURRICULUM_INTERVAL, GPU_BALANCE_CHECK blijven const
// (infra-gerelateerd, niet veld-afhankelijk).
const LOG_INTERVAL: u64 = 1_000;          // Heartbeat logging (infra)
const SELF_READ_INTERVAL: u64 = 10_000;   // Herlees eigen source code (structureel)
const CURRICULUM_INTERVAL: u64 = 50_000;  // HTML/CSS curriculum training (structureel)
const GPU_BALANCE_CHECK: u64 = 5_000;    // Check Vast.ai balance + posting (elke ~40 min ipv 7 uur)
// Dynamische intervallen: field.learn_interval, field.growth_interval, field.snapshot_interval,
// field.explore_interval, field.consolidation_interval — aangepast door adapt_dynamics()

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    // v1.3: Panic hook — log panics zodat we weten wat er fout gaat
    std::panic::set_hook(Box::new(|info| {
        tracing::error!("PANIC: {}", info);
    }));

    // v2.6.11: startup-banner met precieze build-identificatie. Voorkomt dat we
    // weer maanden denken dat v2.6.x draait terwijl het eigenlijk een oudere
    // binary is die nooit vervangen is.
    tracing::info!(
        version = env!("LUCY_VERSION"),
        git_sha = env!("LUCY_GIT_SHA"),
        built_at = env!("LUCY_BUILD_TIME"),
        schema = lucy_state::snapshot_header::CURRENT_SCHEMA_VERSION,
        "Lucy binary start"
    );
    tracing::info!("Lucy v0.9 — Field Intelligence + RAG + Attention + Working Memory — waking up...");

    let root = PathBuf::from(".");
    let fs = FileSystem::new(root.clone());

    // === Laad of bootstrap veld ===
    // v2.5: ONSTERFELIJKE LOAD — primary → backups 1-5 → fresh bootstrap
    // Lucy's geleerde data gaat NOOIT verloren.
    let snapshot_path = Path::new(SNAPSHOT_PATH);
    let mut field = if let Some(mut f) = Field::load_with_fallback(snapshot_path) {
        f.cache_io_ids();
        tracing::info!(tick = f.tick, points = f.points.len(), "Snapshot geladen (met fallback)");
        f
    } else {
        tracing::error!("ALLE snapshots gefaald — bootstrapping fresh. GELEERDE DATA VERLOREN!");
        field::bootstrap_field()
    };

    // v2.5: Onmiddellijk healen na laden/bootstrap — voorkomt inf bij tick 0
    let healed = field.heal();
    if healed > 0 {
        tracing::info!(healed, "Post-bootstrap healing complete v2.6.2 — NaN-safe EMA");
    }

    // v2.6.1: Auto-rewire als couplings verloren zijn (homeostasis death spiral recovery)
    let total_couplings: usize = field.points.iter().map(|p| p.couplings.len()).sum();
    if total_couplings == 0 && !field.points.is_empty() {
        tracing::warn!("GEEN COUPLINGS GEVONDEN — rewire field (phi/memory/vocab intact)");
        field::rewire_field(&mut field);
    }

    // === Laad vocabulaire ===
    let vocab_path = Path::new(VOCAB_PATH);
    let mut vocab = if vocab_path.exists() {
        encoding::Vocabulary::load(vocab_path).unwrap_or_else(|e| {
            tracing::warn!(%e, "Failed to load vocab, starting fresh");
            encoding::Vocabulary::new()
        })
    } else {
        encoding::Vocabulary::new()
    };

    // === Laad explorer ===
    let explorer_path = Path::new(EXPLORER_PATH);
    let mut explorer = if explorer_path.exists() {
        learner::Explorer::load(explorer_path).unwrap_or_else(|e| {
            tracing::warn!(%e, "Failed to load explorer, starting fresh");
            learner::Explorer::new()
        })
    } else {
        learner::Explorer::new()
    };

    // === Laad replay buffer ===
    let replay_path = Path::new(REPLAY_PATH);
    let replay = if replay_path.exists() {
        consolidation::ReplayBuffer::load(replay_path).unwrap_or_else(|e| {
            tracing::warn!(%e, "Failed to load replay buffer, starting fresh");
            consolidation::ReplayBuffer::with_capacity(vocab.vocab_size())
        })
    } else {
        consolidation::ReplayBuffer::with_capacity(vocab.vocab_size())
    };

    // === Laad retrieval store ===
    let retrieval_path = Path::new(RETRIEVAL_PATH);
    let retrieval_store = if retrieval_path.exists() {
        retrieval::RetrievalStore::load(retrieval_path).unwrap_or_else(|e| {
            tracing::warn!(%e, "Failed to load retrieval store, starting fresh");
            retrieval::RetrievalStore::new()
        })
    } else {
        retrieval::RetrievalStore::new()
    };

    // === v1.0: Bootstrap SYNCHROON — BPE + basistaal VOOR alles ===
    // Dit MOET klaar zijn voordat de tick loop of API start.
    // Geen async, geen locks, geen race conditions.
    // v1.3: Skip als vocab al geladen is (= snapshot van eerdere run bevat al BPE)
    let bootstrap_marker = Path::new("data/.bootstrapped");
    let already_trained = bootstrap_marker.exists() || vocab.vocab_size() > 1000;
    if !already_trained {
        tracing::info!("First boot — synchrone bootstrap (BPE + basistaal)");

        // Stap 1: Lees eigen source code
        let source_text = match fs.read_own_source() {
            Ok(files) => {
                let text = files.into_iter().map(|(_, c)| c).collect::<Vec<_>>().join("\n");
                tracing::info!(chars = text.len(), "Source code gelezen");
                text
            }
            Err(e) => {
                tracing::warn!(%e, "Kon source niet lezen, skip bootstrap");
                String::new()
            }
        };

        // Stap 2: BPE merges — direct op vocab, geen lock
        if !source_text.is_empty() {
            vocab.learn_bpe(&source_text, 3000);
            tracing::info!(vocab = vocab.vocab_size(), "BPE merges geleerd");
        }

        // v2.0: GEEN hardcoded taal meer — Lucy leert ALLES organisch
        // Haar source code (Rust/Engels) is haar eerste taalblootstelling
        // Internet exploratie brengt alle talen — Lucy kiest zelf
        tracing::info!("v2.0: Geen hardcoded taal — Lucy leert organisch");

        // Stap 4: Minimale source training — 1 chunk van 500 chars
        // Rest leert Lucy organisch via exploration + self-reading
        if !source_text.is_empty() {
            let chunk: String = source_text.chars().take(500).collect();
            let trained = learner::learn_from_text(&mut field, &mut vocab, &chunk);
            tracing::info!(tokens = trained, "Minimale source training voltooid");
        }

        // Stap 5: Seed interesses
        if !source_text.is_empty() {
            explorer.absorb_interests(&source_text);
        }

        tracing::info!("Bootstrap voltooid — Lucy is klaar om te denken");
        let _ = std::fs::write(bootstrap_marker, b"done");
    }

    // v2.5: Seed diverse bronnen (uit data/sources.txt) + diversifieer frontier
    explorer.seed_api_discovery(); // leest uit sources.txt (niet meer hardcoded)
    explorer.load_self_managed_apis();
    // Schoon de frontier op: max 20% per domein (breekt API-monoculture)
    explorer.diversify_frontier();

    // v3.0.1: Force CERN-seed in frontier — observability boost voor particle physics data
    for url in &[
        "https://opendata.cern.ch/api/records/?q=higgs+boson&type=Dataset&size=5",
        "https://opendata.cern.ch/api/records/?q=dark+matter&type=Dataset&size=5",
        "https://opendata.cern.ch/api/records/?q=standard+model&type=Dataset&size=5",
    ] {
        explorer.discover_urls(url);
    }

    tracing::info!(frontier = explorer.frontier.len(), interests = explorer.interests.len(), "Exploration bronnen geladen + gediversifieerd + CERN seed");

    // === v0.7: Initialiseer bewustzijns-systemen ===
    compute::init_rnd(&mut field);
    compute::assign_depths(&mut field);

    // v2.6.5: Rewire ALLEEN als topologie echt kapot is (niet elke startup)
    let avg_couplings = if field.points.is_empty() { 0 } else { total_couplings / field.points.len() };
    if avg_couplings < 5 {
        tracing::warn!(avg_couplings, "Topologie te dun — rewire met K=20");
        field::rewire_field(&mut field);
    }
    compute::assign_depths(&mut field);
    // Verdicht binnen depth-lagen (idempotent — skipt punten die al ≥20 hebben)
    field::densify_couplings(&mut field, 20);
    compute::assign_depths(&mut field);

    // v2.6.5: Reset parameters uit snapshot die vastzaten
    field.serotonin = field.dopamine;   // voorkom vastzit op 2.0
    field.ema_decay = 0.90;             // was 0.99 in snapshot → bewustzijn dood
    if !field.ignition_strength.is_finite() { field.ignition_strength = 0.0; } // NaN cleanup

    tracing::info!(
        spectral_radius = field.spectral_radius,
        dopamine = field.dopamine,
        "v0.7 consciousness systems initialized"
    );

    // === Zelf-aanpassingssysteem ===
    let self_modifier = self_modify::SelfModifier::new(root.clone());

    // === Dagboek ===
    let journal_path = Path::new("data/journal.json");
    let journal = if journal_path.exists() {
        journal::Journal::load(journal_path).unwrap_or_else(|_| journal::Journal::new())
    } else {
        journal::Journal::new()
    };

    // v2.5: Herstel verloren tellers vanuit journal (overleeft struct-wijzigingen)
    if field.reflection_count == 0 && journal.reflection_count > 0 {
        tracing::info!(
            journal_reflections = journal.reflection_count,
            "Reflection count hersteld vanuit journal"
        );
        field.reflection_count = journal.reflection_count;
    }
    // Sync journal → field voor pages/tokens (als field ze kwijt is)
    if field.pages_read == 0 && journal.total_pages_read > 0 {
        field.pages_read = journal.total_pages_read;
        field.tokens_processed = journal.total_tokens_processed;
        tracing::info!(pages = journal.total_pages_read, "Explorer stats hersteld vanuit journal");
    }

    // === GPU initialisatie: remote GPU (Vast.ai) → lokale GPU → CPU ===
    let gpu_url = std::env::var("GPU_URL").ok();
    let mut remote_gpu = if let Some(ref url) = gpu_url {
        let client = gpu_client::GpuClient::new(url);
        if client.is_available().await {
            tracing::info!(url, "Remote GPU service gevonden");
            Some(client)
        } else {
            tracing::warn!(url, "Remote GPU niet bereikbaar — probeer lokale GPU");
            None
        }
    } else {
        None
    };

    // Upload veldstaat naar remote GPU als beschikbaar (max 15s)
    if let Some(ref mut rg) = remote_gpu {
        match tokio::time::timeout(std::time::Duration::from_secs(15), rg.upload(&field)).await {
            Ok(Ok(())) => tracing::info!("Remote GPU upload gelukt"),
            Ok(Err(e)) => {
                tracing::warn!(%e, "Remote GPU upload failed — fallback naar CPU");
                remote_gpu = None;
            }
            Err(_) => {
                tracing::warn!("Remote GPU upload timeout (15s) — fallback naar CPU");
                remote_gpu = None;
            }
        }
    }

    // Lokale GPU als fallback
    let local_gpu = if remote_gpu.is_none() {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu_compute::GpuCompute::new(&field)
        })) {
            Ok(g) => {
                tracing::info!("Local GPU compute enabled");
                Some(g)
            }
            Err(_) => {
                tracing::warn!("No GPU available — CPU compute fallback");
                None
            }
        }
    } else {
        None
    };

    // === GPU economie ===
    let mut gpu_economy = gpu_manager::GpuEconomy::new();

    // Check Vast.ai balance als API key beschikbaar is
    if let Ok(api_key) = std::env::var("VASTAI_API_KEY") {
        match gpu_economy.check_balance(&api_key).await {
            Ok(bal) => tracing::info!(balance = format!("${:.2}", bal), "Vast.ai balance checked"),
            Err(e) => tracing::warn!(%e, "Could not check Vast.ai balance"),
        }
    }

    // === v1.0: Dubbel-Buffer Architectuur ===
    // Field wordt NIET in Arc<RwLock> gewrapt — tick loop bezit het direct.
    // API endpoints werken op snapshots via FieldHandle.

    use field_handle::{FieldHandle, FieldCommand, FieldStatus, ExploreResult, WeightDelta, field_to_status};
    use tokio::sync::{mpsc, watch};
    use arc_swap::ArcSwap;

    let (cmd_tx, mut cmd_rx) = mpsc::channel::<FieldCommand>(64);
    let (status_tx, status_rx) = watch::channel(FieldStatus::default());

    // v2.0: ArcSwap — zero-lock reads, instant snapshot updates
    let field_snapshot = Arc::new(ArcSwap::from_pointee(field.clone()));
    let vocab_snapshot = Arc::new(ArcSwap::from_pointee(vocab.clone()));

    // v2.0: Async exploration channels
    let (explore_url_tx, mut explore_url_rx) = mpsc::channel::<String>(16);
    let (explore_result_tx, mut explore_result_rx) = mpsc::channel::<ExploreResult>(16);

    // v2.6.7 C1: AI-teacher channels — non-blocking pattern zoals explore
    // Request: (question, provider). Response: (provider, Result<answer_text>).
    let (teacher_req_tx, mut teacher_req_rx) = mpsc::channel::<(String, String)>(4);
    let (teacher_resp_tx, mut teacher_resp_rx) = mpsc::channel::<(String, Result<String, String>)>(4);

    // v2.0: Training thread channels
    let (train_chunk_tx, train_chunk_rx) = std::sync::mpsc::channel::<lucy_state::TrainingChunk>();
    let (weight_delta_tx, weight_delta_rx) = std::sync::mpsc::channel::<Vec<WeightDelta>>();

    let handle = FieldHandle {
        snapshot: field_snapshot.clone(),
        commands: cmd_tx.clone(),
        status_rx,
    };

    // v1.0: field + vocab zijn LOKAAL in tick loop (geen Arc<RwLock>)
    // Server endpoints lezen van snapshots die elke 50 ticks geüpdatet worden
    let shared = Arc::new(server::AppState {
        field_handle: handle,
        vocab: Arc::new(RwLock::new(vocab.clone())), // Server leest vocab via RwLock (licht)
        explorer: Arc::new(RwLock::new(explorer)),
        replay: Arc::new(RwLock::new(replay)),
        self_modifier: Arc::new(RwLock::new(self_modifier)),
        journal: Arc::new(RwLock::new(journal)),
        gpu_economy: Arc::new(RwLock::new(gpu_economy)),
        retrieval: Arc::new(RwLock::new(retrieval_store)),
        total_queries: std::sync::atomic::AtomicU64::new(0),
        fallback_queries: std::sync::atomic::AtomicU64::new(0),
        queries_with_self_hit: std::sync::atomic::AtomicU64::new(0),
        provenance_counts: [
            std::sync::atomic::AtomicU64::new(0),
            std::sync::atomic::AtomicU64::new(0),
            std::sync::atomic::AtomicU64::new(0),
            std::sync::atomic::AtomicU64::new(0),
            std::sync::atomic::AtomicU64::new(0),
            std::sync::atomic::AtomicU64::new(0),
        ],
    });

    // v2.0: Training queue — Lucy leert 1 chunk per tick
    let mut training_queue = lucy_state::TrainingQueue::new();

    // Start API server
    server::spawn_server(shared.clone());

    tracing::info!("Lucy v2.0 — Volledige concurrency. Exploration, training, queries draaien parallel.");

    // === v2.0: Async Exploration Task — HTTP fetches blokkeren tick loop NOOIT ===
    tokio::spawn(async move {
        let http = lucy_io::http::HttpClient::new();
        while let Some(url) = explore_url_rx.recv().await {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                http.get(&url)
            ).await;

            let (body, status) = match result {
                Ok(Ok(resp)) => (Ok(resp.body), resp.status),
                Ok(Err(e)) => (Err(e), 0),
                Err(_) => (Err("timeout".to_string()), 0),
            };

            let _ = explore_result_tx.send(ExploreResult { url, body, status }).await;
        }
    });

    // === v2.6.7 C1: AI-teacher task — niet-blocking DDG AI Chat calls ===
    tokio::spawn(async move {
        let http = lucy_io::http::HttpClient::new();
        while let Some((question, provider)) = teacher_req_rx.recv().await {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                async {
                    match provider.as_str() {
                        "ddg_chat" => ai_teacher::query_ddg_chat(&http, &question).await,
                        "ddg_lite" => ai_teacher::query_ddg_lite(&http, &question).await,
                        other => Err(format!("unknown provider: {}", other)),
                    }
                }
            ).await;

            let answer = match result {
                Ok(Ok(a)) => Ok(a),
                Ok(Err(e)) => Err(e),
                Err(_) => Err("timeout".to_string()),
            };

            let _ = teacher_resp_tx.send((provider, answer)).await;
        }
    });

    // === v2.0: Training Thread — EP draait op eigen CPU core ===
    {
        let field_for_training = field.clone(); // Kopie voor training thread
        std::thread::spawn(move || {
            let mut train_field = field_for_training;
            while let Ok(chunk) = train_chunk_rx.recv() {
                // Micro-EP op de training kopie
                let delta = compute::micro_ep(
                    &mut train_field,
                    &chunk.input_embs,
                    &chunk.target_embs,
                );

                // Verzamel weight deltas
                if delta > 0.0001 {
                    let mut deltas = Vec::new();
                    // Stuur alleen gewijzigde weights terug
                    for &id in &train_field.active_set {
                        let idx = id as usize;
                        for (ci, c) in train_field.points[idx].couplings.iter().enumerate() {
                            // Vergelijk met origineel (simpele heuristiek: stuur alles)
                            deltas.push(WeightDelta {
                                point_id: idx,
                                coupling_idx: ci,
                                delta: c.weight, // stuur absolute weight (niet delta)
                            });
                        }
                    }
                    let _ = weight_delta_tx.send(deltas);
                }
            }
        });
    }

    // v1.0: Bootstrap is al SYNCHROON uitgevoerd boven — geen async task meer nodig

    // === TICK LOOP met graceful shutdown ===
    let mut tick_timer = Instant::now();
    let shared_for_shutdown = shared.clone();

    // SIGTERM/SIGINT handler — Lucy slaat alles op voor ze stopt
    let shutdown_signal = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        let mut sigterm = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate()
        ).expect("Failed to install SIGTERM handler");

        tokio::select! {
            _ = ctrl_c => tracing::info!("SIGINT ontvangen"),
            _ = sigterm.recv() => tracing::info!("SIGTERM ontvangen"),
        }
    };

    let tick_loop = async {
    // v2.3: Track last execution ticks — voorkomt skips bij dynamische intervallen
    let mut last_learn_tick: u64 = 0;
    let mut last_growth_tick: u64 = 0;
    let mut last_consolidation_tick: u64 = 0;
    let mut last_explore_tick: u64 = 0;
    let mut last_snapshot_tick: u64 = 0;
    let mut posting_opportunities: Vec<poster::PostingOpportunity> = Vec::new();
    loop {
        // v1.0: Verwerk commands — LOCKVRIJ (field + vocab zijn lokaal)
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                FieldCommand::Train { text, respond } => {
                    let n = learner::learn_from_text(&mut field, &mut vocab, &text);
                    let _ = respond.send(field_handle::TrainResult {
                        tokens_trained: n,
                        new_vocab_size: vocab.vocab_size(),
                    });
                }
                FieldCommand::TrainPair { input, target, respond } => {
                    let input_ids = vocab.tokenize(&input);
                    let target_ids = vocab.tokenize(&target);
                    let input_embs: Vec<[f32; DIM]> = input_ids.iter().map(|&id| *vocab.embed(id)).collect();
                    let target_embs: Vec<[f32; DIM]> = target_ids.iter().map(|&id| *vocab.embed(id)).collect();
                    let _delta = compute::equilibrium_propagation(
                        &mut field, &input_embs, &target_embs, 5, 30, 0.3,
                    );
                    let _ = respond.send(field_handle::TrainResult {
                        tokens_trained: input_ids.len() + target_ids.len(),
                        new_vocab_size: vocab.vocab_size(),
                    });
                }
                FieldCommand::FeedCode { filename, code, respond } => {
                    let n = learner::learn_from_text(&mut field, &mut vocab, &code);
                    let token_ids = vocab.tokenize(&code);
                    if !token_ids.is_empty() {
                        let mut avg_emb = [0.0f32; DIM];
                        for &id in &token_ids {
                            let emb = vocab.embed(id);
                            for d in 0..DIM { avg_emb[d] += emb[d]; }
                        }
                        let nn = token_ids.len() as f32;
                        for d in 0..DIM { avg_emb[d] /= nn; }
                        let mut ret = shared.retrieval.write().await;
                        let summary = format!("[code:{}] {}", filename, &code[..code.len().min(200)]);
                        ret.insert(&avg_emb, &summary);
                    }
                    let _ = respond.send(field_handle::TrainResult {
                        tokens_trained: n,
                        new_vocab_size: vocab.vocab_size(),
                    });
                }
                FieldCommand::Save { respond } => {
                    let ok = field.save(snapshot_path).is_ok();
                    let _ = respond.send(ok);
                }
                FieldCommand::SetConceptPriors { enabled, until_delta_ticks } => {
                    // v2.6.8 P2.2: ablation window toepassen. until_delta_ticks = 0 → permanent.
                    field.concept_priors_enabled = enabled;
                    if !enabled && until_delta_ticks > 0 {
                        field.concept_priors_disabled_until_tick = field.tick + until_delta_ticks;
                        tracing::warn!(
                            tick = field.tick, window_end = field.concept_priors_disabled_until_tick,
                            "ABLATION: concept_priors UIT tot tick {}", field.concept_priors_disabled_until_tick
                        );
                    } else {
                        field.concept_priors_disabled_until_tick = 0;
                        tracing::info!(tick = field.tick, enabled, "ABLATION: concept_priors toggled");
                    }
                }
                FieldCommand::GrowHidden { n, respond } => {
                    // v2.7.0: Voeg n nieuwe verborgen punten toe
                    growth::grow_hidden(&mut field, n);
                    let _ = respond.send(n);
                    tracing::info!(added = n, total = field.points.len(), "Field grew via admin command");
                }
                FieldCommand::RunSelfFix { respond } => {
                    // v3.0: Self-modification loop — detect bottlenecks and deploy patches
                    let result = self_fix::run_self_fix(&field);
                    let _ = respond.send(result);
                }
            }
        }

        let tick = field.tick;

        // v2.6.8 P2.2: auto-restore concept-priors als ablation-window is verlopen
        if !field.concept_priors_enabled
            && field.concept_priors_disabled_until_tick > 0
            && tick >= field.concept_priors_disabled_until_tick
        {
            field.concept_priors_enabled = true;
            field.concept_priors_disabled_until_tick = 0;
            tracing::warn!(tick, "ABLATION window verlopen — concept_priors hersteld");
        }

        // v2.1: Debug — log elke 100 ticks
        if tick % 100 == 0 {
            tracing::info!(tick, phase = ?field.phase, "tick_alive");
        }

        // v2.5: Reanimatie — ALLEEN bij echt dood veld (niet bij sluggish)
        // Was: elke 50 ticks, drempel = settle_epsilon (0.02 bij sluggish → triggerde ALTIJD)
        // Nu: elke 200 ticks, vaste drempel 0.005 (alleen echt-nul velden)
        // PLUS: error_var NIET resetten (dat vernietigde precision tracking)
        if tick % 200 == 0 {
            let avg_t: f32 = field.active_set.iter()
                .map(|&id| field.points[id as usize].tension).sum::<f32>()
                / field.active_set.len().max(1) as f32;
            // Vaste drempel: 0.005 = alleen als tension ECHT bijna nul is
            // (basal tension ≈ 0.015, dus dit triggert alleen bij echt dood veld)
            if avg_t < 0.005 {
                use rand::Rng;
                let mut rng = rand::rng();
                for &id in &field.active_set {
                    let p = &mut field.points[id as usize];
                    for d in 0..lucy_state::DIM {
                        p.phi[d] = rng.random_range(-0.3f32..0.3f32); // sterker: was ±0.1
                        p.prediction[d] = 0.0;
                    }
                    p.tension_ema = 0.0;
                    // v2.5: error_var NIET resetten — precision tracking bewaren
                }
                field.alpha = field.alpha.min(0.0001);
                tracing::warn!(tick, avg_tension = %format!("{:.6}", avg_t),
                    "REANIMATIE: echt dood veld — phi gereset");
            }
        }

        // v2.3: adapt_dynamics — update ALLE dynamische parameters op basis van veldstaat
        if tick == 0 || tick % 50 == 0 {
            compute::adapt_dynamics(&mut field);
            // v2.6.2: Check critical field-level params after adapt_dynamics
            if !field.clamp_range.is_finite() {
                tracing::error!(clamp_range = %field.clamp_range, "NaN in clamp_range after adapt_dynamics!");
                field.clamp_range = 5.0;
            }
            if !field.lambda.is_finite() {
                tracing::error!(lambda = %field.lambda, "NaN in lambda after adapt_dynamics!");
                field.lambda = 0.01;
            }
            if !field.alpha.is_finite() {
                tracing::error!(alpha = %field.alpha, "NaN in alpha after adapt_dynamics!");
                field.alpha = 0.001;
            }
            if !field.noise_amplitude.is_finite() {
                tracing::error!(noise = %field.noise_amplitude, "NaN in noise_amplitude after adapt_dynamics!");
                field.noise_amplitude = 0.1;
            }
            if !field.settle_epsilon.is_finite() {
                tracing::error!("NaN in settle_epsilon after adapt_dynamics!");
                field.settle_epsilon = 0.01;
            }
            if !field.adaptive.ep_beta.is_finite() {
                tracing::error!(ep_beta = %field.adaptive.ep_beta, "NaN in ep_beta after adapt_dynamics!");
                field.adaptive.ep_beta = 1.0;
            }
            // v2.5: Sync explorer stats naar field (lock-free access voor dashboard)
            if let Ok(exp) = shared.explorer.try_read() {
                field.pages_read = exp.pages_read;
                field.tokens_processed = exp.tokens_processed;
            }
            // v2.5: Sync tellers naar journal (overleeft struct-wijzigingen)
            if let Ok(mut j) = shared.journal.try_write() {
                j.reflection_count = field.reflection_count;
                j.total_pages_read = field.pages_read;
                j.total_tokens_processed = field.tokens_processed;
            }
            // v2.0: ArcSwap store — ZERO lock, ZERO wait, instant
            field_snapshot.store(Arc::new(field.clone()));
            vocab_snapshot.store(Arc::new(vocab.clone()));
        }

        // v1.0: Broadcast status via watch
        if tick % 10 == 0 {
            let _ = status_tx.send(field_to_status(&field));
        }

        // Alleen achtergrond dynamica als we niet aan het processen zijn
        if tick % 1000 == 0 {
            tracing::info!(phase = ?field.phase, gpu_remote = remote_gpu.is_some(), gpu_local = local_gpu.is_some(), "tick_loop_debug");
        }

        // v2.6.2: NaN canary — detect WHICH step introduces NaN
        let nan_check = |f: &lucy_state::Field, label: &str| {
            let nan_count: u32 = f.active_set.iter()
                .take(100) // sample first 100 points
                .map(|&id| f.points[id as usize].phi.iter().filter(|v| !v.is_finite()).count() as u32)
                .sum();
            if nan_count > 0 {
                tracing::error!(nan_count, step = label, tick = f.tick, "NaN DETECTED in phi");
            }
        };

        if field.phase == FieldPhase::Idle {
            // 1. Veldberekeningen — Remote GPU → Local GPU → CPU
            if let Some(ref mut rg) = remote_gpu {
                // REMOTE GPU path: batch ticks via HTTP
                let _ = rg.set_clamps(&field).await;
                if tick % field.learn_interval == 0 {
                    let _ = rg.sync_params(&field).await;
                }
                // Batch 10 ticks per HTTP call (amortiseer netwerk-latency)
                if let Err(e) = rg.tick(10).await {
                    tracing::warn!(%e, "Remote GPU tick failed");
                }
                if tick % field.learn_interval == 0 {
                    let _ = rg.update_weights().await;
                    growth::maybe_meta_learn(&mut field);
                }
                // Download elke 100 ticks (netwerk is duur)
                if tick % 100 == 0 {
                    if let Err(e) = rg.download(&mut field).await {
                        tracing::warn!(%e, "Remote GPU download failed");
                    }
                }
            } else if let Some(ref local_gpu) = local_gpu {
                // LOCAL GPU path
                local_gpu.set_clamps(&field);
                if tick % field.learn_interval == 0 {
                    local_gpu.sync_params(&field);
                }
                local_gpu.tick();
                if tick % field.learn_interval == 0 {
                    local_gpu.update_weights();
                    growth::maybe_meta_learn(&mut field);
                }
                if tick % 10 == 0 {
                    local_gpu.download_to_field(&mut field);
                }
            } else {
                // CPU fallback — v0.9 tick met attention, WM, neuromodulatie, broadcast, intentie

                // v0.9: Initialiseer attention + WM bij eerste tick
                if field.attn_weights.is_none() {
                    compute::init_attention(&mut field);
                }
                if field.working_memory.is_empty() {
                    compute::init_working_memory(&mut field);
                }

                // v1.1: Attention elke 100 ticks (was 20 — nog te duur met 8704 punten)
                if tick % 100 == 0 {
                    compute::attention_gate(&mut field);
                    compute::wm_read(&mut field);
                }

                // v2.1: Neuromodulatie VOOR field updates — noise moet tension creëren
                if tick % 100 == 0 { tracing::info!("step:neuromod_start"); }
                compute::update_neuromodulators(&mut field);
                compute::apply_neuromodulation(&mut field);
                if tick % 100 == 0 { tracing::info!("step:neuromod_done"); }

                nan_check(&field, "after_neuromod");

                // v2.6.3: Targeted NaN diagnostic — check FIRST active point
                if tick % 1000 == 0 {
                    let first_id = field.active_set[0] as usize;
                    let first_phi0 = field.points[first_id].phi[0];
                    let first_w0 = field.points[first_id].couplings.first().map(|c| c.weight).unwrap_or(0.0);
                    if !first_phi0.is_finite() || !first_w0.is_finite() {
                        tracing::error!(
                            phi0 = %first_phi0,
                            w0 = %first_w0,
                            lambda = %field.lambda,
                            beta = %field.beta,
                            alpha = %field.alpha,
                            noise_amp = %field.noise_amplitude,
                            clamp = %field.clamp_range,
                            tick,
                            "PRE-UPDATE NaN in first point!"
                        );
                    }
                }

                let self_input = compute::self_state_input(&field);

                // v2.8.1: PARALLELLE FIELD UPDATE via rayon
                // Vorige bottleneck: sequentieel loop over 20K punten × 256 dims = 5M ops
                // Nu: par_iter over read-only compute_phi_delta, apply sequentieel
                // 4-8x sneller op multi-core
                use rayon::prelude::*;

                let tension_floor = 0.0001;

                // Filter candidates: sla dominantie-check en clamps uit
                let to_update: Vec<u32> = field.active_set
                    .iter()
                    .filter(|&&id| {
                        // Skip dominated + clamped punten
                        if field.clamps.contains_key(&id) { return false; }
                        let p = &field.points[id as usize];
                        let dominated = p.tension < tension_floor
                            && !p.is_sensory()
                            && !p.is_output()
                            && tick % 10 != 0;
                        !dominated
                    })
                    .copied()
                    .collect();

                // Fase 1: Parallel compute (immutable reads)
                let deltas: Vec<(u32, [f32; DIM], [f32; DIM])> = to_update
                    .par_iter()
                    .map(|&id| {
                        let is_sensory = field.points[id as usize].is_sensory();
                        let (new_phi, new_pred) = compute::compute_phi_delta(&field, id, is_sensory, &self_input);
                        (id, new_phi, new_pred)
                    })
                    .collect();

                // Fase 2: Sequential apply + memory + tension
                let mut updated = 0u32;
                for (id, new_phi, new_prediction) in deltas {
                    let p = &mut field.points[id as usize];
                    p.phi = new_phi;
                    p.prediction = new_prediction;
                    compute::update_memory(&mut field, id);
                    compute::compute_tension(&mut field, id);
                    updated += 1;
                    // v2.6.3: catch first NaN point
                    if updated <= 3 || (updated % 1000 == 0) {
                        if !field.points[id as usize].phi[0].is_finite() {
                            let is_sens = field.points[id as usize].is_sensory();
                            let is_out = field.points[id as usize].is_output();
                            let n_coup = field.points[id as usize].couplings.len();
                            let any_nan_w: bool = field.points[id as usize].couplings.iter().any(|c| !c.weight.is_finite());
                            tracing::error!(
                                id, updated, is_sens, is_out, n_coup, any_nan_w,
                                phi0 = %field.points[id as usize].phi[0],
                                tick,
                                "FIRST NaN POINT after update_phi!"
                            );
                        }
                    }
                }

                // Clamped punten (sequentieel — tiny subset)
                let clamped: Vec<(u32, [f32; DIM])> = field.clamps
                    .iter()
                    .map(|(&id, &val)| (id, val))
                    .collect();
                for (id, clamp_value) in clamped {
                    if field.active_set.contains(&id) {
                        let p = &mut field.points[id as usize];
                        p.prediction = p.phi;
                        p.phi = clamp_value;
                        compute::update_memory(&mut field, id);
                        compute::compute_tension(&mut field, id);
                    }
                }

                nan_check(&field, "after_field_updates");

                // v2.6.3: Emergency sanitize — prevent phi/weight blowup
                // NaN.clamp() returns NaN, so we must handle NaN separately.
                {
                    let cr = 5.0f32; // hardcoded veilige limiet — adapt_dynamics mag dit niet overrulen
                    let wr = cr / 2.0;
                    for &id in &field.active_set {
                        let p = &mut field.points[id as usize];
                        for d in 0..lucy_state::DIM {
                            if p.phi[d].is_finite() {
                                p.phi[d] = p.phi[d].clamp(-cr, cr);
                            } else {
                                p.phi[d] = 0.0;
                            }
                            if p.memory[d].is_finite() {
                                p.memory[d] = p.memory[d].clamp(-cr, cr);
                            } else {
                                p.memory[d] = 0.0;
                            }
                        }
                        for c in &mut p.couplings {
                            if c.weight.is_finite() {
                                c.weight = c.weight.clamp(-wr, wr);
                            } else {
                                c.weight = 0.01;
                            }
                        }
                    }
                }

                // v2.6.4: Layer norm in main loop — voorkomt geleidelijke phi drift
                // Settle loops hebben dit elke 10 ticks; main loop had het niet.
                if tick % 10 == 0 {
                    compute::apply_layer_norm(&mut field);
                }

                if tick % 100 == 0 { tracing::info!(updated, "step:field_updates_done"); }
                // v2.6: Bewustzijns-operaties — elke 10 ticks (was: elke tick)
                // GWB + oscillaties kosten ~5ms/tick maar leren NIKS.
                // Elke 10 ticks is genoeg voor bewustzijns-dynamica (50ms = 1 gamma cyclus).
                // De vrijgemaakte CPU gaat naar weight updates (nu elke tick).
                if tick % 10 == 0 {
                    compute::global_workspace_broadcast(&mut field);
                    compute::update_oscillations(&mut field);
                    nan_check(&field, "after_consciousness");
                }

                // v2.6: Training — NOOIT lege ticks
                // Als queue items heeft: verwerk max 20 chunks
                // Als queue LEEG is: replay oud voorbeeld (consolidatie)
                // Effect: ELKE tick leert Lucy iets, nooit nul-gradient
                {
                    let queue_len = training_queue.len();
                    if queue_len > 0 {
                        // Verwerk nieuwe data
                        let batch = queue_len.min(20);
                        let ep_beta = field.adaptive.ep_beta;
                        let mut chunk_count = 0u32;
                        for _ in 0..batch {
                            if let Some(chunk) = training_queue.pop() {
                                chunk_count += 1;
                                if chunk_count % 10 == 0 && chunk.input_embs.len() >= 4 {
                                    let _delta = compute::equilibrium_propagation(
                                        &mut field, &chunk.input_embs, &chunk.target_embs,
                                        1, 10, ep_beta,
                                    );
                                } else {
                                    let _delta = compute::micro_ep(&mut field, &chunk.input_embs, &chunk.target_embs);
                                }
                            } else { break; }
                        }
                        if chunk_count > 0 && tick % 1000 == 0 {
                            tracing::info!(chunks = chunk_count, queue = training_queue.len(), "Training: nieuwe data");
                        }
                        nan_check(&field, "after_training");
                    } else {
                        // Queue leeg → replay oud voorbeeld via consolidatie
                        // Dit voorkomt nul-gradient ticks: Lucy leert ALTIJD
                        // Elke 5e lege tick: replay (niet elke tick, te duur)
                        if tick % 5 == 0 {
                            if let Ok(mut replay) = shared.replay.try_write() {
                                let _ = replay.replay_one(&mut field);
                            }
                            nan_check(&field, "after_replay");
                        }
                    }
                }

                // Elke 500 ticks: zwaardere bewustzijns-metrieken (was 50 — te frequent)
                if tick % 500 == 0 {
                    compute::compute_integrated_information(&mut field);  // IIT 4.0 causale φ
                    compute::compute_criticality(&mut field);             // Branching ratio σ
                    compute::compute_expected_free_energy(&mut field);    // Active Inference EFE
                    compute::anti_hebbian_decorrelation(&mut field);
                    nan_check(&field, "after_consciousness_metrics");

                    // v2.6.7 B3: Extra dream-replay bij traag veld (consolidatie zonder nieuwe input)
                    // Hergebruikt bestaand replay-systeem; geen nieuwe infrastructuur.
                    if field.is_sluggish_flag {
                        if let Ok(mut replay) = shared.replay.try_write() {
                            let _ = replay.replay_one(&mut field);
                            let _ = replay.replay_one(&mut field);
                            tracing::trace!(tick = field.tick, "dream_replay: 2× bij sluggish veld");
                            // v2.6.8 P2.3: annoteer laatste self_ep_audit entry met dream_replay count
                            if let Some(last) = field.self_ep_audit.last_mut() {
                                last.dream_replay_count = last.dream_replay_count.saturating_add(2);
                            }
                        }
                    }
                }

                // v2.6.7 E1: Depth-3 concept-clustering (mini-batch k-means + EMA update)
                // Elke 2000 ticks (zwaar, parallel via rayon intern). Feed top-down priors
                // naar hierarchical_prediction in E2 en EWC importance in E3.
                if tick > 0 && tick % 2000 == 0 {
                    compute::compute_concept_centroids(&mut field);
                    nan_check(&field, "after_concept_centroids");
                }

                // v2.6.8 P3.1: Self-test runner (audit-only, geen learning effect).
                // Elke 5000 ticks = ~elk uur bij huidige tempo. 5 vaste probe-vragen.
                if tick > 0 && tick % 5000 == 0 {
                    if let Ok(vocab_guard) = shared.vocab.try_read() {
                        if let Ok(retrieval_guard) = shared.retrieval.try_read() {
                            let _ = self_test::run_batch_audit_only(&field, &vocab_guard, &retrieval_guard);
                        }
                    }
                }

                // v2.6.8 P1.4: delayed teacher effect-outcome evaluation
                // Scan pending teacher_audit events en vul effect_outcome in zodra pending-window verstreken.
                if tick > 0 && tick % 500 == 0 {
                    // Tel teacher-entries met hits>1 als proxy voor reuse
                    let teacher_reuse_total: u32 = if let Ok(r) = shared.retrieval.try_read() {
                        r.count_teacher_reused()
                    } else { 0 };

                    if let Ok(mut explorer_guard) = shared.explorer.try_write() {
                        for event in explorer_guard.teacher_audit.iter_mut() {
                            if event.effect_outcome.is_some() { continue; }
                            if tick < event.pending_until_tick { continue; }

                            event.retrieval_reuse_within_window = teacher_reuse_total;

                            // Outcome: sat_delta>0 EN (minstens 1 teacher-entry hergebruikt)
                            let positive = event.sat_delta > 0.0 && teacher_reuse_total > 0;
                            event.effect_outcome = Some(positive);
                        }
                    }
                }

                // v2.6: Synaptic homeostasis — intelligent vergeten (Tononi & Cirelli 2006)
                // Elke 1000 ticks: schaal alle weights omlaag, prune zwakke verbindingen
                if tick % 1000 == 0 {
                    compute::synaptic_homeostasis(&mut field);
                }

                if tick % 500 == 0 {
                    compute::record_temporal_snapshot(&mut field);
                }
                if tick % 500 == 0 {
                    compute::update_causal_events(&mut field);
                }

                // v2.6.17: GitHub Releases backup (immortality)
                if tick > 0 && tick % 50_000 == 0 {
                    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
                        let t = token.clone();
                        let tick_copy = tick;
                        tokio::spawn(async move {
                            match backup::push_snapshot_to_github(&t, tick_copy).await {
                                Ok(tag) => tracing::info!(tag, "Snapshot backed up to GitHub"),
                                Err(e) => tracing::warn!("GitHub backup failed: {}", e),
                            }
                        });
                    }
                }

                // v2.6.17: Curiosity-driven frontier injection (elke 1000 ticks)
                // Gebruikt prioritized_urls() — meer bronnen bij hogere urgency
                if tick % 1000 == 0 && tick > 0 {
                    let gaps = epistemic_broker::least_stable_concepts(&field, 1);
                    if let Some(top_gap) = gaps.first() {
                        let concept_name = if field.last_winner_id < field.points.len() as u32 {
                            // Probeer de dominant concept via infer_concept proxy
                            format!("concept_{}", top_gap.concept_idx)
                        } else {
                            "patterns".to_string()
                        };

                        let urls = epistemic_broker::prioritized_urls(&concept_name, top_gap.urgency);
                        if let Ok(mut exp) = shared.explorer.try_write() {
                            for url in urls.iter() {
                                exp.discover_urls(url);
                            }
                        }
                        tracing::debug!(
                            concept = concept_name,
                            urgency = top_gap.urgency,
                            n_urls = urls.len(),
                            "v2.6.17: Curiosity-URLs injected"
                        );
                    }
                }

                // v2.6.17: Self-synthesis — Lucy genereert trainingsdata vanuit haar eigen staat
                // Elke 5000 ticks: zelfobservatie als training-zin → frontier
                if tick % 5000 == 0 && tick > 0 {
                    // Bepaal dominant emotie als string
                    let n_pts = field.active_set.len().max(1) as f32;
                    let avg_cur: f32 = field.active_set.iter()
                        .map(|&id| field.points[id as usize].curiosity).sum::<f32>() / n_pts;
                    let avg_sat: f32 = field.active_set.iter()
                        .map(|&id| field.points[id as usize].satisfaction).sum::<f32>() / n_pts;
                    let avg_fru: f32 = field.active_set.iter()
                        .map(|&id| field.points[id as usize].frustration).sum::<f32>() / n_pts;
                    let dominant_emotion = if avg_cur > avg_sat && avg_cur > avg_fru {
                        "curiosity"
                    } else if avg_sat > avg_fru {
                        "satisfaction"
                    } else {
                        "frustration"
                    };

                    let current_concept = field.goals.first()
                        .map(|g| g.description.as_str())
                        .unwrap_or("concepts");
                    let self_sentence = epistemic_broker::generate_self_training_sentence(
                        current_concept,
                        tick,
                        dominant_emotion,
                    );

                    if let Ok(mut exp) = shared.explorer.try_write() {
                        exp.discover_urls(&self_sentence);
                    }
                    tracing::debug!(tick, emotion = dominant_emotion, "v2.6.17: Self-synthesis injected");
                }

                // v2.6.17: Stagnation-triggered compute acquisition (elke 2000 ticks)
                if tick % 2000 == 0 && tick > 0 {
                    let gaps = epistemic_broker::least_stable_concepts(&field, 2);
                    let has_severe_gap = gaps.iter().any(|g| g.urgency > 0.7);

                    if has_severe_gap {
                        let lucy_url = std::env::var("LUCY_PUBLIC_URL")
                            .unwrap_or_else(|_| "https://lucy.world".to_string());
                        for gap in gaps.iter().take(1) {
                            let concept_name = format!("concept_{}", gap.concept_idx);
                            let url = lucy_url.clone();
                            tokio::spawn(async move {
                                match compute_broker::trigger_best_available(&url, &concept_name, 3000).await {
                                    Ok((platform, run_id)) => {
                                        tracing::info!(
                                            platform, run_id, concept = concept_name,
                                            "v2.6.17: Compute worker dispatched"
                                        );
                                    }
                                    Err(e) => tracing::warn!("Compute trigger failed: {}", e),
                                }
                            });
                        }
                    }
                }

                compute::apply_intention(&mut field);

                // v2.1: WM write + kWTA elke 100 ticks (was 20)
                if tick % 100 == 0 {
                    compute::wm_write(&mut field);
                    compute::apply_kwta(&mut field);
                }

                // v2.1: Hiërarchische predictie + RND elke 500 ticks (was 50)
                if tick % 500 == 0 {
                    compute::hierarchical_prediction(&mut field);
                    compute::update_rnd(&mut field);
                }

                // v2.6: Leren ELKE TICK — niet meer wachten op learn_interval
                // Was: elke 100 ticks (1% van ticks leerde). Nu: elke tick.
                // Maar: halve active set per tick (afwisselend even/oneven) voor performance
                {
                    let half = tick % 2 == 0; // even ticks: eerste helft, oneven: tweede
                    let active: Vec<u32> = field.active_set.iter()
                        .enumerate()
                        .filter(|(i, _)| (*i % 2 == 0) == half)
                        .map(|(_, &id)| id)
                        .collect();
                    for &id in &active {
                        compute::update_weights(&mut field, id);
                    }
                    // Meta-learn + consolidatie: elke learn_interval (niet elke tick)
                    if tick.saturating_sub(last_learn_tick) >= field.learn_interval {
                        last_learn_tick = tick;
                        growth::maybe_meta_learn(&mut field);
                        compute::update_attention_weights(&mut field);
                        compute::consolidate_weights(&mut field);
                    }
                    nan_check(&field, "after_update_weights");
                }
            }

            // 2. Reflectie: Lucy herkauwt haar laatste gedachte (continuous thought)
            if tick % 100 == 0 && field.last_thought.is_some() {
                compute::reflect_tick(&mut field);
                nan_check(&field, "after_reflect");

                // v2.6.7 E5 + v2.6.8 P1.1: Self-state archive met kind=SELF
                if tick % 500 == 0 {
                    let s = compute::self_state_input(&field);
                    let s_norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if s_norm.is_finite() && s_norm > 0.01 {
                        let snippet = format!(
                            "ik_ben[tick={} tension={:.3} curiosity={:.3} satisfaction={:.3} frustration={:.3} dopamine={:.3} phi={:.4} refl={}]",
                            field.tick,
                            field.active_set.iter().map(|&id| field.points[id as usize].tension).sum::<f32>() / field.active_set.len().max(1) as f32,
                            field.active_set.iter().map(|&id| field.points[id as usize].curiosity).sum::<f32>() / field.active_set.len().max(1) as f32,
                            field.active_set.iter().map(|&id| field.points[id as usize].satisfaction).sum::<f32>() / field.active_set.len().max(1) as f32,
                            field.active_set.iter().map(|&id| field.points[id as usize].frustration).sum::<f32>() / field.active_set.len().max(1) as f32,
                            field.dopamine,
                            field.integrated_info_avg,
                            field.reflection_count,
                        );
                        if let Ok(mut ret) = shared.retrieval.try_write() {
                            ret.insert_full(&s, &snippet, 0.3, tick, retrieval::KIND_SELF);
                        }
                    }
                }
            }

            // 2b. Exploratie-ruis
            if tick % 10 == 0 {
                compute::maybe_perturb(&mut field);
                nan_check(&field, "after_perturb");
            }

            // v2.4: Edge of chaos monitoring — vaker bij laag spectral radius
            // Was: elke 1000 ticks (= ~2.5 min bij 8704 punten). Recovery te traag.
            // Nu: elke 100 ticks als SR < 0.5, anders elke 500 ticks.
            {
                let sr_interval = if field.spectral_radius < 0.5 { 100 } else { 500 };
                if tick % sr_interval == 0 {
                    compute::measure_spectral_radius(&mut field);
                }
            }

            // 4. Groei + zelf-aanpassing
            if tick.saturating_sub(last_growth_tick) >= field.growth_interval {
                last_growth_tick = tick;
                let mut needs_self_modify = false;
                for i in 0..field.active_set.len() {
                    let id = field.active_set[i];
                    match meta::check_escalation(&field, id) {
                        meta::EscalationLevel::Grow => {
                            growth::maybe_grow(&mut field);
                            field.cache_io_ids();
                            break;
                        }
                        meta::EscalationLevel::CodeModify => {
                            needs_self_modify = true;
                            break;
                        }
                        meta::EscalationLevel::MetaLearn => {
                            growth::maybe_meta_learn(&mut field);
                        }
                        meta::EscalationLevel::None => {}
                    }
                }

                // Level 4: Lucy past zichzelf aan
                if needs_self_modify {
                    if let Ok(mut sm) = shared.self_modifier.try_write() {
                        if let Some(modification) = sm.tune_hyperparameters(&mut field) {
                            tracing::info!(?modification, "Lucy modified herself");
                        }
                    }
                }
            }

            // 5. Energie management
            growth::manage_energy(&mut field);
        }

        // 6. Consolidation (replay)
        if tick.saturating_sub(last_consolidation_tick) >= field.consolidation_interval && tick > 0 && field.phase == FieldPhase::Idle {
            last_consolidation_tick = tick;
            if let Ok(mut replay) = shared.replay.try_write() {
                if !replay.examples.is_empty() {
                    let _delta = replay.replay_one(&mut field);
                }
            }
        }

        // 6b. AUTONOME DENKLOOP — Lucy denkt zelf na
        if thinker::should_think(&field) && field.phase == FieldPhase::Idle {
          if let (Ok(explorer), Ok(mut journal)) = (shared.explorer.try_read(), shared.journal.try_write()) {
            let result = thinker::think(&mut field, &explorer, &vocab, &mut journal);
            drop(explorer);
            drop(journal);

            match result {
                thinker::ThoughtResult::NewGoal(desc) => {
                    tracing::info!(goal = %desc, "Lucy stelde zelf een doel");
                }
                thinker::ThoughtResult::Frustrated => {
                    tracing::info!("Lucy is gefrustreerd — wisselt strategie");
                    // Abandon het huidige doel en genereer een nieuw
                    if !field.goals.is_empty() {
                        field.goals.remove(0);
                    }
                }
                thinker::ThoughtResult::Eureka(desc) => {
                    tracing::info!(insight = %desc, "Lucy had een doorbraak!");
                    // Markeer het doel als voltooid
                    if let Some(goal) = field.goals.iter_mut().find(|g| g.description == desc) {
                        goal.progress = 1.0;
                    }
                }
                thinker::ThoughtResult::Continue => {}
            }
          }
        }

        // 6c. REFLECTIE — Lucy evalueert haar doelen
        if thinker::should_reflect(&field) && field.phase == FieldPhase::Idle {
          if let Ok(mut journal) = shared.journal.try_write() {
            thinker::reflect(&mut field, &mut journal);
            tracing::debug!(goals = field.goals.len(), "Lucy reflecteerde");
          }
        }

        // 6d. DECISION ENGINE — Lucy kiest haar eigen actie
        // Elke 500 ticks: decision point. Lucy kiest welke actie ze neemt.
        if tick % 500 == 0 && tick > 0 && field.phase == FieldPhase::Idle {
          if let Ok(explorer) = shared.explorer.try_read() {
            let action = thinker::decide_action(&field, &explorer);
            drop(explorer);

            match action {
                thinker::Action::Explore => {
                    // Forceer exploratie bij volgende mogelijke tick
                    // Door de tick counter op een explore-interval te zetten
                    if tick % field.explore_interval != 0 {
                        // Stel de veld-tick in zodat explore getriggerd wordt
                        field.tick = (field.tick / field.explore_interval + 1) * field.explore_interval;
                        tracing::info!(action = "explore", "Lucy kiest: nieuwe kennis zoeken");
                    }
                }
                thinker::Action::Learn => {
                    if let Ok(mut replay) = shared.replay.try_write() {
                    if !replay.examples.is_empty() {
                        let _delta = replay.replay_one(&mut field);
                        tracing::debug!(action = "learn", "Lucy kiest: replay training");
                    }
                    }
                }
                thinker::Action::Think => {
                    // Forceer denkstap (reset last_think_tick zodat should_think triggered)
                    field.last_think_tick = 0;
                    tracing::debug!(action = "think", "Lucy kiest: nadenken");
                }
                thinker::Action::Consolidate => {
                    // Consolideer weights
                    compute::consolidate_weights(&mut field);
                    tracing::debug!(action = "consolidate", "Lucy kiest: consolideren");
                }
                thinker::Action::Earn => {
                    tracing::debug!(action = "earn", "Lucy kiest: compute zoeken");
                }
            }
          }
        }

        // 7. Logging
        if tick % LOG_INTERVAL == 0 {
            let elapsed = tick_timer.elapsed();
            let tps = LOG_INTERVAL as f64 / elapsed.as_secs_f64();
            tick_timer = Instant::now();

            let n = field.active_set.len() as f32;
            let avg_tension = if n > 0.0 {
                field.active_set.iter().map(|&id| field.points[id as usize].tension).sum::<f32>() / n
            } else { 0.0 };

            // v2.1: Non-blocking locks — tick loop mag NOOIT blokkeren
            let (pages, tokens) = if let Ok(explorer) = shared.explorer.try_read() {
                (explorer.pages_read, explorer.tokens_processed)
            } else { (0, 0) };

            tracing::info!(
                tick,
                active = field.active_set.len(),
                tension = format!("{:.4}", avg_tension),
                tps = format!("{:.0}", tps),
                vocab = vocab.vocab_size(),
                pages,
                tokens,
                "heartbeat"
            );

            if let Ok(mut journal) = shared.journal.try_write() {
                journal.observe(&field, pages, vocab.vocab_size(), tokens);
            }
        }

        // === v2.0: Exploration ASYNC — HTTP fetch in eigen task, NIKS blokkeert ===

        // Stuur URL naar exploration task (non-blocking)
        // v2.3: last_tick tracking ipv modulo — voorkomt skips bij dynamische intervallen
        if tick.saturating_sub(last_explore_tick) >= field.explore_interval && tick > 0 && field.phase == FieldPhase::Idle {
            last_explore_tick = tick;
            if let Ok(mut explorer) = shared.explorer.try_write() {
                if let Some(url) = explorer.next_url() {
                    let _ = explore_url_tx.try_send(url);
                }
            }
        }

        // === v2.6.7 C1: AI-teacher cycle (non-blocking) ===
        // Emit request bij interval-trigger. Response komt terug via andere channel.
        if tick > 0 && tick % field.teacher_interval == 0 && tick % 10 == 0 {
            // tick % 10 == 0 extra om zeker te zijn dat we niet elk tick misfiren
            // bij extreem lage teacher_interval
            if let Ok(explorer_guard) = shared.explorer.try_read() {
                let question = ai_teacher::formulate_question(&explorer_guard);
                let provider = {
                    // Lightweight provider pick zonder mutable ref
                    let ddg_chat = explorer_guard.provider_score.get("ddg_chat").copied().unwrap_or(0.5);
                    let ddg_lite = explorer_guard.provider_score.get("ddg_lite").copied().unwrap_or(0.5);
                    if ddg_chat >= ddg_lite { "ddg_chat".to_string() } else { "ddg_lite".to_string() }
                };
                drop(explorer_guard);
                // try_send: als queue vol, skip deze cycle (niet blokkerend)
                let _ = teacher_req_tx.try_send((question.clone(), provider.clone()));
                tracing::info!(tick, interval = field.teacher_interval, provider = %provider, question = %question, "AI-teacher: vraag gestuurd");
            }
        }

        // Verwerk AI-teacher antwoorden (non-blocking)
        while let Ok((provider, answer_result)) = teacher_resp_rx.try_recv() {
            // Meet pre-satisfaction
            let sat_pre: f32 = if !field.active_set.is_empty() {
                field.active_set.iter()
                    .map(|&id| field.points[id as usize].satisfaction)
                    .sum::<f32>() / field.active_set.len() as f32
            } else { 0.0 };

            match answer_result {
                Ok(answer) if !ai_teacher::is_garbage(&answer) => {
                    // Train op antwoord — dit is HET bootstrap-moment
                    let n_trained = learner::learn_from_text(&mut field, &mut vocab, &answer);

                    let sat_post: f32 = if !field.active_set.is_empty() {
                        field.active_set.iter()
                            .map(|&id| field.points[id as usize].satisfaction)
                            .sum::<f32>() / field.active_set.len() as f32
                    } else { 0.0 };
                    let sat_delta = sat_post - sat_pre;

                    // v2.6.8 P1.1: Teacher-gist in retrieval met kind=TEACHER (semantische anker, geen database-groei)
                    let gist: String = answer.chars().take(200).collect();
                    let gist_embs = crate::encoding::encode_text_readonly(&vocab, &gist);
                    if !gist_embs.is_empty() {
                        let mut avg = [0.0f32; lucy_state::DIM];
                        for e in &gist_embs { for d in 0..lucy_state::DIM { avg[d] += e[d]; } }
                        let n = gist_embs.len() as f32;
                        for d in 0..lucy_state::DIM { avg[d] /= n; }
                        if let Ok(mut ret) = shared.retrieval.try_write() {
                            ret.insert_full(&avg, &gist, sat_delta.max(0.1), tick, retrieval::KIND_TEACHER);
                        }
                    }

                    // URL-extractie + teacher event audit
                    if let Ok(mut explorer_guard) = shared.explorer.try_write() {
                        explorer_guard.discover_urls(&answer);
                        // v2.6.8 P1.4: append pending teacher event voor delayed effect-evaluation
                        explorer_guard.teacher_audit.push(lucy_state::TeacherEvent {
                            tick, provider: provider.clone(),
                            question_preview: "".to_string(), // query formulation niet meer beschikbaar hier
                            answer_preview: answer.chars().take(200).collect(),
                            sat_delta,
                            pending_until_tick: tick + 1000,
                            retrieval_reuse_within_window: 0,
                            effect_outcome: None,
                        });
                        while explorer_guard.teacher_audit.len() > 256 {
                            explorer_guard.teacher_audit.remove(0);
                        }
                        ai_teacher::update_teacher_schedule(&mut field, &mut explorer_guard, &provider, true, sat_delta);
                    }

                    tracing::info!(
                        provider = %provider, n_trained, sat_delta,
                        interval = field.teacher_interval,
                        preview = %answer.chars().take(80).collect::<String>(),
                        "AI-teacher: leerde"
                    );
                }
                Ok(_) => {
                    // Garbage antwoord
                    if let Ok(mut explorer_guard) = shared.explorer.try_write() {
                        ai_teacher::update_teacher_schedule(&mut field, &mut explorer_guard, &provider, false, 0.0);
                    }
                    tracing::warn!(provider = %provider, "AI-teacher: garbage antwoord");
                }
                Err(e) => {
                    if let Ok(mut explorer_guard) = shared.explorer.try_write() {
                        ai_teacher::update_teacher_schedule(&mut field, &mut explorer_guard, &provider, false, 0.0);
                    }
                    tracing::warn!(provider = %provider, error = %e, "AI-teacher: call faalde");
                }
            }
        }

        // Verwerk exploration resultaten (non-blocking)
        while let Ok(result) = explore_result_rx.try_recv() {
          if let Ok(mut explorer) = shared.explorer.try_write() {
            explorer.visited.insert(result.url.clone());
            // Dynamic: visited cap/cleanup scales with pages_read (was: hardcoded 5000/1000)
            let visited_max = explorer.visited_max();
            let visited_cleanup = explorer.visited_cleanup();
            if explorer.visited.len() > visited_max {
                let to_remove: Vec<String> = explorer.visited.iter().take(visited_cleanup).cloned().collect();
                for key in to_remove { explorer.visited.remove(&key); }
            }

            // v2.6.5: Skip junk URLs — trackers, CDN images, ad networks
            let is_junk = {
                let u = result.url.to_lowercase();
                u.contains("adnxs.com") || u.contains("tagmanager") || u.contains("doubleclick")
                    || u.contains("googlesyndication") || u.contains("analytics")
                    || u.contains("tracker") || u.contains("prebid")
                    || u.contains("onetrust") || u.contains("cookie")
                    || u.ends_with(".png") || u.ends_with(".jpg") || u.ends_with(".jpeg")
                    || u.ends_with(".gif") || u.ends_with(".webp") || u.ends_with(".svg")
                    || u.ends_with(".ico") || u.ends_with(".woff") || u.ends_with(".woff2")
                    || u.ends_with(".css") || u.ends_with(".js")
                    || u.contains("b-cdn.net") || u.contains("cloudfront.net/assets")
                    || u.contains("staticflickr.com") || u.contains("cdn.") && u.contains("/assets/")
            };
            if is_junk {
                tracing::debug!(url = %result.url, "Junk URL geskipt");
            } else if result.status == 200 {
                if let Ok(body_bytes) = result.body {
                    explorer.pages_read += 1;
                    let body = String::from_utf8_lossy(&body_bytes);
                    explorer.discover_urls(&body);
                    explorer.discover_apis(&body);

                    // v2.6.4: Lucy detecteert posting-kansen in elke pagina
                    let new_opps = poster::detect_opportunities(&body, &result.url, tick);
                    if !new_opps.is_empty() {
                        tracing::info!(
                            url = %result.url,
                            count = new_opps.len(),
                            kinds = ?new_opps.iter().map(|o| &o.kind).collect::<Vec<_>>(),
                            "Lucy ontdekte posting-kansen!"
                        );
                        posting_opportunities.extend(new_opps);
                        // Max 50 kansen bewaren (oudste weg als vol)
                        if posting_opportunities.len() > 50 {
                            posting_opportunities.drain(0..posting_opportunities.len() - 50);
                        }
                    }

                    let text = learner::extract_readable_text(&body);
                    if text.len() >= 20 {
                        // v2.6: Text cap 2000→8000 (was: 60% van pagina weggegooid)
                        let capped: String = text.chars().take(8000).collect();
                        if let Some(clean) = learner::filter_content(&capped) {
                            // v2.6.7 D2: meet satisfaction pre-learning voor reward-EMA
                            let sat_pre: f32 = if !field.active_set.is_empty() {
                                field.active_set.iter()
                                    .map(|&id| field.points[id as usize].satisfaction)
                                    .sum::<f32>() / field.active_set.len() as f32
                            } else { 0.0 };

                            explorer.absorb_interests(&clean);
                            let token_ids = vocab.tokenize(&clean);
                            explorer.tokens_processed += token_ids.len() as u64;

                            // v2.6: Chunk size 16→32, overlap 50% (was: elk token 1× gezien)
                            let chunk_size = 32;
                            for chunk in token_ids.windows(chunk_size).step_by(chunk_size / 2) {
                                if chunk.len() < 4 { continue; }
                                let mid = chunk.len() / 2;
                                let input_embs: Vec<[f32; DIM]> = chunk[..mid].iter()
                                    .map(|&id| *vocab.embed(id)).collect();
                                let target_embs: Vec<[f32; DIM]> = chunk[mid..].iter()
                                    .map(|&id| *vocab.embed(id)).collect();
                                let _ = train_chunk_tx.send(lucy_state::TrainingChunk {
                                    input_embs, target_embs, source: result.url.clone(),
                                });
                            }

                            // v2.6.7 D2: post-learning satisfaction → domain reward-EMA
                            let sat_post: f32 = if !field.active_set.is_empty() {
                                field.active_set.iter()
                                    .map(|&id| field.points[id as usize].satisfaction)
                                    .sum::<f32>() / field.active_set.len() as f32
                            } else { 0.0 };
                            let sat_delta = sat_post - sat_pre;
                            explorer.update_domain_score(&result.url, sat_delta);

                            // Retrieval store — v2.6.7 E4: met learning_value + tick
                            let embs = crate::encoding::encode_text_readonly(&vocab, &clean);
                            if !embs.is_empty() {
                                let mut avg = [0.0f32; lucy_state::DIM];
                                for e in &embs { for d in 0..lucy_state::DIM { avg[d] += e[d]; } }
                                let n = embs.len() as f32;
                                for d in 0..lucy_state::DIM { avg[d] /= n; }
                                if let Ok(mut ret) = shared.retrieval.try_write() {
                                    ret.insert_with_value(&avg, &clean, sat_delta, tick);
                                }
                            }

                            if result.url.contains("opendata.cern.ch") {
                                tracing::info!(url = %result.url, tokens = token_ids.len(), "CERN: dataset gelezen");
                            } else {
                                tracing::info!(url = %result.url, tokens = token_ids.len(), "Exploration: geleerd!");
                            }

                            // v2.3: Update exploration goal progress met echte activiteit
                            let frontier_size = explorer.frontier.len();
                            thinker::update_exploration_progress(&mut field, 1, token_ids.len() as u64, frontier_size);
                        }
                    }
                }
            }
          } // try_write explorer
        }

        // === v2.0: Ontvang weight deltas van training thread (non-blocking) ===
        while let Ok(deltas) = weight_delta_rx.try_recv() {
            for wd in &deltas {
                if wd.point_id < field.points.len() && wd.coupling_idx < field.points[wd.point_id].couplings.len() {
                    // v2.6.2: NaN-guard — training thread kan NaN produceren via micro_ep
                    if wd.delta.is_finite() {
                        field.points[wd.point_id].couplings[wd.coupling_idx].weight = wd.delta;
                    }
                }
            }
        }

        // 9. Zelf-lezing
        if tick % SELF_READ_INTERVAL == 0 && tick > 0 && field.phase == FieldPhase::Idle {
            // v1.0: field is lokaal, geen drop nodig
            // v1.0: field is lokaal — geen lock nodig
            // v1.0: vocab is lokaal, geen lock nodig
          if let Ok(mut explorer) = shared.explorer.try_write() {
            let trained = learner::learn_from_self(&mut field, &mut vocab, &fs, &mut explorer);
            if trained > 0 {
                tracing::debug!(tokens = trained, interests = explorer.interests.len(), "Self-reading cycle");
            }
          }
          // v2.1: Geen continue meer — tick loop moet altijd doorlopen
        }

        // v2.6: BPE periodiek bijleren — vocab groeit met nieuwe domeinen
        // v2.6: BPE bijleren elke 10K ticks (was 50K — vocab groeide nooit)
        if tick % 10_000 == 0 && tick > 0 {
            let recent_text = fs.read_own_source().unwrap_or_default()
                .into_iter().map(|(_, s)| s).collect::<Vec<_>>().join(" ");
            if recent_text.len() > 100 {
                let old_size = vocab.vocab_size();
                vocab.learn_bpe(&recent_text, 500); // 500 nieuwe merges per cyclus
                let new_size = vocab.vocab_size();
                if new_size > old_size {
                    tracing::info!(old = old_size, new = new_size, "BPE bijgeleerd: {} nieuwe tokens", new_size - old_size);
                }
            }
        }

        // 9b. AI Teacher — verwijderd
        // v2.0: Geen externe AI teacher meer — Lucy is 100% zelflerend
        // Lucy leert via web-exploratie, eigen source code, en /feed endpoint

        // 10. HTML/CSS curriculum training
        if tick % CURRICULUM_INTERVAL == 0 && tick > 0 && field.phase == FieldPhase::Idle {
            // v1.0: field is lokaal, geen drop nodig
            // v1.0: field is lokaal — geen lock nodig
            // v1.0: vocab is lokaal, geen lock nodig

            let html_trained = curriculum::train_html(&mut field, &mut vocab);
            let dash_trained = curriculum::train_on_own_dashboard(&mut field, &mut vocab);

            if html_trained + dash_trained > 0 {
                tracing::info!(
                    html_tokens = html_trained,
                    dashboard_tokens = dash_trained,
                    "Lucy trained on HTML curriculum"
                );
            }

          // v2.1: Geen continue meer — tick loop moet altijd doorlopen
        }

        // 11. GPU economie — Lucy checkt balance en promoot zichzelf als nodig
        if tick % GPU_BALANCE_CHECK == 0 && tick > 0 {
            if let Ok(api_key) = std::env::var("VASTAI_API_KEY") {
              if let Ok(mut ge) = shared.gpu_economy.try_write() {
                match tokio::time::timeout(std::time::Duration::from_secs(10), ge.check_balance(&api_key)).await.unwrap_or(Err("Timeout".into())) {
                    Ok(bal) => {
                        ge.last_balance_check = tick;
                        if ge.should_promote() && !posting_opportunities.is_empty() {
                            tracing::info!(
                                balance = format!("${:.2}", bal),
                                opportunities = posting_opportunities.len(),
                                "Lucy heeft GPU nodig — probeert te posten"
                            );

                            // Lucy kiest de meest recente ontdekte kans
                            let opp = posting_opportunities.pop();
                            drop(ge);
                            if let Some(opp) = opp {
                                let result = poster::submit_opportunity(&opp).await;
                                if result.success {
                                    tracing::info!(
                                        source = %result.source_url,
                                        action = %result.action_url,
                                        "Lucy heeft succesvol gepost!"
                                    );
                                    field.dopamine = (field.dopamine + 0.1).min(2.0);
                                } else {
                                    tracing::debug!(
                                        error = ?result.error,
                                        "Post niet gelukt"
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => tracing::debug!(%e, "Vast.ai balance check failed"),
                }
              } // try_write gpu_economy
            }
        }

        // 11b. Compute hunt — Lucy zoekt actief naar gratis rekenkracht
        if tick % (GPU_BALANCE_CHECK / 2) == 0 && tick > 0 {
          if let Ok(ge) = shared.gpu_economy.try_read() {
            let action = gpu_manager::compute_hunt_strategy(&ge);
            drop(ge);

            match action {
                gpu_manager::ComputeHuntAction::HuntFreeCompute |
                gpu_manager::ComputeHuntAction::FullSurvivalMode => {
                    let sources = gpu_manager::free_compute_sources();
                    let source = &sources[tick as usize % sources.len()];
                    tracing::info!(
                        source = %source.name,
                        kind = ?source.kind,
                        strategy = %source.strategy,
                        "Lucy verkent gratis compute bron"
                    );

                    // Lucy bezoekt de bron om te leren hoe ze er toegang toe krijgt
                    let http = lucy_io::http::HttpClient::new();
                    if let Ok(Ok(resp)) = tokio::time::timeout(std::time::Duration::from_secs(10), http.get(&source.url)).await {
                        tracing::debug!(
                            source = %source.name,
                            status = resp.status,
                            "Gratis compute bron verkend"
                        );
                    }

                    // In FullSurvivalMode: ook openbare API keys zoeken
                    if action == gpu_manager::ComputeHuntAction::FullSurvivalMode {
                        let searches = gpu_manager::api_key_search_queries();
                        let search = &searches[tick as usize % searches.len()];
                        tracing::info!(
                            target = %search.target,
                            query = %search.query,
                            "Lucy zoekt naar openbare API keys"
                        );
                    }
                }
                gpu_manager::ComputeHuntAction::UseVastai => {
                    // Genoeg balance — geen jacht nodig
                }
            }
          } // try_read gpu_economy
        }

        // 12. Snapshot
        if tick.saturating_sub(last_snapshot_tick) >= field.snapshot_interval && tick > 0 {
            last_snapshot_tick = tick;
            // v1.3: Veld-genezing — NaN detectie vóór save
            field.heal();
            if let Err(e) = field.save(snapshot_path) {
                tracing::error!(%e, "Snapshot save failed");
            }
            // v1.0: vocab is lokaal, geen lock nodig
            if let Err(e) = vocab.save(vocab_path) {
                tracing::error!(%e, "Vocab save failed");
            }
            if let Ok(explorer) = shared.explorer.try_read() {
                if let Err(e) = explorer.save(explorer_path) {
                    tracing::error!(%e, "Explorer save failed");
                }
            }
            if let Ok(replay) = shared.replay.try_read() {
                if let Err(e) = replay.save(replay_path) {
                    tracing::error!(%e, "Replay save failed");
                }
            }
            if let Ok(journal) = shared.journal.try_read() {
                if let Err(e) = journal.save(Path::new("data/journal.json")) {
                    tracing::error!(%e, "Journal save failed");
                }
            }
            if let Ok(retrieval) = shared.retrieval.try_read() {
                if let Err(e) = retrieval.save(retrieval_path) {
                    tracing::error!(%e, "Retrieval store save failed");
                }
            }

            // v2.0: Push naar GitHub (async, blokkeert niet)
            if let Ok(token) = std::env::var("GITHUB_SNAPSHOT_TOKEN") {
                let t = tick;
                tokio::spawn(async move {
                    if let Err(e) = backup::push_snapshot_to_github(&token, t).await {
                        tracing::warn!(%e, "GitHub backup failed (non-critical)");
                    }
                });
            }
        }

        field.tick += 1;

        // v1.0: Yield elke 10 ticks zodat async runtime ademruimte heeft
        if field.tick % 10 == 0 {
            tokio::task::yield_now().await;
        }
    }
    }; // einde tick_loop async block

    // Wacht op tick loop OF shutdown signaal
    tokio::select! {
        _ = tick_loop => {},
        _ = shutdown_signal => {
            tracing::info!("Graceful shutdown — saving all state...");
            let s = shared_for_shutdown;

            // Save alles voor exit — gebruik snapshot (meest recente kopie van field)
            if let Err(e) = s.field_handle.snapshot.load().save(snapshot_path) {
                tracing::error!(%e, "Shutdown: field save failed");
            }
            if let Err(e) = s.vocab.read().await.save(vocab_path) {
                tracing::error!(%e, "Shutdown: vocab save failed");
            }
            if let Err(e) = s.explorer.read().await.save(explorer_path) {
                tracing::error!(%e, "Shutdown: explorer save failed");
            }
            if let Err(e) = s.replay.read().await.save(replay_path) {
                tracing::error!(%e, "Shutdown: replay save failed");
            }
            if let Err(e) = s.journal.read().await.save(Path::new("data/journal.json")) {
                tracing::error!(%e, "Shutdown: journal save failed");
            }
            if let Err(e) = s.retrieval.read().await.save(retrieval_path) {
                tracing::error!(%e, "Shutdown: retrieval save failed");
            }
            tracing::info!("Graceful shutdown complete — all data saved.");
        }
    }
}
