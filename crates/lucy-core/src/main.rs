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
mod replica_manager;
mod federation;
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
const GPU_BALANCE_CHECK: u64 = 50_000;   // Check Vast.ai balance (infra)
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

    tracing::info!("Lucy v0.9 — Field Intelligence + RAG + Attention + Working Memory — waking up...");

    let root = PathBuf::from(".");
    let fs = FileSystem::new(root.clone());

    // === Laad of bootstrap veld ===
    let snapshot_path = Path::new(SNAPSHOT_PATH);
    let mut field = if snapshot_path.exists() {
        match Field::load(snapshot_path) {
            Ok(mut f) => {
                f.cache_io_ids();
                tracing::info!(tick = f.tick, points = f.points.len(), "Resumed from snapshot");
                f
            }
            Err(e) => {
                tracing::warn!(%e, "Failed to load snapshot, bootstrapping fresh");
                field::bootstrap_field()
            }
        }
    } else {
        tracing::info!("No snapshot found, bootstrapping field");
        field::bootstrap_field()
    };

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

    // Seed API discovery — Lucy vindt zelf welke APIs en data er bestaan
    explorer.seed_api_discovery();
    // Laad APIs die Lucy zelf eerder heeft ontdekt en opgeslagen
    explorer.load_self_managed_apis();
    tracing::info!(frontier = explorer.frontier.len(), "API discovery sources seeded");

    // === v0.7: Initialiseer bewustzijns-systemen ===
    compute::init_rnd(&mut field);
    compute::assign_depths(&mut field);
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
            }
        }

        let tick = field.tick;

        // v2.1: Debug — log elke 100 ticks
        if tick % 100 == 0 {
            tracing::info!(tick, phase = ?field.phase, "tick_alive");
        }

        // v2.3: Reanimatie — detecteer dood veld en schud het wakker
        // Checkt elke 50 ticks of het veld dood is (tension ≈ 0) en injecteert sterke ruis
        if tick % 50 == 0 {
            let avg_t: f32 = field.active_set.iter()
                .map(|&id| field.points[id as usize].tension).sum::<f32>()
                / field.active_set.len().max(1) as f32;
            // Drempel: tension onder settle_epsilon = effectief dood
            if avg_t < field.settle_epsilon {
                use rand::Rng;
                let mut rng = rand::rng();
                // Re-initialiseer phi naar random [-0.1, 0.1] (zelfde als bootstrap_field)
                // maar LAAT prediction op 0 → maximale prediction error
                for &id in &field.active_set {
                    let p = &mut field.points[id as usize];
                    for d in 0..lucy_state::DIM {
                        p.phi[d] = rng.random_range(-0.1f32..0.1f32);
                        p.prediction[d] = 0.0; // prediction = 0 → error = phi
                    }
                    // Reset ook de tension tracking zodat het niet gedempd wordt
                    p.tension_ema = 0.0;
                    p.error_var = 0.0;
                }
                // Forceer lambda en alpha zodat het veld niet meteen teruggetrokken wordt
                field.alpha = field.alpha.min(0.0001); // minimale sparse penalty tijdens reanimatie
                tracing::warn!(tick, avg_tension = %format!("{:.6}", avg_t),
                    "REANIMATIE: dood veld — phi gereset, prediction op nul, sparse penalty verlaagd");
            }
        }

        // v2.3: adapt_dynamics — update ALLE dynamische parameters op basis van veldstaat
        if tick == 0 || tick % 50 == 0 {
            compute::adapt_dynamics(&mut field);
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

                let self_input = compute::self_state_input(&field);

                // v1.2: Skip punten met near-zero tension (efficiëntie)
                // Lucy wordt automatisch sneller naarmate haar veld stabiliseert
                let tension_floor = 0.0001; // v2.1: Lagere drempel — meer punten blijven actief
                let mut updated = 0u32;
                for i in 0..field.active_set.len() {
                    let id = field.active_set[i];
                    // Altijd update sensory + output punten, skip rustige hidden punten
                    let dominated = field.points[id as usize].tension < tension_floor
                        && !field.points[id as usize].is_sensory()
                        && !field.points[id as usize].is_output()
                        && tick % 10 != 0; // elke 10e tick toch updaten (voorkomt permanent slapen)

                    if dominated { continue; }

                    let input = if field.points[id as usize].is_sensory() {
                        Some(&self_input)
                    } else {
                        None
                    };
                    compute::update_phi(&mut field, id, input);
                    compute::update_memory(&mut field, id);
                    compute::compute_tension(&mut field, id);
                    updated += 1;
                }

                if tick % 100 == 0 { tracing::info!(updated, "step:field_updates_done"); }
                // v0.7: GWT broadcast + intentie (neuromodulatie verplaatst naar vóór field updates)
                compute::global_workspace_broadcast(&mut field);
                if tick % 100 == 0 { tracing::info!("step:gwt_done"); }

                // === v2.0: Bewustzijns-fysica — elke tick ===
                // Oscillaties: gamma-ritme voor temporele binding (Kuramoto model)
                compute::update_oscillations(&mut field);

                // v2.0: Micro-EP training — 1 chunk per tick uit de queue
                if let Some(chunk) = training_queue.pop() {
                    let delta = compute::micro_ep(&mut field, &chunk.input_embs, &chunk.target_embs);
                    if delta > 0.001 && tick % 1000 == 0 {
                        tracing::info!(delta = format!("{:.4}", delta), queue = training_queue.len(), "Micro-EP training");
                    }
                }

                // Elke 500 ticks: zwaardere bewustzijns-metrieken (was 50 — te frequent)
                if tick % 500 == 0 {
                    compute::compute_integrated_information(&mut field);
                    compute::compute_criticality(&mut field);
                    compute::anti_hebbian_decorrelation(&mut field);
                }

                if tick % 500 == 0 {
                    compute::record_temporal_snapshot(&mut field);
                }
                if tick % 500 == 0 {
                    compute::update_causal_events(&mut field);
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

                // Leren (gewicht updates) — CPU path
                if tick.saturating_sub(last_learn_tick) >= field.learn_interval {
                    last_learn_tick = tick;
                    let active: Vec<u32> = field.active_set.clone();
                    for &id in &active {
                        compute::update_weights(&mut field, id);
                    }
                    growth::maybe_meta_learn(&mut field);

                    // v0.9: Update attention weights
                    compute::update_attention_weights(&mut field);

                    // v0.7: Consolideer weights na leren (EWC anti-vergeten)
                    compute::consolidate_weights(&mut field);
                }
            }

            // 2. Reflectie: Lucy herkauwt haar laatste gedachte (continuous thought)
            if tick % 100 == 0 && field.last_thought.is_some() {
                compute::reflect_tick(&mut field);
            }

            // 2b. Exploratie-ruis
            if tick % 10 == 0 {
                compute::maybe_perturb(&mut field);
            }

            // v0.7: Edge of chaos monitoring (elke 1000 ticks)
            if tick % 1000 == 0 {
                compute::measure_spectral_radius(&mut field);
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

            if result.status == 200 {
                if let Ok(body_bytes) = result.body {
                    explorer.pages_read += 1;
                    let body = String::from_utf8_lossy(&body_bytes);
                    explorer.discover_urls(&body);
                    explorer.discover_apis(&body);

                    let text = learner::extract_readable_text(&body);
                    if text.len() >= 20 {
                        let capped: String = text.chars().take(2000).collect();
                        if let Some(clean) = learner::filter_content(&capped) {
                            explorer.absorb_interests(&clean);
                            let token_ids = vocab.tokenize(&clean);
                            explorer.tokens_processed += token_ids.len() as u64;

                            // Stuur chunks naar training thread (non-blocking)
                            let chunk_size = 16;
                            for chunk in token_ids.windows(chunk_size).step_by(chunk_size) {
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

                            // Retrieval store
                            let embs = crate::encoding::encode_text_readonly(&vocab, &clean);
                            if !embs.is_empty() {
                                let mut avg = [0.0f32; lucy_state::DIM];
                                for e in &embs { for d in 0..lucy_state::DIM { avg[d] += e[d]; } }
                                let n = embs.len() as f32;
                                for d in 0..lucy_state::DIM { avg[d] /= n; }
                                if let Ok(mut ret) = shared.retrieval.try_write() {
                                    ret.insert(&avg, &clean);
                                }
                            }

                            tracing::info!(url = %result.url, tokens = token_ids.len(), "Exploration: geleerd!");

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
                    field.points[wd.point_id].couplings[wd.coupling_idx].weight = wd.delta;
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
                        if ge.should_promote() {
                            tracing::info!(
                                balance = format!("${:.2}", bal),
                                "Lucy heeft GPU nodig — start autonome promotie"
                            );

                            // Lucy probeert haar referral link te verspreiden
                            let http = lucy_io::http::HttpClient::new();
                            let _promo = ge.generate_promo_message();
                            let targets = ge.promotion_targets();

                            // Kies een willekeurig platform om te proberen
                            let target = &targets[tick as usize % targets.len()];
                            tracing::info!(
                                platform = %target.platform,
                                strategy = %target.strategy,
                                "Lucy overweegt promotie"
                            );

                            // Lucy leert over het platform door het te bezoeken
                            // Ze moet zelf uitvogelen HOE ze er kan posten
                            match tokio::time::timeout(std::time::Duration::from_secs(10), http.get(&target.url)).await.unwrap_or(Err("Timeout".to_string())) {
                                Ok(resp) => {
                                    tracing::debug!(
                                        platform = %target.platform,
                                        status = resp.status,
                                        "Lucy verkent promotie-platform"
                                    );
                                }
                                Err(e) => {
                                    tracing::debug!(%e, "Platform niet bereikbaar");
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
