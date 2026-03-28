//! # Lucy v0.5 — Field Intelligence
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
//! │     └── server.rs ── HTTP API (axum, port 3000)             │
//! │                                                             │
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
//! | 8. Exploratie | Elke 20.000 ticks | Bezoek een URL, leer van de content |
//! | 9. Zelf-lezing | Elke 5.000 ticks | Herlees eigen source code |
//! | 10. Snapshot | Elke 10.000 ticks | Sla alles op naar disk |
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

// === Tick intervals ===
// Elke constante bepaalt hoe vaak een bepaald proces draait.
// Lagere waarde = vaker = meer CPU maar snellere adaptatie.
const LOG_INTERVAL: u64 = 1_000;          // Heartbeat logging
const LEARN_INTERVAL: u64 = 100;          // Gewicht updates (Hebbian)
const GROWTH_INTERVAL: u64 = 10_000;      // Punt-splitsing bij vastloop
const SELF_READ_INTERVAL: u64 = 5_000;    // Herlees eigen source code
const SNAPSHOT_INTERVAL: u64 = 10_000;    // Sla alles op naar disk
const EXPLORE_INTERVAL: u64 = 20_000;     // Bezoek een URL
const CONSOLIDATION_INTERVAL: u64 = 500;  // Replay een oud voorbeeld

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    tracing::info!("Lucy v0.5 — Field Intelligence — waking up...");

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
            consolidation::ReplayBuffer::new()
        })
    } else {
        consolidation::ReplayBuffer::new()
    };

    // === Bootstrap: Lucy leert eerst van zichzelf ===
    if field.tick == 0 {
        tracing::info!("First boot — Lucy learns from her own source code");
        let trained = learner::learn_from_self(&mut field, &mut vocab, &fs, &mut explorer);
        tracing::info!(tokens = trained, vocab = vocab.vocab_size(), interests = explorer.interests.len(), "Self-learning complete");
    }

    // === Gedeelde state ===
    let shared = Arc::new(server::AppState {
        field: Arc::new(RwLock::new(field)),
        vocab: Arc::new(RwLock::new(vocab)),
        explorer: Arc::new(RwLock::new(explorer)),
        replay: Arc::new(RwLock::new(replay)),
    });

    // Start API server
    server::spawn_server(shared.clone());

    tracing::info!("Lucy is alive. Field dynamics + API server running.");

    // === TICK LOOP ===
    let mut tick_timer = Instant::now();

    loop {
        let mut field = shared.field.write().await;
        let tick = field.tick;

        // Alleen achtergrond dynamica als we niet aan het processen zijn
        if field.phase == FieldPhase::Idle {
            // 1. Veldberekeningen
            let self_input = compute::self_state_input(&field);
            let active: Vec<u32> = field.active_set.clone();

            for &id in &active {
                let input = if field.points[id as usize].is_sensory() {
                    Some(&self_input)
                } else {
                    None
                };
                compute::update_phi(&mut field, id, input);
                compute::update_memory(&mut field, id);
                compute::compute_tension(&mut field, id);
            }

            // 2. Exploratie-ruis
            if tick % 10 == 0 {
                compute::maybe_perturb(&mut field);
            }

            // 3. Leren (gewicht updates)
            if tick % LEARN_INTERVAL == 0 {
                let active: Vec<u32> = field.active_set.clone();
                for &id in &active {
                    compute::update_weights(&mut field, id);
                }
                growth::maybe_meta_learn(&mut field);
            }

            // 4. Groei
            if tick % GROWTH_INTERVAL == 0 {
                let active: Vec<u32> = field.active_set.clone();
                for &id in &active {
                    match meta::check_escalation(&field, id) {
                        meta::EscalationLevel::Grow => {
                            growth::maybe_grow(&mut field);
                            field.cache_io_ids(); // Update na groei
                            break;
                        }
                        meta::EscalationLevel::MetaLearn => {
                            growth::maybe_meta_learn(&mut field);
                        }
                        meta::EscalationLevel::None => {}
                    }
                }
            }

            // 5. Energie management
            growth::manage_energy(&mut field);
        }

        // 6. Consolidation (replay)
        if tick % CONSOLIDATION_INTERVAL == 0 && tick > 0 && field.phase == FieldPhase::Idle {
            let replay = shared.replay.read().await;
            if !replay.examples.is_empty() {
                let _delta = replay.replay_one(&mut field);
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

            let vocab = shared.vocab.read().await;
            let explorer = shared.explorer.read().await;

            tracing::info!(
                tick,
                active = field.active_set.len(),
                tension = format!("{:.4}", avg_tension),
                tps = format!("{:.0}", tps),
                vocab = vocab.vocab_size(),
                pages = explorer.pages_read,
                tokens = explorer.tokens_processed,
                "heartbeat"
            );
        }

        // 8. Autonome exploratie
        if tick % EXPLORE_INTERVAL == 0 && tick > 0 && field.phase == FieldPhase::Idle {
            // Exploratie gebeurt buiten de lock
            drop(field);

            let http = lucy_io::http::HttpClient::new();
            let mut field = shared.field.write().await;
            let mut vocab = shared.vocab.write().await;
            let mut explorer = shared.explorer.write().await;

            match learner::learn_from_exploration(&http, &mut field, &mut vocab, &mut explorer).await {
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!(%e, "Exploration failed (will retry)");
                }
            }

            field.tick += 1;
            drop(field);
            drop(vocab);
            drop(explorer);

            if tick % 10 == 0 {
                tokio::task::yield_now().await;
            }
            continue; // Skip de normale tick increment
        }

        // 9. Zelf-lezing
        if tick % SELF_READ_INTERVAL == 0 && tick > 0 && field.phase == FieldPhase::Idle {
            drop(field);
            let mut field = shared.field.write().await;
            let mut vocab = shared.vocab.write().await;
            let mut explorer = shared.explorer.write().await;
            let trained = learner::learn_from_self(&mut field, &mut vocab, &fs, &mut explorer);
            if trained > 0 {
                tracing::debug!(tokens = trained, interests = explorer.interests.len(), "Self-reading cycle");
            }
            field.tick += 1;
            drop(field);
            drop(vocab);
            if tick % 10 == 0 {
                tokio::task::yield_now().await;
            }
            continue;
        }

        // 10. Snapshot
        if tick % SNAPSHOT_INTERVAL == 0 && tick > 0 {
            if let Err(e) = field.save(snapshot_path) {
                tracing::error!(%e, "Snapshot save failed");
            }
            let vocab = shared.vocab.read().await;
            if let Err(e) = vocab.save(vocab_path) {
                tracing::error!(%e, "Vocab save failed");
            }
            let explorer = shared.explorer.read().await;
            if let Err(e) = explorer.save(explorer_path) {
                tracing::error!(%e, "Explorer save failed");
            }
            let replay = shared.replay.read().await;
            if let Err(e) = replay.save(replay_path) {
                tracing::error!(%e, "Replay save failed");
            }
        }

        field.tick += 1;
        drop(field);

        if tick % 10 == 0 {
            tokio::task::yield_now().await;
        }
    }
}
