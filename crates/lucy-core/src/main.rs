mod compute;
mod field;
mod growth;
mod io_bridge;
mod meta;
mod server;

use lucy_io::config::Config;
use lucy_io::fs::FileSystem;
use lucy_state::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

const SNAPSHOT_PATH: &str = "data/lucy.snapshot";
const SNAPSHOT_INTERVAL: u64 = 10_000;
const LEARN_INTERVAL: u64 = 100;
const GROWTH_INTERVAL: u64 = 10_000;
const LOG_INTERVAL: u64 = 1_000;
const IO_INTERVAL: u64 = 500;
const SELF_READ_INTERVAL: u64 = 5_000;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    tracing::info!("Lucy is waking up...");

    let root = PathBuf::from(".");
    let fs = FileSystem::new(root.clone());
    let snapshot_path = Path::new(SNAPSHOT_PATH);

    // Boot: laad snapshot of bootstrap vanuit source code
    let mut field = if snapshot_path.exists() {
        match Field::load(snapshot_path) {
            Ok(f) => {
                tracing::info!(tick = f.tick, points = f.points.len(), "Resumed from snapshot");
                f
            }
            Err(e) => {
                tracing::warn!(%e, "Failed to load snapshot, bootstrapping fresh");
                bootstrap_fresh(&fs)
            }
        }
    } else {
        tracing::info!("No snapshot found, bootstrapping from source code");
        bootstrap_fresh(&fs)
    };

    // Configuratie
    let config = Config::from_env();

    // I/O Bridge
    let mut io = io_bridge::IoBridge::new(FileSystem::new(root.clone()), config);

    // Eerste zelf-lezing
    let self_data = io.read_self();
    let n_self = self_data.len().min(100);
    io.feed_sensory_input(&mut field, &self_data[..n_self]);
    tracing::info!(chunks = n_self, "Initial self-read complete");

    // Gedeelde state voor API server
    let shared_field: server::SharedField = Arc::new(RwLock::new(field));

    // Start API server op achtergrond
    server::spawn_server(shared_field.clone());

    tracing::info!("Lucy is alive. Tick loop + API server starting.");

    // === DE TICK LOOP — Lucy's hartslag ===
    let mut tick_timer = Instant::now();

    loop {
        // Lock het veld voor deze tick
        let mut field = shared_field.write().await;
        let tick = field.tick;

        // 1. Actieve punten: bereken ε, update Φ, update M
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

        // 2. Exploratie
        if tick % 10 == 0 {
            compute::maybe_perturb(&mut field);
        }

        // 3. Interne simulatie
        if tick % 50 == 0 {
            let active: Vec<u32> = field.active_set.clone();
            for &id in &active {
                let p = &field.points[id as usize];
                if p.tension > field.growth_threshold * 0.5 && p.energy > 0.3 {
                    let _reduction = meta::simulate(&field, id, 5);
                }
            }
        }

        // 4. Energie management
        growth::manage_energy(&mut field);

        // 5. Leren
        if tick % LEARN_INTERVAL == 0 {
            let active: Vec<u32> = field.active_set.clone();
            for &id in &active {
                compute::update_weights(&mut field, id);
            }
            growth::maybe_meta_learn(&mut field);
        }

        // 6. Groei + escalatie
        if tick % GROWTH_INTERVAL == 0 {
            let active: Vec<u32> = field.active_set.clone();
            for &id in &active {
                match meta::check_escalation(&field, id) {
                    meta::EscalationLevel::Grow => {
                        growth::maybe_grow(&mut field);
                        break;
                    }
                    meta::EscalationLevel::MetaLearn => {
                        growth::maybe_meta_learn(&mut field);
                    }
                    meta::EscalationLevel::None => {}
                }
            }
        }

        // 7. Logging
        if tick % LOG_INTERVAL == 0 {
            let elapsed = tick_timer.elapsed();
            let ticks_per_sec = LOG_INTERVAL as f64 / elapsed.as_secs_f64();
            tick_timer = Instant::now();

            let avg_tension: f32 = if !field.active_set.is_empty() {
                field.active_set.iter().map(|&id| field.points[id as usize].tension).sum::<f32>()
                    / field.active_set.len() as f32
            } else { 0.0 };

            let avg_energy: f32 = if !field.active_set.is_empty() {
                field.active_set.iter().map(|&id| field.points[id as usize].energy).sum::<f32>()
                    / field.active_set.len() as f32
            } else { 0.0 };

            tracing::info!(
                tick,
                points = field.points.len(),
                active = field.active_set.len(),
                avg_tension = format!("{:.4}", avg_tension),
                avg_energy = format!("{:.4}", avg_energy),
                ticks_per_sec = format!("{:.0}", ticks_per_sec),
                balance_cents = field.economy.balance_cents,
                profitable = field.economy.is_profitable(),
                "heartbeat"
            );
        }

        // 8. I/O: motor output
        if tick % IO_INTERVAL == 0 {
            let actions = io.check_motor_output(&field);
            if !actions.is_empty() {
                io.execute_actions(&mut field, actions).await;
            }
        }

        // 9. Zelf-lezing
        if tick % SELF_READ_INTERVAL == 0 && tick > 0 {
            let self_data = io.read_self();
            let n = self_data.len().min(50);
            io.feed_sensory_input(&mut field, &self_data[..n]);
        }

        // 10. Snapshot
        if tick % SNAPSHOT_INTERVAL == 0 && tick > 0 {
            if let Err(e) = field.save(snapshot_path) {
                tracing::error!(%e, "Snapshot save failed");
            }
        }

        field.tick += 1;

        // Drop de lock zodat API server ertussen kan
        drop(field);

        // Yield naar tokio (API requests kunnen nu verwerkt worden)
        if tick % 10 == 0 {
            tokio::task::yield_now().await;
        }
    }
}

fn bootstrap_fresh(fs: &FileSystem) -> Field {
    let sources = fs.read_own_source().unwrap_or_default();

    if sources.is_empty() {
        tracing::warn!("No source files found, creating minimal field");
        let mut field = Field::new();
        let mut rng = rand::thread_rng();
        use rand::Rng;

        for _ in 0..1000 {
            let mut p = FieldPoint::new();
            for d in 0..DIM {
                p.phi[d] = rng.gen_range(-1.0..1.0);
            }
            p.energy = 1.0;
            field.points.push(p);
        }

        let n = field.points.len();
        for i in 0..n {
            let n_connections = rng.gen_range(2..8);
            for _ in 0..n_connections {
                let target = rng.gen_range(0..n);
                if target != i {
                    field.points[i].couplings.push(Coupling {
                        target: target as u32,
                        weight: rng.gen_range(-0.5..0.5),
                        momentum: 0.0,
                    });
                }
            }
        }

        field.total_energy = n as f32;
        field.rebuild_active_set();
        field
    } else {
        tracing::info!(files = sources.len(), "Bootstrapping from source code");
        field::bootstrap_from_source(&sources)
    }
}
