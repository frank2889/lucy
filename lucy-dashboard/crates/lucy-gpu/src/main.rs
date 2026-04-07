//! # lucy-gpu — Remote GPU Compute Service
//!
//! Standalone server die op een Vast.ai GPU-machine draait.
//! Lucy op DigitalOcean stuurt velddata → GPU rekent → stuurt resultaat terug.
//!
//! Endpoints:
//!   POST /upload    — Upload volledige veldstaat naar GPU
//!   POST /tick      — Run N GPU ticks (phi + memory + tension)
//!   POST /weights   — Run weight update op GPU
//!   POST /clamps    — Upload clamp data
//!   POST /params    — Update hyperparameters
//!   GET  /download  — Download veldstaat van GPU
//!   GET  /status    — GPU health check

mod gpu;

use axum::{Router, Json, extract::State};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::Deserialize;

use gpu::FieldSnapshot;

type SharedGpu = Arc<Mutex<Option<gpu::GpuCompute>>>;

#[derive(Deserialize)]
struct TickRequest {
    n_ticks: u32,
}

#[derive(Deserialize)]
struct ParamsRequest {
    lambda: f32,
    beta: f32,
    alpha: f32,
    rho: f32,
    mu: f32,
    kappa: f32,
    omega: f32,
    ema_decay: f32,
    growth_threshold: f32,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    tracing::info!("lucy-gpu — Remote GPU Compute Service starting...");

    let gpu_state: SharedGpu = Arc::new(Mutex::new(None));

    let app = Router::new()
        .route("/status", axum::routing::get(status))
        .route("/upload", axum::routing::post(upload))
        .route("/tick", axum::routing::post(tick))
        .route("/weights", axum::routing::post(weights))
        .route("/clamps", axum::routing::post(clamps))
        .route("/params", axum::routing::post(params))
        .route("/download", axum::routing::get(download))
        .with_state(gpu_state);

    let addr = "0.0.0.0:3001";
    tracing::info!(addr, "GPU service listening");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn status(State(gpu): State<SharedGpu>) -> Json<serde_json::Value> {
    let lock = gpu.lock().await;
    match lock.as_ref() {
        Some(g) => Json(serde_json::json!({
            "ready": true,
            "gpu_name": g.gpu_name(),
            "n_points": g.n_points(),
        })),
        None => Json(serde_json::json!({
            "ready": false,
            "gpu_name": "not initialized",
            "n_points": 0,
        })),
    }
}

async fn upload(
    State(gpu_state): State<SharedGpu>,
    body: axum::body::Bytes,
) -> Json<serde_json::Value> {
    let snapshot: FieldSnapshot = match bincode::deserialize(&body) {
        Ok(s) => s,
        Err(e) => return Json(serde_json::json!({"error": format!("deserialize: {e}")})),
    };

    let g = gpu::GpuCompute::from_snapshot(&snapshot);
    let n = g.n_points();
    let name = g.gpu_name();

    *gpu_state.lock().await = Some(g);

    tracing::info!(n_points = n, gpu = %name, "Field uploaded to GPU");
    Json(serde_json::json!({"ok": true, "n_points": n, "gpu": name}))
}

async fn tick(
    State(gpu_state): State<SharedGpu>,
    Json(req): Json<TickRequest>,
) -> Json<serde_json::Value> {
    let lock = gpu_state.lock().await;
    match lock.as_ref() {
        Some(g) => {
            let n = req.n_ticks.min(1000); // max 1000 ticks per call
            for _ in 0..n {
                g.tick();
            }
            Json(serde_json::json!({"ok": true, "ticks": n}))
        }
        None => Json(serde_json::json!({"error": "GPU not initialized — upload first"})),
    }
}

async fn weights(State(gpu_state): State<SharedGpu>) -> Json<serde_json::Value> {
    let lock = gpu_state.lock().await;
    match lock.as_ref() {
        Some(g) => {
            g.update_weights();
            Json(serde_json::json!({"ok": true}))
        }
        None => Json(serde_json::json!({"error": "GPU not initialized"})),
    }
}

async fn clamps(
    State(gpu_state): State<SharedGpu>,
    body: axum::body::Bytes,
) -> Json<serde_json::Value> {
    #[derive(Deserialize)]
    struct ClampData {
        mask: Vec<u32>,
        values: Vec<f32>,
    }

    let data: ClampData = match bincode::deserialize(&body) {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"error": format!("deserialize: {e}")})),
    };

    let lock = gpu_state.lock().await;
    match lock.as_ref() {
        Some(g) => {
            g.set_clamps_raw(&data.mask, &data.values);
            Json(serde_json::json!({"ok": true}))
        }
        None => Json(serde_json::json!({"error": "GPU not initialized"})),
    }
}

async fn params(
    State(gpu_state): State<SharedGpu>,
    Json(req): Json<ParamsRequest>,
) -> Json<serde_json::Value> {
    let lock = gpu_state.lock().await;
    match lock.as_ref() {
        Some(g) => {
            g.sync_params_raw(
                req.lambda, req.beta, req.alpha, req.rho, req.mu,
                req.kappa, req.omega, req.ema_decay, req.growth_threshold,
            );
            Json(serde_json::json!({"ok": true}))
        }
        None => Json(serde_json::json!({"error": "GPU not initialized"})),
    }
}

async fn download(State(gpu_state): State<SharedGpu>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let lock = gpu_state.lock().await;
    match lock.as_ref() {
        Some(g) => {
            let snapshot = g.download_snapshot();
            let bytes = bincode::serialize(&snapshot).unwrap();
            (
                [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
                bytes,
            ).into_response()
        }
        None => Json(serde_json::json!({"error": "GPU not initialized"})).into_response(),
    }
}
