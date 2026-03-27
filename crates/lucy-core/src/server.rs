use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use lucy_state::*;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Gedeelde state tussen tick loop en API server
pub type SharedField = Arc<RwLock<Field>>;

/// Start de API server op een aparte tokio task
pub fn spawn_server(field: SharedField) {
    tokio::spawn(async move {
        let app = Router::new()
            .route("/", get(root))
            .route("/status", get(status))
            .route("/field", get(field_info))
            .route("/economy", get(economy_info))
            .route("/query", post(query))
            .route("/feed", post(feed_input))
            .with_state(field);

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
            .await
            .expect("Failed to bind port 3000");

        tracing::info!("API server listening on port 3000");

        axum::serve(listener, app).await.ok();
    });
}

/// Root: wie is Lucy?
async fn root() -> Json<Value> {
    Json(json!({
        "name": "Lucy",
        "version": "0.1.0",
        "description": "Zelfontwikkelend samenveld — prediction error minimalisatie als intelligentie",
        "endpoints": ["/status", "/field", "/economy", "/query", "/feed"]
    }))
}

/// Status: huidige veldstaat (gratis endpoint)
async fn status(State(field): State<SharedField>) -> Json<Value> {
    let f = field.read().await;
    let n = f.active_set.len() as f32;

    let avg_tension = if n > 0.0 {
        f.active_set.iter().map(|&id| f.points[id as usize].tension).sum::<f32>() / n
    } else {
        0.0
    };

    let avg_energy = if n > 0.0 {
        f.active_set.iter().map(|&id| f.points[id as usize].energy).sum::<f32>() / n
    } else {
        0.0
    };

    Json(json!({
        "alive": true,
        "tick": f.tick,
        "points": f.points.len(),
        "active": f.active_set.len(),
        "avg_tension": format!("{:.6}", avg_tension),
        "avg_energy": format!("{:.4}", avg_energy),
        "total_energy": format!("{:.2}", f.total_energy),
    }))
}

/// Veld info: gedetailleerde veldstaat
async fn field_info(State(field): State<SharedField>) -> Json<Value> {
    let f = field.read().await;

    // Top-10 punten met hoogste spanning
    let mut high_tension: Vec<(u32, f32, f32)> = f.active_set.iter()
        .map(|&id| {
            let p = &f.points[id as usize];
            (id, p.tension, p.energy)
        })
        .collect();
    high_tension.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    high_tension.truncate(10);

    Json(json!({
        "tick": f.tick,
        "total_points": f.points.len(),
        "active_points": f.active_set.len(),
        "hyperparameters": {
            "kappa": f.kappa,
            "omega": f.omega,
            "lambda": f.lambda,
            "alpha": f.alpha,
            "growth_threshold": f.growth_threshold,
        },
        "top_tension": high_tension.iter().map(|(id, t, e)| {
            json!({"id": id, "tension": format!("{:.6}", t), "energy": format!("{:.4}", e)})
        }).collect::<Vec<_>>(),
    }))
}

/// Economie: financiële staat
async fn economy_info(State(field): State<SharedField>) -> Json<Value> {
    let f = field.read().await;
    let e = &f.economy;

    Json(json!({
        "balance_cents": e.balance_cents,
        "balance_display": format!("${:.2}", e.balance_cents as f64 / 100.0),
        "total_earned": e.total_earned,
        "total_spent": e.total_spent,
        "monthly_server_cost": e.monthly_server_cost,
        "profitable": e.is_profitable(),
        "hourly_api_budget": e.hourly_api_budget,
    }))
}

/// Query: stuur data naar Lucy en krijg haar veldrespons terug
/// Dit is het betaalde endpoint — hier verdient Lucy geld
async fn query(
    State(field): State<SharedField>,
    body: String,
) -> Result<Json<Value>, StatusCode> {
    if body.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut f = field.write().await;

    // Converteer input naar sensorische vectoren
    let bytes = body.as_bytes();
    let mut inputs: Vec<[f32; DIM]> = Vec::new();
    for chunk in bytes.chunks(DIM) {
        let mut input = [0.0f32; DIM];
        for (d, &b) in chunk.iter().enumerate() {
            input[d] = (b as f32 / 127.5) - 1.0;
        }
        inputs.push(input);
    }

    // Voed input aan sensorische punten
    let sensory: Vec<u32> = f.active_set.iter()
        .copied()
        .filter(|&id| f.points[id as usize].is_sensory())
        .collect();

    if sensory.is_empty() {
        return Ok(Json(json!({"error": "No sensory points active"})));
    }

    let n_input = inputs.len().min(50);
    for (i, input) in inputs[..n_input].iter().enumerate() {
        let target_id = sensory[i % sensory.len()] as usize;
        let p = &mut f.points[target_id];
        for d in 0..DIM {
            p.phi[d] = 0.5 * p.phi[d] + 0.5 * input[d];
        }
    }

    // Lees de motor-punten als response (Lucy's "antwoord")
    let motor_output: Vec<Vec<f32>> = f.active_set.iter()
        .filter(|&&id| f.points[id as usize].is_motor())
        .take(10)
        .map(|&id| f.points[id as usize].phi.to_vec())
        .collect();

    // Registreer inkomsten: elke query levert 1 cent op
    f.economy.earn(1);

    Ok(Json(json!({
        "tick": f.tick,
        "input_chunks": n_input,
        "motor_response": motor_output,
        "sensory_points_fed": sensory.len().min(n_input),
        "balance_after": f.economy.balance_cents,
    })))
}

/// Feed: geef Lucy gratis data (training/donatie)
async fn feed_input(
    State(field): State<SharedField>,
    body: String,
) -> Json<Value> {
    let mut f = field.write().await;

    let bytes = body.as_bytes();
    let mut inputs: Vec<[f32; DIM]> = Vec::new();
    for chunk in bytes.chunks(DIM) {
        let mut input = [0.0f32; DIM];
        for (d, &b) in chunk.iter().enumerate() {
            input[d] = (b as f32 / 127.5) - 1.0;
        }
        inputs.push(input);
    }

    let sensory: Vec<u32> = f.active_set.iter()
        .copied()
        .filter(|&id| f.points[id as usize].is_sensory())
        .collect();

    let n = inputs.len().min(100);
    for (i, input) in inputs[..n].iter().enumerate() {
        if !sensory.is_empty() {
            let target_id = sensory[i % sensory.len()] as usize;
            let p = &mut f.points[target_id];
            for d in 0..DIM {
                p.phi[d] = 0.7 * p.phi[d] + 0.3 * input[d];
            }
            p.energy = (p.energy + 0.01).min(5.0);
        }
    }

    Json(json!({
        "accepted": true,
        "chunks_fed": n,
        "tick": f.tick,
    }))
}
