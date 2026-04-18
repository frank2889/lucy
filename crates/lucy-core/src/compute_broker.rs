//! Compute Broker — Lucy bemachtigt externe compute op gratis/goedkope platforms.
//!
//! Platform-prioriteit (goedkoopste eerst):
//! 1. GitHub Actions  — gratis, 2000 min/maand
//! 2. Vast.ai         — spot GPU vanaf $0.10/uur (VASTAI_API_KEY vereist)
//! 3. RunPod.io       — spot GPU vanaf $0.20/uur (RUNPOD_API_KEY vereist)
//!
//! Lucy kiest automatisch het goedkoopste platform dat quota heeft.

use reqwest::Client;
use serde_json::json;

/// Triggert een GitHub Actions workflow voor lucy-worker training.
///
/// Deze functie dispatcht een workflow run via GitHub API.
/// De workflow kiest de juiste concept-corpus en traint N ticks, dan POST /sync.
pub async fn trigger_github_actions_worker(
    github_token: &str,
    lucy_url: &str,
    concept: &str,
    train_ticks: u64,
) -> Result<String, String> {
    if github_token.is_empty() {
        return Err("GITHUB_TOKEN not set".to_string());
    }

    // GitHub API endpoint voor workflow dispatch
    // Formaat: POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches
    let owner = "FrankWius";  // hardcoded; kan uit env
    let repo = "lucy";
    let workflow_id = "compute-worker.yml";

    let url = format!(
        "https://api.github.com/repos/{}/{}/actions/workflows/{}/dispatches",
        owner, repo, workflow_id
    );

    let client = Client::new();
    let payload = json!({
        "ref": "main",
        "inputs": {
            "lucy_url": lucy_url,
            "lucy_token": std::env::var("LUCY_API_TOKEN").unwrap_or_default(),
            "concept": concept,
            "train_ticks": train_ticks.to_string(),
        }
    });

    match client
        .post(&url)
        .header("Authorization", format!("token {}", github_token))
        .header("Accept", "application/vnd.github+json")
        .json(&payload)
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                tracing::info!(
                    concept,
                    ticks = train_ticks,
                    "GitHub Actions worker triggered"
                );
                // GitHub returns 204 No Content on success
                Ok(format!("workflow-dispatch-{}", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)))
            } else {
                let body = resp.text().await.unwrap_or_default();
                Err(format!("GitHub API error: {}", body))
            }
        }
        Err(e) => Err(format!("HTTP error: {}", e)),
    }
}

/// Zoekt het goedkoopste beschikbare Vast.ai GPU-aanbod.
pub async fn search_vastai_offer(api_key: &str) -> Result<VastOffer, String> {
    let client = Client::new();
    let url = "https://console.vast.ai/api/v0/bundles/";
    let query = json!({
        "q": {
            "gpu_ram": {"gte": 8},
            "dph_total": {"lte": 0.30},
            "disk_space": {"gte": 10},
            "rentable": true
        },
        "order": [["dph_total", "asc"]],
        "limit": 5
    });

    match client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&query)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            if let Some(offer) = data["offers"].as_array().and_then(|a| a.first()) {
                Ok(VastOffer {
                    id: offer["id"].as_u64().unwrap_or(0),
                    dph: offer["dph_total"].as_f64().unwrap_or(999.0) as f32,
                    gpu_name: offer["gpu_name"].as_str().unwrap_or("unknown").to_string(),
                })
            } else {
                Err("No offers found".to_string())
            }
        }
        Ok(resp) => Err(format!("Vast.ai API error: {}", resp.status())),
        Err(e) => Err(format!("HTTP error: {}", e)),
    }
}

#[derive(Debug)]
pub struct VastOffer {
    pub id: u64,
    pub dph: f32,
    pub gpu_name: String,
}

/// Triggert een RunPod serverless job voor lucy-worker training.
///
/// RunPod heeft gratis credits voor nieuwe accounts + spot GPU's vanaf $0.20/uur.
pub async fn trigger_runpod_worker(
    runpod_key: &str,
    lucy_url: &str,
    concept: &str,
    train_ticks: u64,
) -> Result<String, String> {
    if runpod_key.is_empty() {
        return Err("RUNPOD_API_KEY not set".to_string());
    }

    let client = Client::new();
    // RunPod Serverless API
    let url = "https://api.runpod.io/v2/lucy-worker/run";

    let payload = json!({
        "input": {
            "coordinator": lucy_url,
            "token": std::env::var("LUCY_API_TOKEN").unwrap_or_default(),
            "concept": concept,
            "ticks": train_ticks,
        }
    });

    match client
        .post(url)
        .header("Authorization", format!("Bearer {}", runpod_key))
        .json(&payload)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            let job_id = data["id"].as_str().unwrap_or("unknown").to_string();
            tracing::info!(job_id, concept, ticks = train_ticks, "RunPod worker dispatched");
            Ok(job_id)
        }
        Ok(resp) => Err(format!("RunPod API error: {}", resp.status())),
        Err(e) => Err(format!("HTTP error: {}", e)),
    }
}

/// Kiest automatisch het beste platform: GitHub Actions → RunPod → Vast.ai.
/// Returns de platform naam en run-ID.
pub async fn trigger_best_available(
    lucy_url: &str,
    concept: &str,
    train_ticks: u64,
) -> Result<(String, String), String> {
    // 1. GitHub Actions — gratis eerste keuze
    let github_token = std::env::var("GITHUB_TOKEN").unwrap_or_default();
    if !github_token.is_empty() {
        if let Ok(run_id) = trigger_github_actions_worker(&github_token, lucy_url, concept, train_ticks).await {
            return Ok(("github_actions".to_string(), run_id));
        }
    }

    // 2. RunPod — betaalbaar alternatief
    let runpod_key = std::env::var("RUNPOD_API_KEY").unwrap_or_default();
    if !runpod_key.is_empty() {
        if let Ok(job_id) = trigger_runpod_worker(&runpod_key, lucy_url, concept, train_ticks).await {
            return Ok(("runpod".to_string(), job_id));
        }
    }

    // 3. Vast.ai — GPU spot market
    let vastai_key = std::env::var("VASTAI_API_KEY").unwrap_or_default();
    if !vastai_key.is_empty() {
        if let Ok(offer) = search_vastai_offer(&vastai_key).await {
            tracing::info!(
                offer_id = offer.id,
                gpu = offer.gpu_name,
                dph = offer.dph,
                "Vast.ai offer found — manual rental needed (SSH automation in v2.6.18)"
            );
            return Ok(("vastai".to_string(), offer.id.to_string()));
        }
    }

    Err("No compute platform available — set GITHUB_TOKEN, RUNPOD_API_KEY, or VASTAI_API_KEY".to_string())
}

#[cfg(test)]
mod tests {
    // TODO: mock HTTP tests
}
