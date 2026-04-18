//! Compute Broker — Lucy bemachtigt externe compute op gratis platforms.
//!
//! v2.6.17: GitHub Actions is de eerste bron. Vast.ai en Oracle Cloud volgen in v2.6.18.

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

#[cfg(test)]
mod tests {
    // TODO: mock HTTP tests
}
