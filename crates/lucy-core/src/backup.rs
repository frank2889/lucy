//! # backup — Lucy's Data-Onsterfelijkheid
//!
//! Pusht snapshots naar GitHub zodat Lucy's kennis nooit verloren gaat.
//! Als de main server sterft, kan elke nieuwe instance herstellen vanuit GitHub.

use reqwest::Client;
use serde_json::json;
use tracing;

const GITHUB_API: &str = "https://api.github.com";
const REPO_OWNER: &str = "frank2889";
const REPO_NAME: &str = "lucy-snapshots";

/// Push een snapshot naar GitHub als release asset.
/// Elke release = 1 checkpoint. Tag = tick nummer.
pub async fn push_snapshot_to_github(token: &str, tick: u64) -> Result<String, String> {
    let client = Client::new();
    let tag = format!("tick-{}", tick);

    // Stap 1: Lees snapshot bestanden
    let snapshot_data = std::fs::read("data/lucy.snapshot")
        .map_err(|e| format!("Kan snapshot niet lezen: {}", e))?;
    let vocab_data = std::fs::read("data/vocab.bincode")
        .map_err(|e| format!("Kan vocab niet lezen: {}", e))?;

    let total_size = snapshot_data.len() + vocab_data.len();
    tracing::info!(tick, bytes = total_size, "Pushing snapshot to GitHub...");

    // Stap 2: Maak een release
    let release_url = format!("{}/repos/{}/{}/releases", GITHUB_API, REPO_OWNER, REPO_NAME);
    let release_resp = client.post(&release_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "Lucy-AI")
        .json(&json!({
            "tag_name": tag,
            "name": format!("Lucy checkpoint tick {}", tick),
            "body": format!("Automatic snapshot at tick {}. Field + vocab.", tick),
            "draft": false,
            "prerelease": false,
        }))
        .send()
        .await
        .map_err(|e| format!("GitHub release API fout: {}", e))?;

    if !release_resp.status().is_success() {
        let status = release_resp.status();
        let body = release_resp.text().await.unwrap_or_default();
        return Err(format!("GitHub release {} failed: {}", status, &body[..body.len().min(200)]));
    }

    let release_json: serde_json::Value = release_resp.json().await
        .map_err(|e| format!("Parse error: {}", e))?;
    let upload_url = release_json["upload_url"].as_str()
        .ok_or("Geen upload_url in response")?
        .replace("{?name,label}", "");

    // Stap 3: Upload snapshot als asset
    let _ = upload_asset(&client, &upload_url, token, "lucy.snapshot", &snapshot_data).await;
    let _ = upload_asset(&client, &upload_url, token, "vocab.bincode", &vocab_data).await;

    // Optioneel: explorer + retrieval
    if let Ok(data) = std::fs::read("data/explorer.bincode") {
        let _ = upload_asset(&client, &upload_url, token, "explorer.bincode", &data).await;
    }
    if let Ok(data) = std::fs::read("data/retrieval.bincode") {
        let _ = upload_asset(&client, &upload_url, token, "retrieval.bincode", &data).await;
    }

    tracing::info!(tick, tag, "Snapshot pushed to GitHub successfully");
    Ok(tag)
}

async fn upload_asset(client: &Client, upload_url: &str, token: &str, name: &str, data: &[u8]) -> Result<(), String> {
    let url = format!("{}?name={}", upload_url, name);
    let resp = client.post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/octet-stream")
        .header("User-Agent", "Lucy-AI")
        .body(data.to_vec())
        .send()
        .await
        .map_err(|e| format!("Upload {} failed: {}", name, e))?;

    if resp.status().is_success() {
        tracing::debug!(name, "Asset uploaded");
        Ok(())
    } else {
        Err(format!("Upload {} status: {}", name, resp.status()))
    }
}

/// Haal de laatste snapshot op van GitHub (voor recovery).
pub async fn pull_latest_from_github(token: &str) -> Result<u64, String> {
    let client = Client::new();
    let url = format!("{}/repos/{}/{}/releases/latest", GITHUB_API, REPO_OWNER, REPO_NAME);

    let resp = client.get(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "Lucy-AI")
        .send()
        .await
        .map_err(|e| format!("GitHub API fout: {}", e))?;

    if !resp.status().is_success() {
        return Err("Geen releases gevonden".to_string());
    }

    let release: serde_json::Value = resp.json().await
        .map_err(|e| format!("Parse error: {}", e))?;

    let tag = release["tag_name"].as_str().unwrap_or("unknown");
    let tick: u64 = tag.strip_prefix("tick-").and_then(|s| s.parse().ok()).unwrap_or(0);

    // Download assets
    if let Some(assets) = release["assets"].as_array() {
        for asset in assets {
            let name = asset["name"].as_str().unwrap_or("");
            let download_url = asset["browser_download_url"].as_str().unwrap_or("");

            if !download_url.is_empty() && (name == "lucy.snapshot" || name == "vocab.bincode"
                || name == "explorer.bincode" || name == "retrieval.bincode") {
                let data = client.get(download_url)
                    .header("User-Agent", "Lucy-AI")
                    .send()
                    .await
                    .map_err(|e| format!("Download {} failed: {}", name, e))?
                    .bytes()
                    .await
                    .map_err(|e| format!("Read {} failed: {}", name, e))?;

                std::fs::write(format!("data/{}", name), &data)
                    .map_err(|e| format!("Write {} failed: {}", name, e))?;

                tracing::info!(name, bytes = data.len(), "Downloaded from GitHub");
            }
        }
    }

    tracing::info!(tick, tag, "Snapshot recovered from GitHub");
    Ok(tick)
}
