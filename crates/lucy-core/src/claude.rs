use lucy_io::http::HttpClient;

/// Vraag Claude om een antwoord — Lucy's leraar
/// Retourneert (antwoord, kosten_in_centen)
pub async fn ask_claude(
    http: &HttpClient,
    api_key: &str,
    question: &str,
) -> Result<(String, i64), String> {
    let body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": question
        }]
    });

    let body_bytes = serde_json::to_vec(&body).map_err(|e| e.to_string())?;

    // We moeten custom headers sturen, maar de bestaande HttpClient ondersteunt
    // alleen Content-Type. Bouw een directe reqwest call.
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| format!("Claude request failed: {}", e))?;

    let latency = start.elapsed().as_millis();
    let status = resp.status().as_u16();

    let resp_body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Claude response parse failed: {}", e))?;

    if status != 200 {
        let error_msg = resp_body["error"]["message"]
            .as_str()
            .unwrap_or("Unknown error");
        return Err(format!("Claude API error {}: {}", status, error_msg));
    }

    // Extract antwoord
    let answer = resp_body["content"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string();

    // Bereken kosten (Sonnet: $3/M input, $15/M output)
    let input_tokens = resp_body["usage"]["input_tokens"].as_i64().unwrap_or(0);
    let output_tokens = resp_body["usage"]["output_tokens"].as_i64().unwrap_or(0);
    let cost_microdollars = input_tokens * 3 + output_tokens * 15;
    let cost_cents = (cost_microdollars as f64 / 10_000.0).ceil() as i64;
    let cost_cents = cost_cents.max(1); // minimaal 1 cent

    tracing::info!(
        latency_ms = latency,
        input_tokens,
        output_tokens,
        cost_cents,
        "Claude response received"
    );

    Ok((answer, cost_cents))
}
