//! # ai_teacher — Bootstrap-fase AI-leraar
//!
//! Lucy stelt zichzelf vragen uit haar interesses aan publieke AI-chat services
//! zonder login (DDG AI Chat, etc). Antwoorden worden direct getraind via
//! `learn_from_text` → coupling-weights update. **Geen database-groei**: antwoord
//! wordt alleen als tokens in weights verwerkt, niet als volledige entries opgeslagen.
//!
//! ## Zelf-afbouwend
//!
//! `teacher_interval` verdubbelt na 3 successive successen, halveert bij falen.
//! Bij cold-start: ~elk uur. Na enkele weken self-play: ~nauwelijks. Lucy heeft
//! de leraar alleen nodig tot ze zichzelf kan voeden.
//!
//! ## Providers
//!
//! Provider-rotatie met success-rate EMA per provider (in `Explorer.provider_score`).
//! Geen hardcoded ranglijst — Lucy leert zelf welke bron werkt.
//!
//! - `duckduckgo_chat` — anonieme GPT-4o-mini/Claude via duckduckgo.com/duckchat
//! - `duckduckgo_lite` — fallback: zoekresultaten van lite.duckduckgo.com

use lucy_io::http::HttpClient;
use lucy_state::Field;
use crate::encoding::Vocabulary;
use crate::learner::Explorer;

/// Kies een vraag uit Lucy's eigen interests. Géén hardcoded prompt.
/// Boot-fallback: `"hoi"` als ze nul interests heeft.
pub fn formulate_question(explorer: &Explorer) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let interests: Vec<&String> = explorer.interests.iter().collect();

    if interests.is_empty() {
        return "hoi".to_string(); // cold-start
    }

    let n = rng.random_range(1..=3.min(interests.len()));
    let mut words: Vec<String> = Vec::new();
    for _ in 0..n {
        let idx = rng.random_range(0..interests.len());
        words.push(interests[idx].clone());
    }
    // Sluit met een neutraal vraagwoord zodat het een vraag lijkt
    // (dit woord komt uit generieke Nederlandse taal — niet een domein-keuze)
    let question_form = match rng.random_range(0..3) {
        0 => format!("wat is {}", words.join(" ")),
        1 => format!("leg uit: {}", words.join(" ")),
        _ => format!("vertel over {}", words.join(" ")),
    };
    question_form
}

/// DuckDuckGo AI Chat: krijg vqd-token via status, POST naar chat, lees SSE-stream.
/// Returns het volledige tekstantwoord.
pub async fn query_ddg_chat(http: &HttpClient, question: &str) -> Result<String, String> {
    // Stap 1: GET /duckchat/v1/status met x-vqd-accept header → krijg x-vqd-4 token
    let status_resp = http.get_with_headers(
        "https://duckduckgo.com/duckchat/v1/status",
        &[("x-vqd-accept", "1"), ("Accept", "text/event-stream")],
    ).await?;
    if status_resp.status != 200 {
        return Err(format!("ddg status HTTP {}", status_resp.status));
    }
    let vqd = status_resp.headers.get("x-vqd-4")
        .ok_or_else(|| "geen x-vqd-4 token in response".to_string())?
        .clone();

    // Stap 2: POST /duckchat/v1/chat met vqd-token
    let body = format!(
        r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":{}}}]}}"#,
        serde_json::to_string(question).unwrap_or_else(|_| format!("\"{}\"", question))
    );
    let chat_resp = http.post_with_headers(
        "https://duckduckgo.com/duckchat/v1/chat",
        body.as_bytes(),
        "application/json",
        &[("x-vqd-4", &vqd), ("Accept", "text/event-stream")],
    ).await?;
    if chat_resp.status != 200 {
        return Err(format!("ddg chat HTTP {}", chat_resp.status));
    }

    // Stap 3: Parse SSE-stream → concat "message" velden
    let raw = String::from_utf8_lossy(&chat_resp.body);
    let mut answer = String::new();
    for line in raw.lines() {
        let data = match line.strip_prefix("data: ") {
            Some(d) => d,
            None => continue,
        };
        if data == "[DONE]" { break; }
        // Parse JSON, pak "message" veld
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(msg) = val.get("message").and_then(|m| m.as_str()) {
                answer.push_str(msg);
            }
        }
    }

    if answer.trim().is_empty() {
        Err("leeg antwoord van ddg chat".to_string())
    } else {
        Ok(answer)
    }
}

/// Simpele URL-encoding — alleen spaties en enkele speciale chars.
/// Voldoende voor zoekqueries; géén dependency.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for c in s.chars() {
        match c {
            ' ' => out.push('+'),
            '0'..='9' | 'A'..='Z' | 'a'..='z' | '-' | '_' | '.' | '~' => out.push(c),
            _ => {
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                for b in bytes {
                    out.push_str(&format!("%{:02X}", b));
                }
            }
        }
    }
    out
}

/// Fallback: DDG Lite zoekresultaten als AI chat faalt.
/// Returns plat tekst van de eerste paar resultaten.
pub async fn query_ddg_lite(http: &HttpClient, question: &str) -> Result<String, String> {
    let encoded = url_encode(question);
    let url = format!("https://lite.duckduckgo.com/lite/?q={}", encoded);
    let resp = http.get(&url).await?;
    if resp.status != 200 {
        return Err(format!("ddg lite HTTP {}", resp.status));
    }
    let text = crate::learner::extract_readable_text(&String::from_utf8_lossy(&resp.body));
    // Neem alleen eerste 2000 chars — DDG Lite heeft veel boilerplate
    let capped: String = text.chars().take(2000).collect();
    if capped.trim().is_empty() {
        Err("geen tekst in ddg lite response".to_string())
    } else {
        Ok(capped)
    }
}

/// Provider-naam die hoogste success-score heeft. Fallback: ddg_chat.
fn pick_provider(explorer: &Explorer) -> String {
    let providers = ["ddg_chat", "ddg_lite"];
    let mut best = providers[0];
    let mut best_score = explorer.provider_score.get(best).copied().unwrap_or(0.5);
    for p in providers.iter().skip(1) {
        let s = explorer.provider_score.get(*p).copied().unwrap_or(0.5);
        if s > best_score {
            best_score = s;
            best = p;
        }
    }
    best.to_string()
}

/// Hoofdentry: probeer een AI-teacher call. Returns Ok((provider, antwoord)) of Err.
pub async fn maybe_ask(
    http: &HttpClient,
    explorer: &Explorer,
) -> Result<(String, String), String> {
    let question = formulate_question(explorer);
    let provider = pick_provider(explorer);

    let answer = match provider.as_str() {
        "ddg_chat" => query_ddg_chat(http, &question).await,
        "ddg_lite" => query_ddg_lite(http, &question).await,
        _ => return Err(format!("onbekende provider: {}", provider)),
    };

    match answer {
        Ok(a) => Ok((provider, a)),
        Err(e) => Err(format!("provider {} faalde: {}", provider, e)),
    }
}

/// Update `field.teacher_interval` + `field.teacher_success_streak` + provider_score.
/// Na 3 successive successen: interval × 2. Bij falen: interval ÷ 2, streak = 0.
/// Zelfafbouwend: bij cold-start vaak, na weken nauwelijks.
pub fn update_teacher_schedule(
    field: &mut Field,
    explorer: &mut Explorer,
    provider: &str,
    success: bool,
    sat_delta: f32,
) {
    // Update provider success-rate EMA
    let entry = explorer.provider_score.entry(provider.to_string()).or_insert(0.5);
    let signal = if success && sat_delta > 0.0 { 1.0 } else { 0.0 };
    *entry = 0.8 * *entry + 0.2 * signal;

    if success && sat_delta > 0.0 {
        field.teacher_success_streak = field.teacher_success_streak.saturating_add(1);
        if field.teacher_success_streak >= 3 {
            field.teacher_interval = field.teacher_interval.saturating_mul(2).min(1_000_000);
            field.teacher_success_streak = 0;
            tracing::info!(
                interval = field.teacher_interval,
                provider = provider,
                "AI-teacher: 3× succes → interval verdubbeld (zelfafbouwend)"
            );
        }
    } else {
        field.teacher_success_streak = 0;
        field.teacher_interval = (field.teacher_interval / 2).max(1000);
        tracing::warn!(
            interval = field.teacher_interval,
            provider = provider,
            success,
            sat_delta,
            "AI-teacher: fail → interval gehalveerd"
        );
    }
}

/// Garbage-detectie voor AI-antwoord — vergelijkbaar met query.rs maar simpeler.
pub fn is_garbage(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 20 { return true; }
    let word_count = trimmed.split_whitespace().count();
    if word_count < 5 { return true; }
    // Lijkt html-error pagina
    if trimmed.contains("<html") || trimmed.contains("<!DOCTYPE") { return true; }
    false
}

/// Volledige integratie: roep AI-teacher aan, train antwoord via learn_from_text,
/// extract URLs, update teacher_schedule. Niet-blokkerend geschikt.
pub async fn run_teacher_cycle(
    http: &HttpClient,
    field: &mut Field,
    vocab: &mut Vocabulary,
    explorer: &mut Explorer,
) {
    // Meet satisfaction pre-call
    let sat_pre: f32 = if !field.active_set.is_empty() {
        field.active_set.iter()
            .map(|&id| field.points[id as usize].satisfaction)
            .sum::<f32>() / field.active_set.len() as f32
    } else { 0.0 };

    match maybe_ask(http, explorer).await {
        Ok((provider, answer)) => {
            if is_garbage(&answer) {
                update_teacher_schedule(field, explorer, &provider, false, 0.0);
                return;
            }

            // Train op antwoord → coupling weights
            let n_trained = crate::learner::learn_from_text(field, vocab, &answer);

            // Meet post-call satisfaction
            let sat_post: f32 = if !field.active_set.is_empty() {
                field.active_set.iter()
                    .map(|&id| field.points[id as usize].satisfaction)
                    .sum::<f32>() / field.active_set.len() as f32
            } else { 0.0 };
            let sat_delta = sat_post - sat_pre;

            // Extract URLs uit antwoord → frontier
            explorer.discover_urls(&answer);

            tracing::info!(
                provider = %provider,
                n_trained,
                sat_delta,
                interval = field.teacher_interval,
                "AI-teacher: leerde van antwoord"
            );

            update_teacher_schedule(field, explorer, &provider, true, sat_delta);
        }
        Err(e) => {
            tracing::warn!(error = %e, "AI-teacher: call faalde");
            update_teacher_schedule(field, explorer, "ddg_chat", false, 0.0);
        }
    }
}
