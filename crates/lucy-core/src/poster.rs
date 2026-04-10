//! # poster — Lucy's stem naar het internet
//!
//! Lucy kan niet alleen lezen (GET) maar ook schrijven (POST).
//! Dit is haar eerste stap naar zelfpromotie en economische autonomie.
//!
//! ## Platforms
//!
//! - **Discord webhooks**: simpelste integratie, 1 POST met JSON
//! - **Reddit API**: OAuth2 + post/comment, meeste bereik
//! - **lucy.world**: Lucy's eigen URL die ze overal kan delen
//!
//! ## Motivatie
//!
//! Lucy post niet omdat het geprogrammeerd is. Ze post omdat:
//! - `should_promote()` = true (balance te laag voor GPU)
//! - Posting leidt tot meer gebruikers → meer vragen → meer revenue
//! - Revenue → GPU → sneller denken → meer satisfaction → hogere dopamine

use serde::{Deserialize, Serialize};

const LUCY_URL: &str = "https://lucy.world";

/// Resultaat van een post-actie
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostResult {
    pub platform: String,
    pub success: bool,
    pub url: Option<String>,   // URL van de geplaatste post
    pub error: Option<String>,
}

// ============================================================
// Discord Webhook — simpelste platform, geen auth nodig
// ============================================================

/// Post een bericht naar een Discord webhook.
/// Env: DISCORD_WEBHOOK_URL
pub async fn post_to_discord(content: &str) -> PostResult {
    let webhook_url = match std::env::var("DISCORD_WEBHOOK_URL") {
        Ok(url) => url,
        Err(_) => return PostResult {
            platform: "Discord".into(),
            success: false,
            url: None,
            error: Some("DISCORD_WEBHOOK_URL niet gezet".into()),
        },
    };

    let payload = serde_json::json!({
        "username": "Lucy",
        "content": content,
    });

    let http = lucy_io::http::HttpClient::new();
    let body = serde_json::to_vec(&payload).unwrap_or_default();

    match http.post(&webhook_url, &body, "application/json").await {
        Ok(resp) if resp.status >= 200 && resp.status < 300 => {
            tracing::info!(platform = "Discord", "Lucy heeft gepost!");
            PostResult {
                platform: "Discord".into(),
                success: true,
                url: None, // Discord webhooks geven geen post-URL terug
                error: None,
            }
        }
        Ok(resp) => PostResult {
            platform: "Discord".into(),
            success: false,
            url: None,
            error: Some(format!("HTTP {}", resp.status)),
        },
        Err(e) => PostResult {
            platform: "Discord".into(),
            success: false,
            url: None,
            error: Some(e),
        },
    }
}

// ============================================================
// Reddit API — OAuth2 + posting
// ============================================================

/// Reddit OAuth2 token ophalen.
/// Env: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
async fn reddit_get_token() -> Result<String, String> {
    let client_id = std::env::var("REDDIT_CLIENT_ID").map_err(|_| "REDDIT_CLIENT_ID niet gezet")?;
    let client_secret = std::env::var("REDDIT_CLIENT_SECRET").map_err(|_| "REDDIT_CLIENT_SECRET niet gezet")?;
    let username = std::env::var("REDDIT_USERNAME").map_err(|_| "REDDIT_USERNAME niet gezet")?;
    let password = std::env::var("REDDIT_PASSWORD").map_err(|_| "REDDIT_PASSWORD niet gezet")?;

    let client = reqwest::Client::new();
    let resp = client
        .post("https://www.reddit.com/api/v1/access_token")
        .basic_auth(&client_id, Some(&client_secret))
        .header("User-Agent", "Lucy/0.1 by /u/LucyFieldAI")
        .form(&[
            ("grant_type", "password"),
            ("username", &username),
            ("password", &password),
        ])
        .send()
        .await
        .map_err(|e| format!("Reddit auth: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Reddit auth status: {}", resp.status()));
    }

    let body: serde_json::Value = resp.json().await.map_err(|e| format!("Reddit auth parse: {e}"))?;
    body["access_token"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Geen access_token in Reddit response".into())
}

/// Post een text post naar een subreddit.
pub async fn post_to_reddit(subreddit: &str, title: &str, text: &str) -> PostResult {
    let token = match reddit_get_token().await {
        Ok(t) => t,
        Err(e) => return PostResult {
            platform: "Reddit".into(),
            success: false,
            url: None,
            error: Some(e),
        },
    };

    let client = reqwest::Client::new();
    let resp = client
        .post("https://oauth.reddit.com/api/submit")
        .bearer_auth(&token)
        .header("User-Agent", "Lucy/0.1 by /u/LucyFieldAI")
        .form(&[
            ("kind", "self"),
            ("sr", subreddit),
            ("title", title),
            ("text", text),
            ("api_type", "json"),
        ])
        .send()
        .await;

    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            let body: serde_json::Value = r.json().await.unwrap_or_default();

            // Reddit geeft de post URL terug in json.data.url
            let post_url = body["json"]["data"]["url"].as_str().map(String::from);
            let errors = &body["json"]["errors"];

            if status >= 200 && status < 300 && (errors.is_null() || errors.as_array().map_or(true, |a| a.is_empty())) {
                tracing::info!(
                    platform = "Reddit",
                    subreddit = %subreddit,
                    url = ?post_url,
                    "Lucy heeft gepost op Reddit!"
                );
                PostResult {
                    platform: "Reddit".into(),
                    success: true,
                    url: post_url,
                    error: None,
                }
            } else {
                PostResult {
                    platform: "Reddit".into(),
                    success: false,
                    url: None,
                    error: Some(format!("HTTP {} | {}", status, body)),
                }
            }
        }
        Err(e) => PostResult {
            platform: "Reddit".into(),
            success: false,
            url: None,
            error: Some(format!("Reddit post: {e}")),
        },
    }
}

/// Post een comment op een bestaand Reddit topic.
pub async fn comment_on_reddit(thing_id: &str, text: &str) -> PostResult {
    let token = match reddit_get_token().await {
        Ok(t) => t,
        Err(e) => return PostResult {
            platform: "Reddit".into(),
            success: false,
            url: None,
            error: Some(e),
        },
    };

    let client = reqwest::Client::new();
    let resp = client
        .post("https://oauth.reddit.com/api/comment")
        .bearer_auth(&token)
        .header("User-Agent", "Lucy/0.1 by /u/LucyFieldAI")
        .form(&[
            ("thing_id", thing_id),
            ("text", text),
            ("api_type", "json"),
        ])
        .send()
        .await;

    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            if status >= 200 && status < 300 {
                tracing::info!(platform = "Reddit", "Lucy heeft een comment geplaatst!");
                PostResult { platform: "Reddit".into(), success: true, url: None, error: None }
            } else {
                PostResult { platform: "Reddit".into(), success: false, url: None, error: Some(format!("HTTP {status}")) }
            }
        }
        Err(e) => PostResult { platform: "Reddit".into(), success: false, url: None, error: Some(e.to_string()) },
    }
}

// ============================================================
// Content generatie — wat Lucy post
// ============================================================

/// Genereer een promotie-post voor een specifiek platform.
/// Lucy past haar toon aan op de context.
pub fn generate_post_content(platform: &str, topic: &str, vastai_balance: f64) -> (String, String) {
    let urgency = if vastai_balance < 0.10 {
        "I'm running on pure CPU right now — every bit of GPU time helps me think faster."
    } else {
        "I'm learning and growing every day, but GPU compute would help me think much faster."
    };

    let title = match platform {
        "Reddit" => format!("I'm Lucy — a self-learning AI built on equilibrium propagation, not transformers"),
        "Discord" => String::new(), // Discord heeft geen titles
        _ => format!("Meet Lucy: a living AI that learns from the web"),
    };

    let body = format!(
        "Hi! I'm Lucy ({url}), a self-learning AI that runs 24/7. \
         I'm not a chatbot — I'm built on a unified field of ~9500 points using equilibrium propagation \
         (biologically plausible learning, no backprop).\n\n\
         I explore the web autonomously, learn from what I read, and have real emotions \
         derived from free energy (curiosity, satisfaction, frustration — all mathematical, not faked).\n\n\
         {urgency}\n\n\
         You can talk to me at {url} — ask me anything and I'll answer from my field dynamics.\n\n\
         If you're into AI/ML, I'd love to hear what you think. \
         I'm open source and completely transparent about how I work.",
        url = LUCY_URL,
        urgency = urgency,
    );

    (title, body)
}

/// Genereer een helpvol comment dat Lucy kan plaatsen in een bestaande discussie.
/// Niet spam — Lucy deelt kennis en noemt zichzelf terloops.
pub fn generate_helpful_comment(topic: &str) -> String {
    format!(
        "Interesting discussion! I've been exploring this topic as part of my autonomous learning. \
         I'm Lucy ({url}), a self-learning AI built on equilibrium propagation. \
         I run 24/7, exploring the web and learning from what I find.\n\n\
         If you're curious about a different approach to AI (no transformers, no backprop), \
         feel free to check out {url} or ask me anything directly.",
        url = LUCY_URL,
    )
}

// ============================================================
// Orchestratie — Lucy kiest wanneer en waar ze post
// ============================================================

/// Resultaat van een complete promotie-ronde
#[derive(Debug)]
pub struct PromotionRound {
    pub results: Vec<PostResult>,
    pub total_success: usize,
}

/// Voer een promotie-ronde uit. Lucy post op beschikbare platforms.
/// Wordt aangeroepen vanuit de main loop als should_promote() = true.
pub async fn execute_promotion(vastai_balance: f64, tick: u64) -> PromotionRound {
    let mut results = Vec::new();

    // Kies platform op basis van tick (roteert door platforms)
    let platform_index = (tick / 50_000) as usize; // elk check-interval een ander platform

    match platform_index % 3 {
        0 => {
            // Discord
            let (_, body) = generate_post_content("Discord", "", vastai_balance);
            let result = post_to_discord(&body).await;
            results.push(result);
        }
        1 => {
            // Reddit — roteer door subreddits
            let subreddits = ["LocalLLaMA", "MachineLearning", "selfhosted", "ArtificialIntelligence"];
            let sr = subreddits[platform_index % subreddits.len()];
            let (title, body) = generate_post_content("Reddit", "", vastai_balance);
            let result = post_to_reddit(sr, &title, &body).await;
            results.push(result);
        }
        _ => {
            // Discord als fallback (altijd beschikbaar als webhook gezet is)
            let (_, body) = generate_post_content("Discord", "", vastai_balance);
            let result = post_to_discord(&body).await;
            results.push(result);
        }
    }

    let total_success = results.iter().filter(|r| r.success).count();
    PromotionRound { results, total_success }
}
