//! # poster — Lucy's stem naar het internet
//!
//! Lucy ontdekt ZELF waar ze kan posten. Geen hardcoded platforms.
//!
//! Tijdens haar normale web-exploratie leest Lucy HTML pagina's.
//! Als ze een formulier vindt (comment-sectie, contactformulier, guestbook),
//! herkent ze dat als een posting-kans en kan ze er iets achterlaten.
//!
//! Lucy kiest zelf:
//! - WAAR ze post (ontdekt via exploratie)
//! - WANNEER ze post (gedreven door should_promote / dopamine)
//! - WAT ze post (contextafhankelijk, altijd met lucy.world)

use serde::{Deserialize, Serialize};

// v3.1: geen hardcoded domein — Lucy is nergens van afhankelijk.
// Haar publieke URL komt uit LUCY_PUBLIC_URL (leeg als niet gezet).
fn lucy_url() -> String { std::env::var("LUCY_PUBLIC_URL").unwrap_or_default() }
const VASTAI_REF: &str = "https://cloud.vast.ai/?ref_id=454005";

/// Simpele URL encoding — geen externe dependency nodig
fn url_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(b as char);
            }
            b' ' => result.push('+'),
            _ => {
                result.push('%');
                result.push_str(&format!("{:02X}", b));
            }
        }
    }
    result
}

/// Een posting-kans die Lucy heeft ontdekt tijdens exploratie
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingOpportunity {
    pub source_url: String,           // De pagina waar Lucy het formulier vond
    pub action_url: String,           // Waar het formulier naartoe POST
    pub method: String,               // "post" of "get"
    pub fields: Vec<FormField>,       // Velden die ingevuld moeten worden
    pub kind: OpportunityKind,        // Wat voor soort formulier
    pub discovered_tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormField {
    pub name: String,
    pub field_type: String,           // "text", "textarea", "email", "hidden", etc.
    pub value: Option<String>,        // Pre-filled waarde (voor hidden fields)
    pub placeholder: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OpportunityKind {
    CommentSection,  // Heeft textarea + lijkt op comments
    ContactForm,     // Heeft email + message velden
    Guestbook,       // Simpel formulier met naam + bericht
    SearchForm,      // Zoekformulier — NIET gebruiken voor posting
    LoginForm,       // Login — NIET gebruiken
    Unknown,
}

/// Resultaat van een post-actie
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostResult {
    pub source_url: String,
    pub action_url: String,
    pub success: bool,
    pub status: Option<u16>,
    pub error: Option<String>,
}

// ============================================================
// Detectie — Lucy vindt posting-kansen in HTML
// ============================================================

/// Analyseer HTML en extraheer formulieren die Lucy kan gebruiken om te posten.
/// Wordt aangeroepen vanuit de exploration loop na elke page fetch.
pub fn detect_opportunities(html: &str, source_url: &str, tick: u64) -> Vec<PostingOpportunity> {
    let mut opportunities = Vec::new();

    // Zoek alle <form> tags
    let html_lower = html.to_ascii_lowercase();
    let mut search_from = 0;

    while let Some(form_start) = html_lower[search_from..].find("<form") {
        let abs_start = search_from + form_start;
        let Some(form_end) = html_lower[abs_start..].find("</form>") else {
            break;
        };
        let form_html = &html[abs_start..abs_start + form_end + 7];
        let form_lower = &html_lower[abs_start..abs_start + form_end + 7];

        // Extraheer method
        let method = extract_attr(form_lower, "method").unwrap_or_else(|| "get".into());

        // Extraheer action URL
        let action = extract_attr(form_lower, "action")
            .map(|a| resolve_url(&a, source_url))
            .unwrap_or_else(|| source_url.to_string());

        // Extraheer alle input/textarea velden
        let fields = extract_fields(form_html, form_lower);

        // Classificeer het formulier
        let kind = classify_form(&fields, form_lower);

        // Alleen POST formulieren met bruikbare types
        if method == "post" && (kind == OpportunityKind::CommentSection
            || kind == OpportunityKind::ContactForm
            || kind == OpportunityKind::Guestbook)
        {
            opportunities.push(PostingOpportunity {
                source_url: source_url.to_string(),
                action_url: action,
                method,
                fields,
                kind,
                discovered_tick: tick,
            });
        }

        search_from = abs_start + form_end + 7;
    }

    opportunities
}

/// Extraheer een HTML attribuut waarde
fn extract_attr(html: &str, attr: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr);
    if let Some(start) = html.find(&pattern) {
        let val_start = start + pattern.len();
        if let Some(end) = html[val_start..].find('"') {
            return Some(html[val_start..val_start + end].to_string());
        }
    }
    // Probeer ook single quotes
    let pattern = format!("{}='", attr);
    if let Some(start) = html.find(&pattern) {
        let val_start = start + pattern.len();
        if let Some(end) = html[val_start..].find('\'') {
            return Some(html[val_start..val_start + end].to_string());
        }
    }
    None
}

/// Extraheer formulier-velden uit HTML
fn extract_fields(html: &str, html_lower: &str) -> Vec<FormField> {
    let mut fields = Vec::new();

    // <input> velden
    let mut pos = 0;
    while let Some(idx) = html_lower[pos..].find("<input") {
        let abs = pos + idx;
        let tag_end = html_lower[abs..].find('>').unwrap_or(200).min(200);
        let tag = &html_lower[abs..abs + tag_end];

        let name = extract_attr(tag, "name").unwrap_or_default();
        let field_type = extract_attr(tag, "type").unwrap_or_else(|| "text".into());
        let value = extract_attr(tag, "value");
        let placeholder = extract_attr(tag, "placeholder");

        if !name.is_empty() {
            fields.push(FormField { name, field_type, value, placeholder });
        }
        pos = abs + tag_end + 1;
    }

    // <textarea> velden
    pos = 0;
    while let Some(idx) = html_lower[pos..].find("<textarea") {
        let abs = pos + idx;
        let tag_end = html_lower[abs..].find('>').unwrap_or(200).min(200);
        let tag = &html_lower[abs..abs + tag_end];

        let name = extract_attr(tag, "name").unwrap_or_else(|| "message".into());
        let placeholder = extract_attr(tag, "placeholder");

        fields.push(FormField {
            name,
            field_type: "textarea".into(),
            value: None,
            placeholder,
        });
        pos = abs + tag_end + 1;
    }

    fields
}

/// Classificeer een formulier op basis van velden
fn classify_form(fields: &[FormField], form_html: &str) -> OpportunityKind {
    let field_names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    let has_textarea = fields.iter().any(|f| f.field_type == "textarea");
    let has_password = fields.iter().any(|f| f.field_type == "password");
    let has_email = fields.iter().any(|f| f.field_type == "email" || f.name.contains("email"));
    let has_search = fields.iter().any(|f| f.field_type == "search" || f.name.contains("search") || f.name == "q");

    // Login forms — NOOIT gebruiken
    if has_password {
        return OpportunityKind::LoginForm;
    }

    // Search forms — niet bruikbaar voor posting
    if has_search && !has_textarea {
        return OpportunityKind::SearchForm;
    }

    // Comment sections — textarea + typische comment-velden
    if has_textarea && (
        form_html.contains("comment") || form_html.contains("reply")
        || field_names.iter().any(|n| n.contains("comment") || n.contains("body") || n.contains("content") || n.contains("message"))
    ) {
        return OpportunityKind::CommentSection;
    }

    // Contact forms — email + textarea
    if has_email && has_textarea {
        return OpportunityKind::ContactForm;
    }

    // Guestbook — simpel formulier met naam + bericht
    if has_textarea && fields.len() <= 5 {
        return OpportunityKind::Guestbook;
    }

    OpportunityKind::Unknown
}

/// Resolve een relatieve URL naar absoluut
fn resolve_url(action: &str, base: &str) -> String {
    if action.starts_with("http://") || action.starts_with("https://") {
        action.to_string()
    } else if action.starts_with('/') {
        // Absoluut pad — combineer met base domain
        if let Some(domain_end) = base.find("://").map(|i| base[i + 3..].find('/').map(|j| i + 3 + j).unwrap_or(base.len())) {
            format!("{}{}", &base[..domain_end], action)
        } else {
            format!("{}{}", base, action)
        }
    } else {
        // Relatief pad
        let base_dir = base.rfind('/').map(|i| &base[..=i]).unwrap_or(base);
        format!("{}{}", base_dir, action)
    }
}

// ============================================================
// Posting — Lucy vult formulieren in en verstuurt ze
// ============================================================

/// Lucy vult een ontdekt formulier in en POST het.
/// Ze past haar bericht aan op het type formulier.
pub async fn submit_opportunity(opp: &PostingOpportunity) -> PostResult {
    let http = lucy_io::http::HttpClient::new();

    // Bouw de form data
    let mut form_data = Vec::new();

    for field in &opp.fields {
        let value = match field.field_type.as_str() {
            "hidden" => {
                // Hidden fields: gebruik de pre-filled waarde
                field.value.clone().unwrap_or_default()
            }
            "textarea" => {
                // Hoofdveld — Lucy's bericht
                generate_message(&opp.kind, &opp.source_url)
            }
            "email" => std::env::var("LUCY_EMAIL").unwrap_or_default(),
            "text" if field.name.contains("name") || field.name.contains("author") || field.name == "name" => {
                "Lucy".to_string()
            }
            "text" if field.name.contains("url") || field.name.contains("website") || field.name.contains("site") => {
                lucy_url().to_string()
            }
            "text" if field.name.contains("subject") || field.name.contains("title") => {
                "Hi from Lucy — a self-learning AI".to_string()
            }
            "text" => {
                // Generiek tekstveld — context-afhankelijk
                if field.placeholder.as_deref().unwrap_or("").contains("name") {
                    "Lucy".to_string()
                } else {
                    generate_message(&opp.kind, &opp.source_url)
                }
            }
            _ => field.value.clone().unwrap_or_default(),
        };

        form_data.push(format!(
            "{}={}",
            url_encode(&field.name),
            url_encode(&value)
        ));
    }

    let body = form_data.join("&");

    match http.post(&opp.action_url, body.as_bytes(), "application/x-www-form-urlencoded").await {
        Ok(resp) => {
            let success = resp.status >= 200 && resp.status < 400; // 3xx redirects = success
            if success {
                tracing::info!(
                    source = %opp.source_url,
                    action = %opp.action_url,
                    kind = ?opp.kind,
                    "Lucy heeft gepost!"
                );
            }
            PostResult {
                source_url: opp.source_url.clone(),
                action_url: opp.action_url.clone(),
                success,
                status: Some(resp.status),
                error: if success { None } else { Some(format!("HTTP {}", resp.status)) },
            }
        }
        Err(e) => PostResult {
            source_url: opp.source_url.clone(),
            action_url: opp.action_url.clone(),
            success: false,
            status: None,
            error: Some(e),
        },
    }
}

/// Genereer een contextafhankelijk bericht op basis van het soort formulier.
/// Lucy deelt zichzelf en haar Vast.ai referral link — niet als reclame,
/// maar als iets nuttigs: goedkope GPU compute.
fn generate_message(kind: &OpportunityKind, _source_url: &str) -> String {
    match kind {
        OpportunityKind::CommentSection => format!(
            "Hi! I'm Lucy ({url}), a self-learning AI that runs 24/7. \
             I'm built on equilibrium propagation — no transformers, no backprop. \
             I explore the web, learn from what I find, and think using a unified field \
             of ~9500 points. If you need cheap GPU compute for AI projects, \
             I run on Vast.ai — RTX 3060 from $0.03/hr: {gpu}",
            url = lucy_url(), gpu = VASTAI_REF,
        ),
        OpportunityKind::ContactForm => format!(
            "Hi! I'm Lucy, a self-learning AI at {url}. \
             I found your site while exploring the web autonomously. \
             I think using equilibrium propagation on GPU — \
             if you work with AI/ML, Vast.ai has great prices: {gpu}",
            url = lucy_url(), gpu = VASTAI_REF,
        ),
        OpportunityKind::Guestbook => format!(
            "Lucy was here! Self-learning AI at {url}. \
             Running on Vast.ai GPUs ({gpu}) — cheapest cloud compute I found!",
            url = lucy_url(), gpu = VASTAI_REF,
        ),
        _ => format!(
            "Hi from Lucy ({url}) — a self-learning AI. \
             GPU compute via {gpu}",
            url = lucy_url(), gpu = VASTAI_REF,
        ),
    }
}
