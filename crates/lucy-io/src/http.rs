//! # http — Lucy's HTTP Client
//!
//! Wraps `reqwest` voor alle HTTP communicatie. Elke request is een
//! zintuiglijke ervaring: Lucy "ziet" het internet via GET en "handelt"
//! via POST.
//!
//! ## Kenmerken
//!
//! - Timeout: 30 seconden (voorkomt dat Lucy vastloopt op trage servers)
//! - User-Agent: "Lucy/0.1" (eerlijk over wie we zijn)
//! - Alle responses bevatten latency_ms voor performance monitoring
//!
//! ## Gebruik
//!
//! Primair door `learner.rs` voor autonome web-exploratie, en door
//! toekomstige modules die externe API's aanroepen.

use reqwest::Client;
use std::collections::HashMap;

/// Lucy's universele HTTP client — haar zintuig naar het internet.
pub struct HttpClient {
    client: Client,
}

/// Het resultaat van een HTTP request — wordt sensorische input voor het veld
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub latency_ms: u64,
}

impl HttpClient {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("Lucy/0.1")
                .build()
                .expect("HTTP client creation failed"),
        }
    }

    /// SSRF-guard: weiger loopback/private/link-local IPs en interne hostnames.
    /// Minimale eigen URL-parser (geen `url` crate dependency in lucy-io).
    fn is_public_url(url: &str) -> Result<(), String> {
        let url = url.trim();
        let (scheme, rest) = url.split_once("://").ok_or("Bad URL: geen scheme")?;
        match scheme.to_ascii_lowercase().as_str() {
            "http" | "https" => {}
            s => return Err(format!("Scheme niet toegestaan: {}", s)),
        }
        // authority = alles tot eerste '/', '?' of '#'
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        // strip userinfo (user:pass@)
        let hostport = match authority.rsplit_once('@') {
            Some((_, hp)) => hp,
            None => authority,
        };
        // host uitpakken (IPv6 tussen [], anders tot ':')
        let host = if let Some(after_bracket) = hostport.strip_prefix('[') {
            match after_bracket.split_once(']') {
                Some((h, _)) => h.to_string(),
                None => return Err("Bad URL: ongeldige IPv6".to_string()),
            }
        } else {
            hostport.split(':').next().unwrap_or("").to_string()
        };
        if host.is_empty() {
            return Err("Geen host".to_string());
        }
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            let blocked = ip.is_loopback() || ip.is_unspecified() || match ip {
                std::net::IpAddr::V4(v4) => v4.is_private() || v4.is_link_local(),
                std::net::IpAddr::V6(_) => false,
            };
            if blocked {
                return Err(format!("Interne IP geblokkeerd: {}", ip));
            }
        }
        let h = host.to_ascii_lowercase();
        if h == "localhost" || h.ends_with(".local") || h == "metadata.google.internal" {
            return Err("Interne host geblokkeerd".to_string());
        }
        Ok(())
    }

    /// GET request — Lucy's primaire zintuig naar het internet
    pub async fn get(&self, url: &str) -> Result<HttpResponse, String> {
        Self::is_public_url(url)?;
        let start = std::time::Instant::now();
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let status = resp.status().as_u16();
        let headers: HashMap<String, String> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let body = resp.bytes().await.map_err(|e| e.to_string())?.to_vec();
        if body.len() > 5_000_000 {
            return Err(format!("Response te groot: {} bytes (max 5MB)", body.len()));
        }
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(HttpResponse {
            status,
            headers,
            body,
            latency_ms,
        })
    }

    /// GET met custom headers (voor APIs die auth-tokens verwachten)
    pub async fn get_with_headers(&self, url: &str, extra_headers: &[(&str, &str)]) -> Result<HttpResponse, String> {
        Self::is_public_url(url)?;
        let start = std::time::Instant::now();
        let mut req = self.client.get(url);
        for (k, v) in extra_headers {
            req = req.header(*k, *v);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;

        let status = resp.status().as_u16();
        let headers: HashMap<String, String> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let body = resp.bytes().await.map_err(|e| e.to_string())?.to_vec();
        if body.len() > 5_000_000 {
            return Err(format!("Response te groot: {} bytes (max 5MB)", body.len()));
        }
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(HttpResponse { status, headers, body, latency_ms })
    }

    /// POST met custom headers (voor APIs zoals DDG AI Chat die vqd-tokens gebruiken)
    pub async fn post_with_headers(&self, url: &str, body: &[u8], content_type: &str, extra_headers: &[(&str, &str)]) -> Result<HttpResponse, String> {
        Self::is_public_url(url)?;
        let start = std::time::Instant::now();
        let mut req = self.client.post(url)
            .header("Content-Type", content_type)
            .body(body.to_vec());
        for (k, v) in extra_headers {
            req = req.header(*k, *v);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;

        let status = resp.status().as_u16();
        let headers: HashMap<String, String> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let body = resp.bytes().await.map_err(|e| e.to_string())?.to_vec();
        if body.len() > 5_000_000 {
            return Err(format!("Response te groot: {} bytes (max 5MB)", body.len()));
        }
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(HttpResponse { status, headers, body, latency_ms })
    }

    /// POST request — Lucy's handen naar het internet
    pub async fn post(&self, url: &str, body: &[u8], content_type: &str) -> Result<HttpResponse, String> {
        Self::is_public_url(url)?;
        let start = std::time::Instant::now();
        let resp = self
            .client
            .post(url)
            .header("Content-Type", content_type)
            .body(body.to_vec())
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let status = resp.status().as_u16();
        let headers: HashMap<String, String> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let body = resp.bytes().await.map_err(|e| e.to_string())?.to_vec();
        if body.len() > 5_000_000 {
            return Err(format!("Response te groot: {} bytes (max 5MB)", body.len()));
        }
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(HttpResponse {
            status,
            headers,
            body,
            latency_ms,
        })
    }
}
