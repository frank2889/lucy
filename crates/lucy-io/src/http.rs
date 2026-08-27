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
                // SSRF: hercontroleer ELKE redirect-hop (literal-IP + string-blocklist),
                // en cap het aantal hops. DNS-namen worden bij connect door reqwest
                // geresolved; de literal-check vangt directe redirects naar interne IPs.
                .redirect(reqwest::redirect::Policy::custom(|attempt| {
                    if attempt.previous().len() >= 5 {
                        return attempt.error("te veel redirects (SSRF-guard)".to_string());
                    }
                    match attempt.url().host_str() {
                        Some(h) => match HttpClient::classify_host_literal(h) {
                            Ok(()) => attempt.follow(),
                            Err(e) => attempt.error(e),
                        },
                        None => attempt.error("redirect zonder host".to_string()),
                    }
                }))
                .build()
                .expect("HTTP client creation failed"),
        }
    }

    /// Classificeer een IP: Err als het niet-globaal is
    /// (loopback/private/link-local/ULA/CGNAT/unspecified/etc).
    fn classify_ip(ip: std::net::IpAddr) -> Result<(), String> {
        use std::net::IpAddr;
        let bad = match ip {
            IpAddr::V4(v4) => {
                let o = v4.octets();
                v4.is_loopback()
                    || v4.is_private()
                    || v4.is_link_local()
                    || v4.is_unspecified()
                    || v4.is_broadcast()
                    || v4.is_documentation()
                    || o[0] == 0
                    // 100.64.0.0/10 CGNAT
                    || (o[0] == 100 && (o[1] & 0xC0) == 64)
                    // 192.0.0.0/24 IETF protocol assignments
                    || (o[0] == 192 && o[1] == 0 && o[2] == 0)
            }
            IpAddr::V6(v6) => {
                // IPv4-mapped (::ffff:a.b.c.d) via de v4-regels beoordelen
                if let Some(v4) = v6.to_ipv4_mapped() {
                    return Self::classify_ip(IpAddr::V4(v4));
                }
                let seg0 = v6.segments()[0];
                v6.is_loopback()
                    || v6.is_unspecified()
                    || (seg0 & 0xfe00) == 0xfc00 // fc00::/7 unique-local
                    || (seg0 & 0xffc0) == 0xfe80 // fe80::/10 link-local
            }
        };
        if bad {
            Err(format!("Interne/niet-globale IP geblokkeerd: {}", ip))
        } else {
            Ok(())
        }
    }

    /// Sync host-check: string-blocklist + literal-IP-classificatie. DNS-namen
    /// passeren hier (die worden in `validate_public_url` async geresolved). Wordt
    /// óók per redirect-hop aangeroepen door de redirect-policy.
    fn classify_host_literal(host: &str) -> Result<(), String> {
        let h = host.to_ascii_lowercase();
        if h == "localhost" || h.ends_with(".local") || h == "metadata.google.internal" {
            return Err("Interne host geblokkeerd".to_string());
        }
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            return Self::classify_ip(ip);
        }
        Ok(())
    }

    /// SSRF-guard (async): parse de URL, blokkeer interne literals, en RESOLVE de
    /// host via DNS — weiger als ook maar één geresolved IP niet-globaal is. Fail-
    /// closed als resolutie faalt/leeg is. Sluit de A-record->intern-IP-bypass die
    /// de oude string-only check openliet. Minimale eigen URL-parser (geen `url` crate).
    async fn validate_public_url(url: &str) -> Result<(), String> {
        let url = url.trim();
        let (scheme, rest) = url.split_once("://").ok_or("Bad URL: geen scheme")?;
        let default_port: u16 = match scheme.to_ascii_lowercase().as_str() {
            "http" => 80,
            "https" => 443,
            s => return Err(format!("Scheme niet toegestaan: {}", s)),
        };
        // authority = alles tot eerste '/', '?' of '#'
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        // strip userinfo (user:pass@)
        let hostport = match authority.rsplit_once('@') {
            Some((_, hp)) => hp,
            None => authority,
        };
        // host + optionele expliciete poort (IPv6 tussen [], anders tot laatste ':')
        let (host, explicit_port) = if let Some(after_bracket) = hostport.strip_prefix('[') {
            match after_bracket.split_once(']') {
                Some((h, tail)) => {
                    let p = tail.strip_prefix(':').and_then(|s| s.parse::<u16>().ok());
                    (h.to_string(), p)
                }
                None => return Err("Bad URL: ongeldige IPv6".to_string()),
            }
        } else {
            match hostport.rsplit_once(':') {
                Some((h, p)) if !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => {
                    (h.to_string(), p.parse::<u16>().ok())
                }
                _ => (hostport.to_string(), None),
            }
        };
        if host.is_empty() {
            return Err("Geen host".to_string());
        }
        Self::classify_host_literal(&host)?;
        let port = explicit_port.unwrap_or(default_port);
        // DNS-resolutie (async, non-blocking). Fail-closed bij fout of lege set.
        let addrs = tokio::net::lookup_host((host.as_str(), port))
            .await
            .map_err(|e| format!("DNS-resolutie faalde (fail-closed): {}", e))?;
        let mut any = false;
        for sa in addrs {
            any = true;
            Self::classify_ip(sa.ip())?;
        }
        if !any {
            return Err("Geen IP's geresolved (fail-closed)".to_string());
        }
        Ok(())
    }

    /// GET request — Lucy's primaire zintuig naar het internet
    pub async fn get(&self, url: &str) -> Result<HttpResponse, String> {
        Self::validate_public_url(url).await?;
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
        Self::validate_public_url(url).await?;
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
        Self::validate_public_url(url).await?;
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
        Self::validate_public_url(url).await?;
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
