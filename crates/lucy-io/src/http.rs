use reqwest::Client;
use std::collections::HashMap;

/// Lucy's universele HTTP client — haar zintuig naar het internet
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

    /// GET request — Lucy's primaire zintuig naar het internet
    pub async fn get(&self, url: &str) -> Result<HttpResponse, String> {
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
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(HttpResponse {
            status,
            headers,
            body,
            latency_ms,
        })
    }

    /// POST request — Lucy's handen naar het internet
    pub async fn post(&self, url: &str, body: &[u8], content_type: &str) -> Result<HttpResponse, String> {
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
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(HttpResponse {
            status,
            headers,
            body,
            latency_ms,
        })
    }
}
