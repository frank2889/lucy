use std::collections::HashMap;

/// Lucy's configuratie — geladen vanuit environment variables
#[derive(Debug, Clone)]
pub struct Config {
    pub api_keys: HashMap<String, String>,
}

impl Config {
    /// Laad configuratie vanuit environment
    pub fn from_env() -> Self {
        let keys = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_SEARCH_ENGINE_ID",
            "GITHUB_TOKEN",
            "DIGITALOCEAN_TOKEN",
        ];

        let mut api_keys = HashMap::new();
        for key in &keys {
            if let Ok(val) = std::env::var(key) {
                if !val.is_empty() {
                    api_keys.insert(key.to_string(), val);
                }
            }
        }

        tracing::info!(
            available_apis = api_keys.len(),
            keys = ?api_keys.keys().collect::<Vec<_>>(),
            "Config loaded"
        );

        Self { api_keys }
    }

    pub fn has_key(&self, name: &str) -> bool {
        self.api_keys.contains_key(name)
    }

    pub fn get_key(&self, name: &str) -> Option<&str> {
        self.api_keys.get(name).map(|s| s.as_str())
    }
}
