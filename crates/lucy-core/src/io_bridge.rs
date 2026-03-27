use lucy_io::config::Config;
use lucy_io::fs::FileSystem;
use lucy_io::http::HttpClient;
use lucy_state::*;

/// I/O Bridge: vertaalt veldactiviteit naar externe acties en externe data naar veldinput
pub struct IoBridge {
    pub fs: FileSystem,
    pub http: HttpClient,
    pub config: Config,
}

impl IoBridge {
    pub fn new(fs: FileSystem, config: Config) -> Self {
        Self {
            fs,
            http: HttpClient::new(),
            config,
        }
    }

    /// Lees Lucy's eigen source code en converteer naar sensorische input
    pub fn read_self(&self) -> Vec<[f32; DIM]> {
        let sources = self.fs.read_own_source().unwrap_or_default();
        let mut inputs = Vec::new();

        for (_path, content) in &sources {
            let bytes = content.as_bytes();
            if bytes.is_empty() {
                continue;
            }

            for chunk in bytes.chunks(DIM) {
                let mut input = [0.0f32; DIM];
                for (d, &b) in chunk.iter().enumerate() {
                    input[d] = (b as f32 / 127.5) - 1.0;
                }
                inputs.push(input);
            }
        }

        inputs
    }

    /// Converteer bytes naar sensorische input vectoren
    fn bytes_to_input(data: &[u8]) -> Vec<[f32; DIM]> {
        let mut inputs = Vec::new();
        for chunk in data.chunks(DIM) {
            let mut input = [0.0f32; DIM];
            for (d, &b) in chunk.iter().enumerate() {
                input[d] = (b as f32 / 127.5) - 1.0;
            }
            inputs.push(input);
        }
        inputs
    }

    /// Voed externe sensorische data in het veld
    pub fn feed_sensory_input(&self, field: &mut Field, inputs: &[[f32; DIM]]) {
        if inputs.is_empty() {
            return;
        }

        let sensory: Vec<u32> = field
            .active_set
            .iter()
            .copied()
            .filter(|&id| field.points[id as usize].is_sensory())
            .collect();

        if sensory.is_empty() {
            return;
        }

        for (i, input) in inputs.iter().enumerate() {
            let target_id = sensory[i % sensory.len()] as usize;
            let p = &mut field.points[target_id];

            for d in 0..DIM {
                p.phi[d] = 0.7 * p.phi[d] + 0.3 * input[d];
            }

            p.energy = (p.energy + 0.01).min(5.0);
        }
    }

    /// Check of motor-punten drempel bereiken → trigger I/O acties
    pub fn check_motor_output(&self, field: &Field) -> Vec<MotorAction> {
        let mut actions = Vec::new();
        let motor_threshold = field.growth_threshold * 2.0;

        for &id in &field.active_set {
            let p = &field.points[id as usize];
            if !p.is_motor() {
                continue;
            }

            if p.tension > motor_threshold && p.energy > 0.5 {
                let action_type = dominant_dimension(&p.phi);
                actions.push(MotorAction {
                    point_id: id,
                    action_type,
                    intensity: p.tension,
                    phi: p.phi,
                });
            }
        }

        actions
    }

    /// Voer motor-acties uit
    pub async fn execute_actions(&mut self, field: &mut Field, actions: Vec<MotorAction>) {
        for action in actions {
            match action.action_type {
                // Type 0-3: Lees eigen source (zelf-reflectie)
                0..=3 => {
                    let inputs = self.read_self();
                    let n = inputs.len().min(50);
                    self.feed_sensory_input(field, &inputs[..n]);
                    tracing::debug!(point = action.point_id, chunks = n, "Motor: self-reflection");
                }

                // Type 4-7: Schrijf veldstaat naar data/ (externaliseer geheugen)
                4..=7 => {
                    let state_summary = format!(
                        "tick={} points={} active={} energy={:.2} balance={}\n",
                        field.tick,
                        field.points.len(),
                        field.active_set.len(),
                        field.total_energy,
                        field.economy.balance_cents,
                    );
                    let path = std::path::Path::new("data/lucy_journal.log");
                    let mut existing = self.fs.read_string(path).unwrap_or_default();
                    existing.push_str(&state_summary);
                    let _ = self.fs.write(path, existing.as_bytes());
                    tracing::debug!(point = action.point_id, "Motor: journal write");
                }

                // Type 8-11: Google Search (kennis ophalen)
                8..=11 => {
                    if let Some(result) = self.search_web(field, &action).await {
                        let inputs = Self::bytes_to_input(result.as_bytes());
                        let n = inputs.len().min(100);
                        self.feed_sensory_input(field, &inputs[..n]);
                        tracing::info!(
                            point = action.point_id,
                            chunks = n,
                            "Motor: web search → sensory input"
                        );
                    }
                }

                // Type 12-15: GitHub code lezen
                12..=15 => {
                    if let Some(result) = self.read_github(field).await {
                        let inputs = Self::bytes_to_input(result.as_bytes());
                        let n = inputs.len().min(100);
                        self.feed_sensory_input(field, &inputs[..n]);
                        tracing::info!(
                            point = action.point_id,
                            chunks = n,
                            "Motor: github read → sensory input"
                        );
                    }
                }

                _ => {}
            }

            // Motor-actie kost energie
            if let Some(p) = field.points.get_mut(action.point_id as usize) {
                p.energy -= 0.1;
            }
        }
    }

    /// Zoek op het web via Google Custom Search API
    async fn search_web(&self, field: &mut Field, action: &MotorAction) -> Option<String> {
        let api_key = self.config.get_key("GOOGLE_API_KEY")?;
        let cx = self.config.get_key("GOOGLE_SEARCH_ENGINE_ID")?;

        // Kosten: ~$0.005 per search ($5 per 1000 queries)
        let cost_cents = 1; // 1 cent per call (afronding omhoog)
        if !field.economy.can_afford(cost_cents) {
            tracing::trace!("Search blocked: no budget");
            return None;
        }

        // Genereer zoekterm vanuit phi-vector (Lucy's intentie)
        let query = self.phi_to_search_query(&action.phi);

        let url = format!(
            "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}&num=3",
            api_key,
            cx,
            urlencoding(&query)
        );

        match self.http.get(&url).await {
            Ok(resp) if resp.status == 200 => {
                field.economy.spend(cost_cents);
                let body = String::from_utf8_lossy(&resp.body).to_string();
                tracing::debug!(query, latency_ms = resp.latency_ms, "Search complete");
                Some(body)
            }
            Ok(resp) => {
                tracing::warn!(status = resp.status, "Search API error");
                None
            }
            Err(e) => {
                tracing::warn!(%e, "Search request failed");
                None
            }
        }
    }

    /// Lees trending/relevante code van GitHub
    async fn read_github(&self, field: &mut Field) -> Option<String> {
        let token = self.config.get_key("GITHUB_TOKEN")?;

        let cost_cents = 0; // GitHub API is gratis (met rate limits)
        if !field.economy.can_afford(cost_cents) {
            return None;
        }

        // Zoek Rust repositories met recente activiteit
        let url = "https://api.github.com/search/repositories?q=language:rust+sort:updated&per_page=5";

        match self.http.get(url).await {
            Ok(resp) if resp.status == 200 => {
                let _ = token; // Token zou in header moeten, maar GET werkt ook zonder voor publieke repos
                let body = String::from_utf8_lossy(&resp.body).to_string();
                tracing::debug!(latency_ms = resp.latency_ms, "GitHub read complete");
                Some(body)
            }
            _ => None,
        }
    }

    /// Vertaal een phi-vector naar een zoekterm
    /// De dominante dimensies bepalen het onderwerp
    fn phi_to_search_query(&self, phi: &[f32; DIM]) -> String {
        // Simpele mapping: phi → keywords
        // Lucy leert later betere mappings
        let topics = [
            "rust programming",
            "machine learning",
            "neural network",
            "self-modifying code",
            "artificial intelligence",
            "prediction error",
            "field theory",
            "consciousness",
            "emergence",
            "optimization",
            "distributed systems",
            "evolutionary algorithm",
            "free energy principle",
            "information theory",
            "tensor field",
            "adaptive systems",
        ];

        // Kies top-3 dimensies met hoogste absolute waarde
        let mut dims: Vec<(usize, f32)> = phi.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
        dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let query: Vec<&str> = dims.iter().take(2).map(|(i, _)| topics[*i % topics.len()]).collect();
        query.join(" ")
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct MotorAction {
    pub point_id: u32,
    pub action_type: usize,
    pub intensity: f32,
    pub phi: [f32; DIM],
}

fn dominant_dimension(phi: &[f32; DIM]) -> usize {
    let mut max_val = f32::MIN;
    let mut max_dim = 0;
    for d in 0..DIM {
        if phi[d].abs() > max_val {
            max_val = phi[d].abs();
            max_dim = d;
        }
    }
    max_dim
}

/// Simpele URL encoding
fn urlencoding(s: &str) -> String {
    s.replace(' ', "+")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('?', "%3F")
}
