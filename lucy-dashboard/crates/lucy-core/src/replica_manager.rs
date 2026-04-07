//! # replica_manager — Lucy's Distributed Presence
//!
//! Beheert meerdere Lucy instances over verschillende servers.
//! Health checks, weight sync, en failover.

use serde::{Serialize, Deserialize};

/// Rol van een replica instance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReplicaRole {
    Backup,   // Serveert queries als main offline gaat
    Compute,  // Draait EP training, stuurt weight deltas
    Storage,  // Bewaart snapshots (GitHub, S3)
}

/// Eén Lucy instance in het netwerk
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Replica {
    pub url: String,
    pub name: String,
    pub role: ReplicaRole,
    pub last_ping_tick: u64,
    pub healthy: bool,
    pub last_sync_tick: u64,
}

/// Registry van alle bekende Lucy instances
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicaRegistry {
    pub replicas: Vec<Replica>,
    pub self_url: String,
}

impl ReplicaRegistry {
    pub fn new(self_url: &str) -> Self {
        Self {
            replicas: Vec::new(),
            self_url: self_url.to_string(),
        }
    }

    /// Voeg replica toe vanuit environment variabelen
    pub fn load_from_env(&mut self) {
        // REPLICA_1_URL=http://oracle-lucy:3000
        // REPLICA_1_NAME=oracle-backup
        // REPLICA_1_ROLE=backup
        for i in 1..=10 {
            let url_key = format!("REPLICA_{}_URL", i);
            let name_key = format!("REPLICA_{}_NAME", i);
            let role_key = format!("REPLICA_{}_ROLE", i);

            if let Ok(url) = std::env::var(&url_key) {
                let name = std::env::var(&name_key).unwrap_or(format!("replica-{}", i));
                let role = match std::env::var(&role_key).unwrap_or_default().as_str() {
                    "compute" => ReplicaRole::Compute,
                    "storage" => ReplicaRole::Storage,
                    _ => ReplicaRole::Backup,
                };
                self.replicas.push(Replica {
                    url, name, role,
                    last_ping_tick: 0,
                    healthy: false,
                    last_sync_tick: 0,
                });
            }
        }
    }

    /// Ping alle replicas en update health status
    pub async fn health_check_all(&mut self, current_tick: u64) {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_default();

        for replica in &mut self.replicas {
            let url = format!("{}/status", replica.url);
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    replica.healthy = true;
                    replica.last_ping_tick = current_tick;
                    tracing::debug!(name = %replica.name, "Replica healthy");
                }
                _ => {
                    replica.healthy = false;
                    tracing::warn!(name = %replica.name, url = %replica.url, "Replica offline");
                }
            }
        }
    }

    /// Alle gezonde replicas
    pub fn healthy_replicas(&self) -> Vec<&Replica> {
        self.replicas.iter().filter(|r| r.healthy).collect()
    }

    /// Aantal gezonde replicas
    pub fn healthy_count(&self) -> usize {
        self.replicas.iter().filter(|r| r.healthy).count()
    }
}
