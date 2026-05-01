//! # Deploy Coordinator — Atomic Binary Swap + Docker Restart
//!
//! Manages safe deployment: backups, swap, pre-deploy flush, restart.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;

pub struct DeployCoordinator {
    production_binary: String,
    max_backups: usize,
}

impl DeployCoordinator {
    pub fn new() -> Self {
        Self {
            production_binary: "/lucy/lucy-core".to_string(),
            max_backups: 5,
        }
    }

    /// Pre-deploy flush: wacht op verse snapshot vóór binary-swap.
    /// Poll snapshot-mtime; SIGTERM tijdens docker restart triggert sowieso een save (main.rs:1870),
    /// maar door eerst te wachten op een natuurlijke save-cycle voorkomen we race-windows.
    pub fn pre_deploy_flush(&self, _api_url: &str, _token: &str) -> Result<(), String> {
        fs::write("/tmp/lucy-pre-deploy-signal", "shutdown").ok();

        let snapshot_path = "/lucy/data/lucy.snapshot";
        let baseline_mtime = fs::metadata(snapshot_path)
            .and_then(|m| m.modified())
            .ok();

        // Poll up to 30s voor mtime-advance. Tick-loop schrijft snapshot op interval.
        for _ in 0..30 {
            thread::sleep(Duration::from_secs(1));
            if let (Some(base), Ok(meta)) = (baseline_mtime, fs::metadata(snapshot_path)) {
                if let Ok(current) = meta.modified() {
                    if current > base {
                        return Ok(());
                    }
                }
            }
        }

        // Geen verse save binnen 30s → ga toch door, SIGTERM-handler vangt het op.
        Ok(())
    }

    /// Create rolling backups: lucy-core → lucy-core.bak.1, shift .1→.2, etc.
    pub fn backup_binary(&self) -> Result<(), String> {
        if !Path::new(&self.production_binary).exists() {
            return Err(format!("{} does not exist", &self.production_binary));
        }

        // Shift backups: .1→.2, .2→.3, ... , .4→.5 (drop .5)
        for i in (1..self.max_backups).rev() {
            let from = format!("{}.bak.{}", &self.production_binary, i);
            let to = format!("{}.bak.{}", &self.production_binary, i + 1);

            if Path::new(&from).exists() {
                fs::rename(&from, &to).map_err(|e| format!("Backup rotate failed: {}", e))?;
            }
        }

        // Backup current: lucy-core → lucy-core.bak.1
        let bak1 = format!("{}.bak.1", &self.production_binary);
        fs::copy(&self.production_binary, &bak1)
            .map_err(|e| format!("Backup copy failed: {}", e))?;

        Ok(())
    }

    /// Swap binary atomically: write to `.new` op zelfde fs, chmod, rename → productie.
    /// fs::copy is NIET atomair; tijdens kopie kan een lezer een half-geschreven binary laden.
    /// rename(2) op POSIX is wél atomair binnen één filesystem.
    pub fn swap_binary(&self, new_binary_path: &str) -> Result<(), String> {
        if !Path::new(new_binary_path).exists() {
            return Err(format!("New binary not found: {}", new_binary_path));
        }

        let staging = format!("{}.new", &self.production_binary);

        fs::copy(new_binary_path, &staging)
            .map_err(|e| format!("Staging copy failed: {}", e))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&staging, fs::Permissions::from_mode(0o755))
                .map_err(|e| format!("chmod failed: {}", e))?;
        }

        fs::rename(&staging, &self.production_binary)
            .map_err(|e| format!("Atomic rename failed: {}", e))?;

        Ok(())
    }

    /// Docker restart: restart lucy container
    pub fn restart_docker(&self) -> Result<(), String> {
        let output = Command::new("docker")
            .args(&["restart", "lucy"])
            .output()
            .map_err(|e| format!("Docker restart failed: {}", e))?;

        if !output.status.success() {
            return Err(format!("Docker restart error: {}", String::from_utf8_lossy(&output.stderr)));
        }

        // Wait for container to be healthy (up to 30s)
        for _ in 0..6 {
            thread::sleep(Duration::from_secs(5));

            let status = Command::new("docker")
                .args(&["exec", "lucy", "curl", "-sf", "http://localhost:3000/status"])
                .output();

            if let Ok(out) = status {
                if out.status.success() {
                    return Ok(());  // Container is healthy
                }
            }
        }

        Err("Container failed to become healthy after 30s".to_string())
    }

    /// Rollback: restore from lucy-core.bak.1
    pub fn rollback(&self) -> Result<(), String> {
        let bak1 = format!("{}.bak.1", &self.production_binary);

        if !Path::new(&bak1).exists() {
            return Err(format!("No rollback backup available: {}", bak1));
        }

        fs::copy(&bak1, &self.production_binary)
            .map_err(|e| format!("Rollback copy failed: {}", e))?;

        // Restart
        self.restart_docker()?;

        Ok(())
    }

    /// Full deploy pipeline
    pub fn deploy(&self, new_binary_path: &str, api_url: &str, api_token: &str) -> Result<(), String> {
        // 1. Pre-deploy flush
        self.pre_deploy_flush(api_url, api_token)?;

        // 2. Backup current binary
        self.backup_binary()?;

        // 3. Swap binary
        self.swap_binary(new_binary_path)?;

        // 4. Restart docker
        self.restart_docker()?;

        Ok(())
    }
}

#[cfg(unix)]
mod unix_perms {
    use std::os::unix::fs::PermissionsExt;
    use std::fs;

    pub fn make_executable(path: &str) -> std::io::Result<()> {
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))
    }
}
