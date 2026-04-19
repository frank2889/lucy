//! # Compile Sandbox — Isolated Build for Patch Testing
//!
//! Compiles patched source in `/tmp/lucy-v3-sandbox/` without touching production.

use std::fs;
use std::path::Path;
use std::process::Command;

pub struct SandboxCompiler {
    sandbox_dir: String,
}

impl SandboxCompiler {
    pub fn new() -> Self {
        Self {
            sandbox_dir: "/tmp/lucy-v3-sandbox".to_string(),
        }
    }

    /// Setup sandbox: copy crates/ → /tmp/lucy-v3-sandbox/
    pub fn setup(&self) -> Result<(), String> {
        // Clean old sandbox
        let _ = fs::remove_dir_all(&self.sandbox_dir);
        fs::create_dir_all(&self.sandbox_dir)
            .map_err(|e| format!("Failed to create sandbox: {}", e))?;

        // Copy Lucy crates
        let cmd = format!("cp -r /lucy/crates {}", &self.sandbox_dir);
        let output = Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .output()
            .map_err(|e| format!("Copy failed: {}", e))?;

        if !output.status.success() {
            return Err(format!("Copy crates failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    /// Apply patch (string replacement) to source file
    pub fn apply_patch(&self, file_path: &str, old_pattern: &str, new_pattern: &str) -> Result<(), String> {
        let sandbox_file = format!("{}/crates/{}", &self.sandbox_dir, file_path);

        // Read file
        let content = fs::read_to_string(&sandbox_file)
            .map_err(|e| format!("Failed to read {}: {}", sandbox_file, e))?;

        // Backup
        fs::write(format!("{}.bak", &sandbox_file), &content)
            .map_err(|e| format!("Backup failed: {}", e))?;

        // Validate patch size (max 500 chars)
        if new_pattern.len() > 500 {
            return Err("Patch too large (>500 chars)".to_string());
        }

        // Apply patch
        if !content.contains(old_pattern) {
            return Err(format!("Pattern not found in {}", sandbox_file));
        }

        let patched = content.replace(old_pattern, new_pattern);
        fs::write(&sandbox_file, &patched)
            .map_err(|e| format!("Patch write failed: {}", e))?;

        Ok(())
    }

    /// Compile in sandbox
    pub fn compile(&self) -> Result<CompileResult, String> {
        let cargo_dir = format!("{}/crates", &self.sandbox_dir);

        let output = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--workspace")
            .current_dir(&cargo_dir)
            .output()
            .map_err(|e| format!("Cargo failed: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            // Extract error lines
            let errors: Vec<&str> = stderr.lines()
                .filter(|l| l.contains("error"))
                .collect();
            return Ok(CompileResult {
                success: false,
                binary_path: None,
                errors: errors.iter().map(|s| s.to_string()).collect(),
                warnings: vec![],
            });
        }

        let binary_path = format!("{}/crates/target/release/lucy-core", &self.sandbox_dir);

        // Verify binary exists
        if !Path::new(&binary_path).exists() {
            return Err(format!("Binary not found at {}", binary_path));
        }

        // Extract warnings
        let warnings: Vec<String> = stdout.lines()
            .filter(|l| l.contains("warning"))
            .map(|s| s.to_string())
            .collect();

        Ok(CompileResult {
            success: true,
            binary_path: Some(binary_path),
            errors: vec![],
            warnings,
        })
    }

    /// Validate compiled binary
    pub fn validate_binary(&self, binary_path: &str) -> Result<(), String> {
        // Check ELF magic
        let content = fs::read(binary_path)
            .map_err(|e| format!("Can't read binary: {}", e))?;

        if content.len() < 4 || &content[0..4] != b"\x7FELF" {
            return Err("Invalid ELF header".to_string());
        }

        // Check size (lucy-core should be 10-100 MB)
        let size_mb = content.len() / (1024 * 1024);
        if size_mb < 10 || size_mb > 100 {
            return Err(format!("Binary size suspicious: {} MB", size_mb));
        }

        Ok(())
    }

    /// Cleanup sandbox
    pub fn cleanup(&self) {
        let _ = fs::remove_dir_all(&self.sandbox_dir);
    }
}

#[derive(Clone, Debug)]
pub struct CompileResult {
    pub success: bool,
    pub binary_path: Option<String>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}
