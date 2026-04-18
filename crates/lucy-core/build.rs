//! Build-time version stamping — zodat je NOOIT meer hoeft te raden welke
//! binary er draait. Levert de env-vars LUCY_VERSION, LUCY_GIT_SHA,
//! LUCY_BUILD_TIME voor `env!(...)` in runtime-code.
//!
//! Triggeren van rebuilds:
//!   - Wanneer `.git/HEAD` of `.git/index` wijzigt (commit/checkout)
//!   - Bij wijziging van `CHANGELOG.md` (human-readable versie)
//!
//! Als git niet beschikbaar is (bv. Docker builds zonder .git): fallback naar
//! "unknown". De `CHANGELOG.md` top-lezen-heuristiek vangt dit op.

use std::process::Command;

fn main() {
    // Rebuild-triggers
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    println!("cargo:rerun-if-changed=../../docs/CHANGELOG.md");
    println!("cargo:rerun-if-env-changed=LUCY_VERSION_OVERRIDE");

    let sha = run_git(&["rev-parse", "--short=10", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = run_git(&["status", "--porcelain"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let sha_marked = if dirty { format!("{}+dirty", sha) } else { sha };

    // Lees de bovenste "## v..." regel uit CHANGELOG als human-readable versie
    let version = std::env::var("LUCY_VERSION_OVERRIDE")
        .ok()
        .or_else(read_changelog_top_version)
        .unwrap_or_else(|| "v0.0.0".to_string());

    let build_time = chrono_utc_now();

    println!("cargo:rustc-env=LUCY_VERSION={}", version);
    println!("cargo:rustc-env=LUCY_GIT_SHA={}", sha_marked);
    println!("cargo:rustc-env=LUCY_BUILD_TIME={}", build_time);
}

fn run_git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn read_changelog_top_version() -> Option<String> {
    // Use CARGO_MANIFEST_DIR to be robust across different build contexts (local, Docker, CI)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let manifest_path = std::path::PathBuf::from(manifest_dir);
    let workspace_root = manifest_path.parent()?.parent()?;  // go from /lucy/crates/lucy-core to /lucy

    let changelog_path = workspace_root.join("docs").join("CHANGELOG.md");
    if let Ok(content) = std::fs::read_to_string(&changelog_path) {
        for line in content.lines() {
            // eerste regel in vorm "## v2.6.11 ..." → pak "v2.6.11"
            if let Some(rest) = line.strip_prefix("## ") {
                if let Some(ver) = rest.split_whitespace().next() {
                    if ver.starts_with('v') {
                        return Some(ver.to_string());
                    }
                }
            }
        }
    }
    None
}

// Minimal UTC timestamp zonder extra dependencies — seconds since unix epoch als
// rfc3339-lite string. chrono is niet in workspace dependencies; we zelf rijmen.
fn chrono_utc_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days-since-epoch → yyyy-mm-dd via naive calculation (Gregorian)
    let days = (secs / 86_400) as i64;
    let (y, m, d) = days_to_ymd(days);
    let h = (secs / 3_600) % 24;
    let mi = (secs / 60) % 60;
    let se = secs % 60;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, h, mi, se)
}

fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm by Howard Hinnant (civil_from_days), works for proleptic Gregorian
    let z = days + 719_468;
    let era = if z >= 0 { z / 146_097 } else { (z - 146_096) / 146_097 };
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y_final = y + if m <= 2 { 1 } else { 0 };
    (y_final, m, d)
}
