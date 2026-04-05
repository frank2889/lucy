//! # fs — Lucy's Filesystem Interface
//!
//! Alle interactie met het lokale bestandssysteem gaat via dit module.
//! Dit geeft Lucy de mogelijkheid om:
//!
//! - Haar eigen source code te lezen (bootstrap zelf-leren)
//! - Data bestanden te lezen/schrijven (snapshots, vocabulaire)
//! - Directory structuren te verkennen
//!
//! ## Zelf-Lezing
//!
//! `read_own_source()` verzamelt recursief alle `.rs` en `.toml` bestanden
//! in Lucy's workspace. Dit is het eerste wat Lucy leest bij first boot:
//! haar eigen code. Ze leert taalpatronen uit Rust syntax EN absorbeert
//! woorden als interesses voor toekomstige web-exploratie.
//!
//! Directories `target/`, `.git/`, en `modules/` worden overgeslagen
//! (build artifacts, git history, externe modules zijn geen zinvolle leerstof).

use std::path::{Path, PathBuf};

/// Lucy's filesystem interface — lezen/schrijven van eigen code, data, geheugen.
pub struct FileSystem {
    /// Root directory van Lucy's workspace
    pub root: PathBuf,
}

impl FileSystem {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Lees een bestand als bytes
    pub fn read(&self, path: &Path) -> Result<Vec<u8>, String> {
        std::fs::read(path).map_err(|e| format!("Read {}: {}", path.display(), e))
    }

    /// Lees een bestand als string
    pub fn read_string(&self, path: &Path) -> Result<String, String> {
        std::fs::read_to_string(path).map_err(|e| format!("Read {}: {}", path.display(), e))
    }

    /// Schrijf bytes naar een bestand
    pub fn write(&self, path: &Path, data: &[u8]) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Mkdir {}: {}", parent.display(), e))?;
        }
        std::fs::write(path, data).map_err(|e| format!("Write {}: {}", path.display(), e))
    }

    /// Lijst bestanden in een directory
    pub fn list_dir(&self, path: &Path) -> Result<Vec<PathBuf>, String> {
        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("List {}: {}", path.display(), e))?;
        let mut paths = Vec::new();
        for entry in entries {
            if let Ok(e) = entry {
                paths.push(e.path());
            }
        }
        Ok(paths)
    }

    /// Lees Lucy's eigen source code — haar eerste zelfkennis
    pub fn read_own_source(&self) -> Result<Vec<(PathBuf, String)>, String> {
        let mut sources = Vec::new();
        self.collect_rust_files(&self.root, &mut sources)?;
        Ok(sources)
    }

    fn collect_rust_files(
        &self,
        dir: &Path,
        out: &mut Vec<(PathBuf, String)>,
    ) -> Result<(), String> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("Read dir {}: {}", dir.display(), e))?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            // Skip target/ en .git/
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "target" || name == ".git" || name == "modules" {
                    continue;
                }
            }

            if path.is_dir() {
                self.collect_rust_files(&path, out)?;
            } else if path.extension().is_some_and(|ext| ext == "rs" || ext == "toml") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    out.push((path, content));
                }
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DATA ONSTERFELIJKHEID — shared functies voor alle save/load operaties
// ═══════════════════════════════════════════════════════════════════════

/// Roteer backups: `data/file.bin` → `data/backups/file.bin.1` t/m `.{keep}`
/// Meest recente = .1, oudste wordt verwijderd.
/// Lucy's geleerde data gaat NOOIT verloren.
pub fn rotate_backups(path: &Path, keep: usize) {
    let backup_dir = path.parent().unwrap_or(Path::new(".")).join("backups");
    let _ = std::fs::create_dir_all(&backup_dir);

    let filename = path.file_name().unwrap_or_default().to_string_lossy();

    // Shift: .4→.5, .3→.4, .2→.3, .1→.2
    for i in (1..keep).rev() {
        let from = backup_dir.join(format!("{}.{}", filename, i));
        let to = backup_dir.join(format!("{}.{}", filename, i + 1));
        let _ = std::fs::rename(&from, &to);
    }

    // Verwijder de oudste die eraf valt
    let oldest = backup_dir.join(format!("{}.{}", filename, keep + 1));
    let _ = std::fs::remove_file(&oldest);

    // Kopieer huidige naar .1
    if path.exists() {
        let newest = backup_dir.join(format!("{}.1", filename));
        let _ = std::fs::copy(path, &newest);
    }
}

/// Schrijf CRC32 checksum sidecar naast een databestand
pub fn write_checksum(path: &Path, data: &[u8]) {
    let checksum = crc32fast::hash(data);
    let checksum_path = path.with_extension("crc32");
    let _ = std::fs::write(&checksum_path, checksum.to_le_bytes());
}

/// Valideer CRC32 checksum. Returns true als ok of als geen checksum bestaat (backward compat).
pub fn validate_checksum(path: &Path) -> bool {
    let checksum_path = path.with_extension("crc32");
    if !checksum_path.exists() {
        return true; // geen checksum = oud formaat, accepteer
    }
    let Ok(data) = std::fs::read(path) else { return false };
    let Ok(stored) = std::fs::read(&checksum_path) else { return false };
    if stored.len() != 4 { return false; }
    let stored_crc = u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]);
    let actual_crc = crc32fast::hash(&data);
    actual_crc == stored_crc
}

/// Probeer een bestand te laden van primary, dan backups 1-5.
/// Retourneert (bytes, bron) of None.
pub fn load_with_fallback(path: &Path) -> Option<(Vec<u8>, String)> {
    // 1. Primary
    if path.exists() {
        if validate_checksum(path) {
            if let Ok(data) = std::fs::read(path) {
                return Some((data, "primary".to_string()));
            }
        }
    }

    // 2. Backups
    let backup_dir = path.parent().unwrap_or(Path::new(".")).join("backups");
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    for i in 1..=5 {
        let backup = backup_dir.join(format!("{}.{}", filename, i));
        if backup.exists() {
            if let Ok(data) = std::fs::read(&backup) {
                // Herstel primary van deze backup
                let _ = std::fs::copy(&backup, path);
                return Some((data, format!("backup#{}", i)));
            }
        }
    }

    None
}

/// Atomic save met checksum + backup rotatie.
/// Dit is de ENIGE manier om data te schrijven. Gebruikt door alle subsystemen.
pub fn atomic_save(path: &Path, data: &[u8], keep_backups: usize) -> Result<(), String> {
    // Valideer: weiger lege data te schrijven (bescherming tegen bugs)
    if data.len() < 16 {
        return Err(format!("Weiger save: data te klein ({} bytes) — mogelijk corrupt", data.len()));
    }

    // Atomic write: tmp → rename
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data).map_err(|e| format!("Write tmp: {}", e))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("Rename: {}", e))?;

    // Checksum sidecar
    write_checksum(path, data);

    // Roteer backups
    rotate_backups(path, keep_backups);

    Ok(())
}
