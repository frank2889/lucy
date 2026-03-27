use std::path::{Path, PathBuf};

/// Lucy's filesystem interface — lezen/schrijven van eigen code, data, geheugen
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
