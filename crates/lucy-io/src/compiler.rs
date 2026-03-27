use std::path::{Path, PathBuf};
use std::process::Command;

/// Lucy's compiler interface — ze kan Rust code schrijven, compileren, en hot-loaden
pub struct Compiler {
    /// Directory voor gegenereerde modules
    pub modules_dir: PathBuf,
}

impl Compiler {
    pub fn new(modules_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&modules_dir).ok();
        Self { modules_dir }
    }

    /// Schrijf een Rust source file
    pub fn write_source(&self, name: &str, code: &str) -> Result<PathBuf, String> {
        let path = self.modules_dir.join(format!("{}.rs", name));
        std::fs::write(&path, code).map_err(|e| format!("Write source: {}", e))?;
        tracing::info!(?path, "Module source written");
        Ok(path)
    }

    /// Compileer een Rust source file naar een dynamische library
    pub fn compile(&self, source: &Path) -> Result<PathBuf, String> {
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("Invalid source path")?;

        let lib_name = format!("lib{}", stem);
        let output = if cfg!(target_os = "macos") {
            self.modules_dir.join(format!("{}.dylib", lib_name))
        } else {
            self.modules_dir.join(format!("{}.so", lib_name))
        };

        let result = Command::new("rustc")
            .arg("--crate-type=cdylib")
            .arg("--edition=2024")
            .arg("-O")
            .arg("-o")
            .arg(&output)
            .arg(source)
            .output()
            .map_err(|e| format!("Compile failed: {}", e))?;

        if result.status.success() {
            tracing::info!(?output, "Module compiled");
            Ok(output)
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            tracing::warn!(%stderr, "Compilation error");
            Err(format!("Compilation error: {}", stderr))
        }
    }

    /// Hot-load een gecompileerde dynamische library
    pub unsafe fn load_module(
        &self,
        path: &Path,
    ) -> Result<libloading::Library, String> {
        unsafe {
            libloading::Library::new(path).map_err(|e| format!("Load module: {}", e))
        }
    }
}
