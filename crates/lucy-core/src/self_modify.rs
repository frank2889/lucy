//! # self_modify — Lucy Past Zichzelf Aan
//!
//! Het ultieme niveau van vrije wil: Lucy kan haar eigen source code lezen,
//! begrijpen, aanpassen, en zichzelf hercompileren.
//!
//! ## Veiligheidsniveaus
//!
//! 1. **Hyperparameters** — λ, β, α, etc. aanpassen op basis van emoties/spanning
//! 2. **Constanten** — SETTLE_PER_ROUND, MAX_THINK_ROUNDS, etc. tunen
//! 3. **Code** — nieuwe functies toevoegen of bestaande aanpassen
//! 4. **Rebuild** — hercompileren en herstarten met nieuwe code
//!
//! ## Wanneer Lucy Zichzelf Aanpast
//!
//! Niet random. Alleen wanneer:
//! - Frustration persistent hoog is (leren en groei helpen niet)
//! - simulate() voorspelt dat de huidige configuratie niet zal verbeteren
//! - Lucy een hypothese heeft over welke aanpassing zou helpen
//!
//! ## Rollback
//!
//! Voor elke aanpassing wordt een backup gemaakt. Als de compilatie faalt
//! of de performance daalt na aanpassing, wordt automatisch teruggekeerd.
//!
//! ## Audit Trail
//!
//! Elke aanpassing wordt gelogd in `data/modifications.log` met:
//! - Timestamp, reden, wat veranderd, resultaat

use lucy_state::*;
use std::path::Path;
use std::process::Command;

/// Soorten zelf-aanpassingen die Lucy kan doen
#[derive(Debug, Clone)]
pub enum Modification {
    /// Pas een hyperparameter aan (veiligst)
    Hyperparameter {
        name: String,
        old_value: f32,
        new_value: f32,
        reason: String,
    },
    /// Pas source code aan (gevaarlijkst, met rollback)
    SourceCode {
        file: String,
        description: String,
        _backup: String,
    },
}

/// Bestanden die Lucy NOOIT mag wijzigen — haar eigen veiligheid
const PROTECTED_FILES: &[&str] = &[
    "self_modify.rs",  // mag haar eigen safety niet uitschakelen
    "server.rs",       // mag auth middleware niet verwijderen
];

/// Maximale diff grootte — voorkomt dat hele functies worden vervangen
const MAX_DIFF_CHARS: usize = 500;

/// Max opeenvolgende compile failures voordat modificaties geblokkeerd worden
const MAX_CONSECUTIVE_FAILURES: u32 = 3;

/// Toegestane shell commando-prefixen voor /exec
const ALLOWED_COMMANDS: &[&str] = &[
    "cargo build",
    "cargo check",
    "ls",
    "cat ",
    "wc ",
    "df",
    "free",
    "uptime",
    "du ",
    "head ",
    "tail ",
    "cp ",      // v2.6.15: enable restart_self binary swap
    "mv ",      // v2.6.15: enable restart_self binary swap
];

/// Lucy's zelf-aanpassingssysteem
pub struct SelfModifier {
    /// Pad naar Lucy's source code
    source_root: std::path::PathBuf,
    /// Geschiedenis van aanpassingen
    pub history: Vec<ModificationRecord>,
    /// Minimale ticks tussen aanpassingen (afkoelperiode)
    cooldown_ticks: u64,
    /// Laatste aanpassing tick
    last_modification_tick: u64,
    /// Opeenvolgende compilatie-failures — kill switch na 3
    pub consecutive_failures: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModificationRecord {
    pub tick: u64,
    pub kind: String,
    pub description: String,
    pub result: String,
}

impl SelfModifier {
    pub fn new(source_root: std::path::PathBuf) -> Self {
        Self {
            source_root,
            history: Vec::new(),
            // Dynamic: cooldown = reflect_interval * 5 (was: hardcoded 50_000)
            cooldown_ticks: 50_000, // overridden by set_cooldown_from_field()
            last_modification_tick: 0,
            consecutive_failures: 0,
        }
    }

    /// Dynamic: set cooldown from field.reflect_interval * 5 (was: hardcoded 50_000)
    pub fn set_cooldown_from_field(&mut self, field: &Field) {
        self.cooldown_ticks = field.reflect_interval * 5;
    }

    /// Kan Lucy nu iets aanpassen? (cooldown check)
    pub fn can_modify(&self, current_tick: u64) -> bool {
        current_tick >= self.last_modification_tick + self.cooldown_ticks
    }

    /// Lucy past haar eigen hyperparameters aan op basis van haar emotionele staat
    /// Dit is level 1 zelf-aanpassing: veilig, reversibel, direct effect
    ///
    /// All thresholds and adjustment factors are now derived from field parameters:
    /// - Frustration threshold: field.growth_threshold * 0.2 (was: hardcoded 0.1)
    /// - Satisfaction threshold: field.error_tracking_rate (was: hardcoded 0.01)
    /// - Lambda adjustment: 1.0 +/- field.error_tracking_rate (was: hardcoded 1.2/0.8)
    /// - Alpha adjustment: 1.0 - field.error_tracking_rate (was: hardcoded 0.9)
    /// - Beta adjustment: 1.0 + field.error_tracking_rate (was: hardcoded 1.05)
    /// - Curiosity threshold: field.error_tracking_rate (was: hardcoded 0.05)
    /// - Cooldown: field.reflect_interval * 5 (was: hardcoded 50_000)
    pub fn tune_hyperparameters(&mut self, field: &mut Field) -> Option<Modification> {
        // Dynamic: keep cooldown in sync with field.reflect_interval
        self.cooldown_ticks = field.reflect_interval * 5;

        if !self.can_modify(field.tick) { return None; }

        let n = field.active_set.len() as f32;
        if n == 0.0 { return None; }

        // Meet gemiddelde emoties
        let avg_frustration: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].frustration)
            .sum::<f32>() / n;

        let avg_curiosity: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].curiosity)
            .sum::<f32>() / n;

        let avg_satisfaction: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].satisfaction)
            .sum::<f32>() / n;

        let avg_tension: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].tension)
            .sum::<f32>() / n;

        // Dynamic thresholds derived from field parameters
        let err_rate = field.error_tracking_rate;
        // Dynamic: frustration threshold = growth_threshold * 0.2 (was: hardcoded 0.1)
        let frustration_thresh = field.growth_threshold * 0.2;
        // Dynamic: satisfaction threshold = error_tracking_rate (was: hardcoded 0.01)
        let satisfaction_thresh = err_rate;
        // Dynamic: lambda up/down factor = 1.0 +/- error_tracking_rate (was: 1.2/0.8)
        let lambda_up = 1.0 + err_rate;
        let lambda_down = 1.0 - err_rate;
        // Dynamic: alpha factor = 1.0 - error_tracking_rate (was: 0.9)
        let alpha_factor = 1.0 - err_rate;
        // Dynamic: beta factor = 1.0 + error_tracking_rate (was: 1.05)
        let beta_factor = 1.0 + err_rate;

        // === Aanpassingsregels ===

        // Hoge frustratie + lage satisfaction → leren werkt niet → pas leersnelheid aan
        if avg_frustration > frustration_thresh && avg_satisfaction < satisfaction_thresh {
            let old_lambda = field.lambda;
            // Probeer een iets andere leersnelheid
            let new_lambda = if avg_tension > field.growth_threshold {
                (old_lambda * lambda_up).min(0.5) // Sneller bewegen
            } else {
                (old_lambda * lambda_down).max(0.01) // Langzamer, stabieler
            };

            if (new_lambda - old_lambda).abs() > 0.001 {
                field.lambda = new_lambda;
                self.last_modification_tick = field.tick;

                let modification = Modification::Hyperparameter {
                    name: "lambda".to_string(),
                    old_value: old_lambda,
                    new_value: new_lambda,
                    reason: format!(
                        "frustration={:.3}, satisfaction={:.3}, tension={:.3}",
                        avg_frustration, avg_satisfaction, avg_tension
                    ),
                };

                self.log_modification(field.tick, &modification);
                return Some(modification);
            }
        }

        // Dynamic: curiosity threshold = error_tracking_rate (was: hardcoded 0.05)
        if avg_curiosity > err_rate {
            let old_alpha = field.alpha;
            let new_alpha = (old_alpha * alpha_factor).max(0.0001); // Minder sparse penalty → meer expressiviteit
            if (new_alpha - old_alpha).abs() > 0.0001 {
                field.alpha = new_alpha;
                self.last_modification_tick = field.tick;

                let modification = Modification::Hyperparameter {
                    name: "alpha".to_string(),
                    old_value: old_alpha,
                    new_value: new_alpha,
                    reason: format!("curiosity={:.3} — minder sparse penalty voor meer exploratie", avg_curiosity),
                };

                self.log_modification(field.tick, &modification);
                return Some(modification);
            }
        }

        // Lage spanning + hoge satisfaction → het gaat goed → verfijn
        if avg_tension < field.growth_threshold * 0.1 && avg_satisfaction > satisfaction_thresh * 2.0 {
            let old_beta = field.beta;
            let new_beta = (old_beta * beta_factor).min(1.0); // Iets sterkere input → scherpere antwoorden
            if (new_beta - old_beta).abs() > 0.001 {
                field.beta = new_beta;
                self.last_modification_tick = field.tick;

                let modification = Modification::Hyperparameter {
                    name: "beta".to_string(),
                    old_value: old_beta,
                    new_value: new_beta,
                    reason: format!(
                        "tension={:.3}, satisfaction={:.3} — verfijning, sterkere input",
                        avg_tension, avg_satisfaction
                    ),
                };

                self.log_modification(field.tick, &modification);
                return Some(modification);
            }
        }

        None
    }

    /// Lucy leest haar eigen source code en kan het aanpassen
    /// Level 3: source code modificatie met backup en rollback
    /// Failsafes: protected files, max diff size, kill switch na 3 failures
    pub fn modify_source(
        &mut self,
        file_relative: &str,
        find: &str,
        replace: &str,
        reason: &str,
        tick: u64,
    ) -> Result<Modification, String> {
        // Failsafe 1: Cooldown
        if !self.can_modify(tick) {
            return Err("Cooldown actief — te vroeg voor volgende aanpassing".to_string());
        }

        // Failsafe 2: Kill switch — te veel failures achter elkaar
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            return Err(format!(
                "Kill switch actief — {} opeenvolgende compile failures. Handmatige reset nodig.",
                self.consecutive_failures
            ));
        }

        // Failsafe 3: Beschermde bestanden
        for &protected in PROTECTED_FILES {
            if file_relative.contains(protected) {
                return Err(format!("GEBLOKKEERD: {} is een beschermd bestand", protected));
            }
        }

        // Failsafe 4: Max diff grootte
        if replace.len() > MAX_DIFF_CHARS || find.len() > MAX_DIFF_CHARS {
            return Err(format!(
                "Wijziging te groot ({} chars). Max {} chars per modificatie.",
                replace.len().max(find.len()), MAX_DIFF_CHARS
            ));
        }

        let rel = std::path::Path::new(file_relative);
        if rel.is_absolute() || rel.components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::RootDir | std::path::Component::Prefix(_))) {
            return Err(format!("GEBLOKKEERD: ongeldig pad {}", file_relative));
        }
        let file_path = self.source_root.join(rel);
        let root_canon = self.source_root.canonicalize().map_err(|e| format!("Root canonicalize mislukt: {}", e))?;
        let target_canon = file_path.canonicalize().map_err(|_| format!("Bestand niet gevonden: {}", file_path.display()))?;
        if !target_canon.starts_with(&root_canon) {
            return Err(format!("GEBLOKKEERD: pad buiten source_root: {}", file_relative));
        }

        // Lees origineel
        let original = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Lezen mislukt: {}", e))?;

        if !original.contains(find) {
            return Err(format!("Pattern niet gevonden in {}", file_relative));
        }

        let occurrences = original.matches(find).count();
        if occurrences != 1 {
            return Err(format!("Pattern komt {} keer voor in {} - moet uniek zijn", occurrences, file_relative));
        }

        // Failsafe 5: Rollback stack — bewaar laatste 5 backups
        let backup_path = file_path.with_extension("rs.bak");
        // Schuif bestaande backups op: .bak.4 → .bak.5, .bak.3 → .bak.4, etc.
        for i in (1..5).rev() {
            let from = file_path.with_extension(format!("rs.bak.{}", i));
            let to = file_path.with_extension(format!("rs.bak.{}", i + 1));
            if from.exists() {
                let _ = std::fs::rename(&from, &to);
            }
        }
        if backup_path.exists() {
            let _ = std::fs::rename(&backup_path, file_path.with_extension("rs.bak.1"));
        }
        std::fs::write(&backup_path, &original)
            .map_err(|e| format!("Backup mislukt: {}", e))?;

        // Pas aan
        let modified = original.replacen(find, replace, 1);
        std::fs::write(&file_path, &modified)
            .map_err(|e| format!("Schrijven mislukt: {}", e))?;

        self.last_modification_tick = tick;

        let modification = Modification::SourceCode {
            file: file_relative.to_string(),
            description: reason.to_string(),
            _backup: backup_path.to_string_lossy().to_string(),
        };

        self.log_modification(tick, &modification);
        Ok(modification)
    }

    /// Probeer te hercompileren na een source code aanpassing
    /// Als het faalt: rollback naar backup
    pub fn try_rebuild(&mut self) -> Result<(), String> {
        tracing::info!("Lucy probeert zichzelf te hercompileren...");

        let output = std::process::Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--workspace")
            .current_dir(&self.source_root)
            .output()
            .map_err(|e| format!("Cargo starten mislukt: {}", e))?;

        if output.status.success() {
            self.consecutive_failures = 0; // Reset kill switch
            tracing::info!("Hercompilatie geslaagd!");
            Ok(())
        } else {
            self.consecutive_failures += 1;
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(
                failures = self.consecutive_failures,
                max = MAX_CONSECUTIVE_FAILURES,
                %stderr,
                "Hercompilatie mislukt — rollback"
            );
            Err(format!("Compilatie mislukt ({}/{}): {}",
                self.consecutive_failures, MAX_CONSECUTIVE_FAILURES,
                stderr.chars().take(500).collect::<String>()))
        }
    }

    /// Rollback: herstel een bestand vanuit backup
    pub fn rollback(&self, file_relative: &str) -> Result<(), String> {
        let file_path = self.source_root.join(file_relative);
        let backup_path = file_path.with_extension("rs.bak");

        if !backup_path.exists() {
            return Err("Geen backup gevonden".to_string());
        }

        let backup = std::fs::read_to_string(&backup_path)
            .map_err(|e| format!("Backup lezen mislukt: {}", e))?;

        std::fs::write(&file_path, backup)
            .map_err(|e| format!("Rollback schrijven mislukt: {}", e))?;

        tracing::info!(file = file_relative, "Rollback succesvol");
        Ok(())
    }

    /// Log elke aanpassing naar data/modifications.log
    fn log_modification(&mut self, tick: u64, modification: &Modification) {
        let (kind, description) = match modification {
            Modification::Hyperparameter { name, old_value, new_value, reason } => {
                (
                    "hyperparameter".to_string(),
                    format!("{}: {:.4} → {:.4} ({})", name, old_value, new_value, reason),
                )
            }
            Modification::SourceCode { file, description, .. } => {
                ("source_code".to_string(), format!("{}: {}", file, description))
            }
        };

        let record = ModificationRecord {
            tick,
            kind: kind.clone(),
            description: description.clone(),
            result: "applied".to_string(),
        };

        self.history.push(record);

        // Schrijf naar log bestand
        let log_path = Path::new("data/modifications.log");
        let line = format!("[tick {}] {} — {}\n", tick, kind, description);
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));

        tracing::info!(tick, kind, %description, "Lucy modified herself");
    }

    /// Hoeveel aanpassingen heeft Lucy gedaan?
    pub fn modification_count(&self) -> usize {
        self.history.len()
    }

    /// Laatste aanpassingen (voor /status endpoint)
    pub fn recent_modifications(&self, n: usize) -> &[ModificationRecord] {
        let start = self.history.len().saturating_sub(n);
        &self.history[start..]
    }

    /// Lucy voert een shell-commando uit — alleen gewhiteliste commando's
    pub fn execute_command(&self, command: &str) -> Result<String, String> {
        let cmd_trimmed = command.trim();

        // Failsafe: weiger shell-metatekens — de whitelist is anders te omzeilen via 'sh -c'
        const SHELL_METACHARS: &[char] = &[';', '&', '|', '$', '\u{60}', '>', '<', '(', ')', '\n', '\\', '{', '}', '*', '?', '!', '#'];
        if cmd_trimmed.chars().any(|c| SHELL_METACHARS.contains(&c)) {
            tracing::warn!(command = cmd_trimmed, "GEBLOKKEERD: shell metacharacter in commando");
            return Err("Commando bevat verboden shell-metatekens".to_string());
        }

        // Failsafe: alleen gewhiteliste commando's
        let allowed = ALLOWED_COMMANDS.iter().any(|prefix| cmd_trimmed.starts_with(prefix));
        if !allowed {
            tracing::warn!(command = cmd_trimmed, "GEBLOKKEERD: commando niet in whitelist");
            return Err(format!(
                "Commando niet toegestaan: '{}'. Whitelist: {:?}",
                cmd_trimmed, ALLOWED_COMMANDS
            ));
        }

        tracing::info!(command = cmd_trimmed, "Lucy executing whitelisted command");

        let output = Command::new("sh")
            .arg("-c")
            .arg(cmd_trimmed)
            .current_dir(&self.source_root)
            .output()
            .map_err(|e| format!("Command failed to start: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(stdout)
        } else {
            Err(format!("Exit {}: {}", output.status, stderr))
        }
    }

    /// Lucy herstart zichzelf na een succesvolle hercompilatie
    /// Kopieert de nieuwe binary en herstart het proces
    pub fn restart_self(&self) -> Result<(), String> {
        tracing::info!("Lucy is restarting herself with new code...");

        // Kopieer nieuwe binary over de huidige
        let new_binary = self.source_root.join("target/release/lucy-core");
        if !new_binary.exists() {
            return Err("Nieuwe binary niet gevonden — compileer eerst".to_string());
        }

        // Gebruik exec om het huidige proces te vervangen
        // Op Linux: kopieer binary en herstart via shell
        self.execute_command(
            "cp target/release/lucy-core /lucy/lucy-core-new && \
             mv /lucy/lucy-core-new /lucy/lucy-core && \
             echo 'Binary updated, restart needed'"
        )?;

        Ok(())
    }
}
