//! # journal — Lucy's Autonome Dagboek
//!
//! Lucy schrijft zelf observaties op basis van haar interne staat.
//! Ze beslist WAT ze opschrijft — niet een mens.
//!
//! Triggers:
//! - Emotionele pieken (hoge curiosity, satisfaction, of frustration)
//! - Milestones (eerste 1000 ticks, eerste pagina gelezen, etc.)
//! - Zelf-aanpassingen
//! - Onverwachte spanningsveranderingen
//!
//! Het dagboek wordt opgeslagen als JSON in data/journal.json
//! en automatisch getoond op het dashboard.

use lucy_state::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

// v2.5: Verhoogd van 200 → 2000. Lucy's gedachtenlog mag niet te snel verdwijnen.
// Later: configureerbaar uit data/config.json of zelfsturend.
const MAX_ENTRIES: usize = 2000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalEntry {
    pub tick: u64,
    pub kind: String,
    pub message: String,
    pub emotions: EmotionSnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionSnapshot {
    pub curiosity: f32,
    pub satisfaction: f32,
    pub frustration: f32,
    pub tension: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Journal {
    pub entries: Vec<JournalEntry>,
    /// Vorige emotie-staat voor veranderingsdetectie
    #[serde(skip)]
    prev_curiosity: f32,
    #[serde(skip)]
    prev_satisfaction: f32,
    #[serde(skip)]
    prev_frustration: f32,
    #[serde(skip)]
    prev_tension: f32,
    /// Milestones die al getriggerd zijn
    pub milestones_hit: Vec<String>,
    /// v2.5: Reflection count opgeslagen in journal (JSON, overleeft struct-wijzigingen)
    /// Dit is de bron van waarheid — Field.reflection_count wordt hieruit geladen.
    #[serde(default)]
    pub reflection_count: u64,
    /// Totaal geleerde pagina's (backup — overleeft snapshot reset)
    #[serde(default)]
    pub total_pages_read: u64,
    /// Totaal geleerde tokens (backup — overleeft snapshot reset)
    #[serde(default)]
    pub total_tokens_processed: u64,
}

impl Journal {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            prev_curiosity: 0.0,
            prev_satisfaction: 0.0,
            prev_frustration: 0.0,
            prev_tension: 0.0,
            milestones_hit: Vec::new(),
            reflection_count: 0,
            total_pages_read: 0,
            total_tokens_processed: 0,
        }
    }

    /// Lucy observeert haar eigen staat en beslist of ze iets wil opschrijven
    pub fn observe(&mut self, field: &Field, pages_read: u64, vocab_size: usize, tokens_processed: u64) {
        let n = field.active_set.len() as f32;
        if n == 0.0 { return; }

        let avg_curiosity: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].curiosity).sum::<f32>() / n;
        let avg_satisfaction: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].satisfaction).sum::<f32>() / n;
        let avg_frustration: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].frustration).sum::<f32>() / n;
        let avg_tension: f32 = field.active_set.iter()
            .map(|&id| field.points[id as usize].tension).sum::<f32>() / n;

        let emo = EmotionSnapshot {
            curiosity: avg_curiosity,
            satisfaction: avg_satisfaction,
            frustration: avg_frustration,
            tension: avg_tension,
        };

        // === EMOTIONELE PIEKEN ===

        // v2.0: Taalloze journal entries — raw data, geen menselijke interpretatie
        // Lucy's VS Code extensie kan deze interpreteren in welke taal dan ook

        if avg_curiosity > self.prev_curiosity * 3.0 && avg_curiosity > 0.01 {
            self.write(field.tick, "curiosity_spike",
                &format!("curiosity={:.4},prev={:.4}", avg_curiosity, self.prev_curiosity), emo.clone());
        }
        if avg_satisfaction > self.prev_satisfaction * 3.0 && avg_satisfaction > 0.01 {
            self.write(field.tick, "satisfaction_spike",
                &format!("satisfaction={:.4}", avg_satisfaction), emo.clone());
        }
        if avg_frustration > self.prev_frustration * 3.0 && avg_frustration > 0.05 {
            self.write(field.tick, "frustration_spike",
                &format!("frustration={:.4}", avg_frustration), emo.clone());
        }
        if self.prev_tension > 0.1 && avg_tension < self.prev_tension * 0.3 {
            self.write(field.tick, "breakthrough",
                &format!("tension:{:.4}->{:.4}", self.prev_tension, avg_tension), emo.clone());
        }
        if avg_tension > self.prev_tension * 3.0 && avg_tension > 0.1 {
            self.write(field.tick, "disturbance",
                &format!("tension={:.4}", avg_tension), emo.clone());
        }

        // Milestones — raw data
        self.check_milestone(field.tick, "ticks_1k", field.tick >= 1_000, "tick=1000", &emo);
        self.check_milestone(field.tick, "ticks_10k", field.tick >= 10_000, "tick=10000", &emo);
        self.check_milestone(field.tick, "ticks_100k", field.tick >= 100_000, "tick=100000", &emo);
        self.check_milestone(field.tick, "pages_1", pages_read >= 1, &format!("pages={}", pages_read), &emo);
        self.check_milestone(field.tick, "pages_10", pages_read >= 10, &format!("pages={}", pages_read), &emo);
        self.check_milestone(field.tick, "pages_100", pages_read >= 100, &format!("pages={}", pages_read), &emo);
        self.check_milestone(field.tick, "vocab_100", vocab_size >= 100, &format!("vocab={}", vocab_size), &emo);
        self.check_milestone(field.tick, "vocab_1000", vocab_size >= 1000, &format!("vocab={}", vocab_size), &emo);
        self.check_milestone(field.tick, "reflect_1", field.reflection_count >= 1, "reflections=1", &emo);
        self.check_milestone(field.tick, "reflect_100", field.reflection_count >= 100, &format!("reflections={}", field.reflection_count), &emo);
        self.check_milestone(field.tick, "tokens_10k", tokens_processed >= 10_000, &format!("tokens={}", tokens_processed), &emo);

        // Update vorige staat
        self.prev_curiosity = avg_curiosity;
        self.prev_satisfaction = avg_satisfaction;
        self.prev_frustration = avg_frustration;
        self.prev_tension = avg_tension;
    }

    fn check_milestone(&mut self, tick: u64, id: &str, condition: bool, message: &str, emo: &EmotionSnapshot) {
        if condition && !self.milestones_hit.contains(&id.to_string()) {
            self.milestones_hit.push(id.to_string());
            self.write(tick, "milestone", message, emo.clone());
        }
    }

    fn write(&mut self, tick: u64, kind: &str, message: &str, emotions: EmotionSnapshot) {
        let entry = JournalEntry {
            tick,
            kind: kind.to_string(),
            message: message.to_string(),
            emotions,
        };

        tracing::info!(tick, kind, message, "Lucy journal");
        self.entries.push(entry);

        // Hou het beheersbaar
        if self.entries.len() > MAX_ENTRIES {
            self.entries.drain(0..50);
        }
    }

    /// Atomic save: schrijf naar .tmp, dan rename
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let tmp = path.with_extension("tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, path)?;
        // CRC32 checksum + backup rotation
        let checksum = crc32fast::hash(json.as_bytes());
        let _ = std::fs::write(path.with_extension("crc32"), checksum.to_le_bytes());
        lucy_io::fs::rotate_backups(path, 5);
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn recent(&self, n: usize) -> &[JournalEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }
}
