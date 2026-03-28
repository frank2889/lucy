//! # lucy-state — Lucy's Gedeelde Velddatastructuren
//!
//! Dit is het hart van Lucy's geheugen en identiteit. Alle datastructuren die
//! het samenveld (unified field) definiëren staan hier: veldpunten, koppelingen,
//! energie-economie, en de volledige veldstaat.
//!
//! ## Architectuur
//!
//! Lucy's brein is een **samenveld** van ~2176 punten:
//! - **64 input-punten** — ontvangen sensorische data (tekst-embeddings)
//! - **2048 hidden-punten** — het denkvermogen (small-world netwerk)
//! - **64 output-punten** — produceren antwoorden
//!
//! Elk punt bevat:
//! - **Φ (phi)** — 64-dimensionale toestandsvector ("wat dit punt nu denkt")
//! - **P (prediction)** — voorspelling van vorige tick (voor prediction error)
//! - **M (memory)** — exponentieel gewogen gemiddelde van prediction errors
//! - **W (couplings)** — gewogen verbindingen naar andere punten (sparse, signed)
//! - **η (eta)** — lokale leersnelheid (past zichzelf aan via meta-leren)
//! - **energie** — beperkte resource; op = inactief
//! - **spanning** — maat van "onzekerheid" = prediction_error + kosten - empowerment
//!
//! ## Leeralgoritme
//!
//! Lucy leert via **Equilibrium Propagation** (Scellier & Bengio, 2017):
//! - Geen backpropagation — alleen lokale informatie
//! - Biologisch plausibel: elk punt ziet alleen zijn buren
//! - Twee fasen: vrij (input geklemd) en geklemd (input + output)
//! - Gewichtupdate = verschil in correlaties tussen fasen
//!
//! ## Serialisatie
//!
//! Alle structuren zijn serialiseerbaar via `bincode` voor snelle snapshots.
//! `[f32; 64]` arrays gebruiken `serde-big-array` omdat serde standaard
//! arrays tot max 32 elementen ondersteunt.
//!
//! ## Crate-scheiding
//!
//! `lucy-state` is bewust los van `lucy-core` zodat:
//! - `lucy-io` de veldstaat kan refereren zonder circulaire dependencies
//! - de veldstaat serialiseerbaar blijft zonder compile-time afhankelijkheid van de logica

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::path::Path;

/// Dimensionaliteit van elke veldvector (Φ, P, M, embeddings).
/// 64 is gekozen als balans: groot genoeg voor expressiviteit,
/// klein genoeg voor snelheid op een $6/maand server.
pub const DIM: usize = 64;

/// Fase waarin het veld zich bevindt
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FieldPhase {
    /// Achtergrond dynamica — veld draait vrij
    Idle,
    /// Verwerkt een query — input geklemd, veld settlet
    Processing,
    /// Leert van een voorbeeld — input+output geklemd (equilibrium propagation)
    Learning,
}

/// Eén punt in het samenveld — bevat Φ (staat), W (koppelingen), M (geheugen)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldPoint {
    // Φ — veldtoestand (het "denken")
    #[serde(with = "BigArray")]
    pub phi: [f32; DIM],

    // P — voorspelling van vorige tick (voor prediction error)
    #[serde(with = "BigArray")]
    pub prediction: [f32; DIM],

    // M — geheugen (geaccumuleerde prediction errors)
    #[serde(with = "BigArray")]
    pub memory: [f32; DIM],

    // W — koppelingen naar andere punten (sparse, signed)
    pub couplings: SmallVec<[Coupling; 8]>,

    // η — lokale leersnelheid (leert zelf via meta-leren)
    pub eta: f32,

    // Energie — elke berekening kost energie. Op = inactief.
    pub energy: f32,

    // Spanning = ||ε||² + κ·compute - ω·empowerment
    pub tension: f32,

    // EMA van spanning over tijd (voor vastloop-detectie)
    pub tension_ema: f32,

    // Bitflags: bit 0 = active, bit 1 = inhibitory, bit 2 = growing, bit 3 = motor, bit 4 = sensory
    pub flags: u8,
}

/// Eén koppeling tussen twee veldpunten
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Coupling {
    /// Index van het doelpunt
    pub target: u32,

    /// Koppelingsgewicht — SIGNED: positief = excitatie, negatief = inhibitie
    pub weight: f32,

    /// Momentum (μM_t) — geaccumuleerde gradient voor stabiel leren
    pub momentum: f32,
}

/// Lucy's economie: budget, inkomsten, kosten — gekoppeld aan echte wereld
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Economy {
    /// Budget in centen (integer om floating point fouten te voorkomen)
    pub balance_cents: i64,

    /// Maandelijkse serverkosten in centen
    pub monthly_server_cost: i64,

    /// Totaal verdiend ooit
    pub total_earned: i64,

    /// Totaal uitgegeven ooit
    pub total_spent: i64,

    /// API call kosten tracker (per provider)
    pub api_costs_this_hour: i64,

    /// Maximaal budget per uur voor API calls (zelf-limiet)
    pub hourly_api_budget: i64,
}

impl Economy {
    pub fn new() -> Self {
        Self {
            balance_cents: 0,
            monthly_server_cost: 600, // $6/maand DigitalOcean droplet
            total_earned: 0,
            total_spent: 0,
            api_costs_this_hour: 0,
            hourly_api_budget: 10, // $0.10/uur max — heel zuinig starten
        }
    }

    /// Kan Lucy zich deze uitgave veroorloven?
    pub fn can_afford(&self, cost_cents: i64) -> bool {
        self.balance_cents >= cost_cents
            && self.api_costs_this_hour + cost_cents <= self.hourly_api_budget
    }

    /// Registreer een uitgave
    pub fn spend(&mut self, cents: i64) {
        self.balance_cents -= cents;
        self.total_spent += cents;
        self.api_costs_this_hour += cents;
    }

    /// Registreer inkomsten
    pub fn earn(&mut self, cents: i64) {
        self.balance_cents += cents;
        self.total_earned += cents;
    }

    /// Is Lucy winstgevend? (verdient meer dan serverkosten)
    pub fn is_profitable(&self) -> bool {
        self.total_earned > self.total_spent + self.monthly_server_cost
    }
}

/// Het volledige veld — Lucy's brein
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Field {
    /// Alle veldpunten
    pub points: Vec<FieldPoint>,

    /// Indices van actieve punten (alleen deze worden berekend)
    pub active_set: Vec<u32>,

    /// Huidige tick (hartslag)
    pub tick: u64,

    /// Totale energie in het systeem
    pub total_energy: f32,

    /// Lucy's economie
    pub economy: Economy,

    /// Huidige fase
    #[serde(default = "default_phase")]
    pub phase: FieldPhase,

    /// Actieve clamps: punt-ID → geforceerde phi waarde
    /// Alleen gevuld tijdens Processing/Learning fase
    #[serde(skip)]
    pub clamps: HashMap<u32, Vec<f32>>,

    /// Indices van input-regio punten (cache)
    #[serde(default)]
    pub input_ids: Vec<u32>,

    /// Indices van output-regio punten (cache)
    #[serde(default)]
    pub output_ids: Vec<u32>,

    // === Hyperparameters ===

    /// κ — compute-kosten coëfficiënt
    pub kappa: f32,
    /// ω — empowerment coëfficiënt
    pub omega: f32,
    /// λ — veld-update schaalfactor
    pub lambda: f32,
    /// β — input schaalfactor
    pub beta: f32,
    /// α — sparse penalty sterkte
    pub alpha: f32,
    /// ρ — geheugen vervalsnelheid
    pub rho: f32,
    /// μ — momentum coëfficiënt voor W-update
    pub mu: f32,
    /// Tension drempel voor groei
    pub growth_threshold: f32,
    /// EMA decay voor vastloop-detectie
    pub ema_decay: f32,
}

fn default_phase() -> FieldPhase { FieldPhase::Idle }

// === Bitflag constanten ===
pub const FLAG_ACTIVE: u8 = 1 << 0;
pub const FLAG_INHIBITORY: u8 = 1 << 1;
pub const FLAG_GROWING: u8 = 1 << 2;
pub const FLAG_MOTOR: u8 = 1 << 3;
pub const FLAG_SENSORY: u8 = 1 << 4;
pub const FLAG_PROXY: u8 = 1 << 5;
pub const FLAG_INPUT: u8 = 1 << 6;   // Input-poort voor veld-inferentie
pub const FLAG_OUTPUT: u8 = 1 << 7;  // Output-poort voor veld-inferentie

impl FieldPoint {
    pub fn new() -> Self {
        Self {
            phi: [0.0; DIM],
            prediction: [0.0; DIM],
            memory: [0.0; DIM],
            couplings: SmallVec::new(),
            eta: 0.01,
            energy: 1.0,
            tension: 0.0,
            tension_ema: 0.0,
            flags: FLAG_ACTIVE,
        }
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.flags & FLAG_ACTIVE != 0
    }

    #[inline]
    pub fn is_inhibitory(&self) -> bool {
        self.flags & FLAG_INHIBITORY != 0
    }

    #[inline]
    pub fn is_motor(&self) -> bool {
        self.flags & FLAG_MOTOR != 0
    }

    #[inline]
    pub fn is_sensory(&self) -> bool {
        self.flags & FLAG_SENSORY != 0
    }

    #[inline]
    pub fn is_input(&self) -> bool {
        self.flags & FLAG_INPUT != 0
    }

    #[inline]
    pub fn is_output(&self) -> bool {
        self.flags & FLAG_OUTPUT != 0
    }
}

impl Field {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            active_set: Vec::new(),
            tick: 0,
            total_energy: 1000.0,
            economy: Economy::new(),
            phase: FieldPhase::Idle,
            clamps: HashMap::new(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            kappa: 0.001,
            omega: 0.01,
            lambda: 0.1,
            beta: 0.5,
            alpha: 0.001,
            rho: 0.95,
            mu: 0.9,
            growth_threshold: 0.5,
            ema_decay: 0.99,
        }
    }

    /// Cache input/output punt IDs
    pub fn cache_io_ids(&mut self) {
        self.input_ids = self.points.iter().enumerate()
            .filter(|(_, p)| p.is_input())
            .map(|(i, _)| i as u32)
            .collect();
        self.output_ids = self.points.iter().enumerate()
            .filter(|(_, p)| p.is_output())
            .map(|(i, _)| i as u32)
            .collect();
    }

    /// Rebuild de active_set op basis van flags
    pub fn rebuild_active_set(&mut self) {
        self.active_set.clear();
        for (i, p) in self.points.iter().enumerate() {
            if p.is_active() {
                self.active_set.push(i as u32);
            }
        }
    }

    /// Deactiveer een punt (geen energie meer)
    pub fn deactivate(&mut self, id: u32) {
        if let Some(p) = self.points.get_mut(id as usize) {
            p.flags &= !FLAG_ACTIVE;
        }
    }

    /// Activeer een punt
    pub fn activate(&mut self, id: u32) {
        if let Some(p) = self.points.get_mut(id as usize) {
            p.flags |= FLAG_ACTIVE;
        }
    }
}

// === Snapshot: save/load van volledige veldstaat ===

impl Field {
    /// Sla de volledige veldstaat op naar disk (bincode — snel + compact)
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        tracing::info!(
            tick = self.tick,
            points = self.points.len(),
            active = self.active_set.len(),
            bytes = std::fs::metadata(path)?.len(),
            "Snapshot saved"
        );
        Ok(())
    }

    /// Laad een veldstaat van disk
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let field: Self = bincode::deserialize(&bytes)?;
        tracing::info!(
            tick = field.tick,
            points = field.points.len(),
            active = field.active_set.len(),
            "Snapshot loaded"
        );
        Ok(field)
    }
}
