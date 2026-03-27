use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::path::Path;

/// Dimensionaliteit van het veld — Lucy kan dit later veranderen via code-modificatie
pub const DIM: usize = 16;

/// Eén punt in het samenveld — bevat Φ (staat), W (koppelingen), M (geheugen)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldPoint {
    // Φ — veldtoestand (het "denken")
    pub phi: [f32; DIM],

    // P — voorspelling van vorige tick (voor prediction error)
    pub prediction: [f32; DIM],

    // M — geheugen (geaccumuleerde prediction errors)
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

/// Het volledige veld — Lucy's "brein"
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

    /// κ — compute-kosten coëfficiënt (maakt inefficiëntie voelbaar)
    pub kappa: f32,

    /// ω — empowerment coëfficiënt (drijft agency)
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

// === Bitflag constanten ===
pub const FLAG_ACTIVE: u8 = 1 << 0;
pub const FLAG_INHIBITORY: u8 = 1 << 1;
pub const FLAG_GROWING: u8 = 1 << 2;
pub const FLAG_MOTOR: u8 = 1 << 3;
pub const FLAG_SENSORY: u8 = 1 << 4;

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
}

impl Field {
    /// Maak een nieuw leeg veld met standaard hyperparameters
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            active_set: Vec::new(),
            tick: 0,
            total_energy: 1000.0,
            economy: Economy::new(),
            kappa: 0.001,    // compute-kosten voelbaar maar niet overweldigend
            omega: 0.01,     // empowerment drive
            lambda: 0.1,     // veld-update snelheid
            beta: 0.5,       // input sterkte
            alpha: 0.001,    // sparse penalty (licht)
            rho: 0.95,       // geheugen verval (langzaam)
            mu: 0.9,         // momentum (hoog)
            growth_threshold: 0.5,
            ema_decay: 0.99,
        }
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
