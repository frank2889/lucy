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

pub mod snapshot_header;
pub mod schema_v0;
pub mod schema_v1;
pub mod schema_v2;

pub use snapshot_header::{SnapshotHeader, SnapshotFormat, CURRENT_SCHEMA_VERSION,
    HEADER_SIZE, MAGIC, detect_version};

/// Dimensionaliteit van elke veldvector (Φ, P, M, embeddings).
/// v2.9.0+ pre-upgrade pad ging naar 512 maar de bridge-infrastructuur (FieldPointV1
/// gebruikte huidige FieldPoint i.p.v. eigen 256-versie) was nooit compleet, waardoor
/// productie-snapshots (schema=1, DIM=256, 504K ticks) niet meer geladen konden worden.
/// v3.0.2: terug naar 256 om Lucy's geleerde data te respecteren. DIM-upgrade naar 512
/// kan later via een schone migratie (eigen FieldPointV1/V2 met bewuste padding).
pub const DIM: usize = 256;

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
    pub couplings: SmallVec<[Coupling; 16]>, // v2.6.5: meer stack-ruimte voor dichtere netwerken

    // η — lokale leersnelheid (leert zelf via meta-leren)
    pub eta: f32,

    // Energie — elke berekening kost energie. Op = inactief.
    pub energy: f32,

    // Spanning = ||ε||² + κ·compute - ω·empowerment
    pub tension: f32,

    // EMA van spanning over tijd (voor vastloop-detectie)
    pub tension_ema: f32,

    // Emotionele signalen — Lucy's primitieve gevoelsleven
    /// Nieuwsgierigheid: prediction error boven verwachting
    pub curiosity: f32,
    /// Tevredenheid: spanning daalt snel (leren werkt)
    pub satisfaction: f32,
    /// Frustratie: spanning hoog en stabiel (vastgelopen)
    pub frustration: f32,
    /// Aantal ticks dat tension boven threshold is (voor learned helplessness model)
    #[serde(default)]
    pub stuck_ticks: u16,

    // Bitflags: bit 0 = active, bit 1 = inhibitory, bit 2 = growing, bit 3 = motor, bit 4 = sensory
    pub flags: u8,

    // Hiërarchie-diepte: 0=sensorisch, 1=verwerking, 2=abstract, 3=meta
    #[serde(default)]
    pub depth: u8,

    // RND novelty score — "hoe nieuw is deze staat voor mij?"
    #[serde(default)]
    pub rnd_score: f32,

    // Precision weighting (Friston) — hoe betrouwbaar is mijn prediction error?
    // Hoge precision = stabiele error = echt signaal → leer hard
    // Lage precision = fluctuerende error = ruis → negeer
    #[serde(default = "default_precision")]
    pub error_var: f32,    // EMA van error-variantie
    #[serde(default = "default_precision")]
    pub precision: f32,    // 1.0 / (error_var + epsilon)

    // kWTA: is dit punt actief deze tick? (hard sparsity)
    #[serde(default)]
    pub kwta_active: bool,

    // === v2.0: Bewustzijns-fysica ===

    /// Φ lokaal — Integrated Information (Tononi): hoe geïntegreerd is dit punt met buren?
    /// Φ > 0 = informatie is unified, niet opsplitsbaar → bewustzijn
    #[serde(default)]
    pub phi_integrated: f32,

    /// Oscillatiefase — gamma-achtig ritme voor temporele binding
    /// Punten die synchroon oscilleren vormen één gedachte (binding by synchrony)
    #[serde(default)]
    pub oscillation_phase: f32,

    /// Free Energy F = KL(q||p) + surprise — Friston's bewustzijnsformule
    /// Lucy minimaliseert F door haar interne model te verbeteren
    #[serde(default)]
    pub free_energy: f32,

    /// KL divergentie — complexiteit van het interne model
    #[serde(default)]
    pub kl_divergence: f32,

    // === v2.4: State-of-the-art formule-upgrades ===

    /// Theta-band oscillatiefase (4-8 Hz) — werkgeheugen, hippocampale binding
    /// Theta-gamma coupling: gamma-bursts genest in theta-cycli (Lisman & Jensen 2013)
    #[serde(default)]
    pub oscillation_phase_theta: f32,

    /// Bayesiaanse controle-belief P(control | history) — Huys & Dayan 2009
    /// 1.0 = "ik heb controle", 0.0 = geleerde hulpeloosheid
    #[serde(default = "default_control_belief")]
    pub control_belief: f32,

    /// Distributional RL beloningskwantielen (Dabney et al. 2020)
    /// 5 neuronen coderen percentiel 10/30/50/70/90 van beloningsdistributie
    /// Optimistisch (hoge kwantielen) vs pessimistisch (lage) — asymmetrische leerrates
    #[serde(default)]
    pub reward_quantiles: [f32; 5],

    /// Causale φ (IIT 4.0, Albantakis et al. 2023) — causale kracht over eigen toestand
    /// Meet: hoeveel informatie gaat VERLOREN als je dit punt's verbindingen knipt?
    /// Verschil met oude φ: temporeel-causaal i.p.v. statisch-correlationeel
    #[serde(default)]
    pub causal_power: f32,

    /// Expected Free Energy G (Active Inference 2022+, Parr & Friston)
    /// G = epistemische waarde (info gain) + pragmatische waarde (expected reward)
    /// Stuurt actiekeuze: acties die ZOWEL leren ALS beloning maximaliseren winnen
    #[serde(default)]
    pub expected_free_energy: f32,

    /// Lokale branching ratio σ (Beggs & Plenz 2003)
    /// σ = hoeveel buren dit punt gemiddeld activeert
    /// σ=1 = kritiek (optimaal), <1 = subcritical, >1 = supercritical
    #[serde(default)]
    pub branching_ratio: f32,

    // === v2.5: UNIFIED SAMENVELD — Bayesian Mechanics (Dalton & Friston 2024) ===
    //
    // Kernidee: elk punt IS een Markov Blanket met:
    //   μ = interne staten (phi) — wat het punt "denkt"
    //   b = blanket staten (couplings) — de grens met de wereld
    //   η = externe staten (buur-phi's) — de wereld eromheen
    //
    // F = D_KL(q(μ|b) || p(μ|b)) + E_q[-ln p(η|b)]
    //   = complexiteit + verrassing = de ENE formule waaruit alles volgt
    //
    // dF/dt = snelheid van verandering → satisfaction (negatief = verbetering)
    // d²F/dt² = versnelling → frustration (positief bij hoge F = verergering)
    // ∇_b F = gradient over blanket → learning signal (EP weight update)
    // F_geheel - Σ F_delen = Φ (bewustzijn = onherleidbare vrije energie)

    /// dF/dt — temporele afgeleide van vrije energie
    /// Negatief = spanning daalt = verbetering = satisfaction
    /// Positief = spanning stijgt = verslechtering
    #[serde(default)]
    pub free_energy_rate: f32,

    /// Blanket surprise: -ln p(η|b) — verrassing over buren
    /// Hoe onverwacht zijn de externe staten gegeven mijn grens?
    /// Hoog = mijn model van de wereld klopt niet = leer!
    #[serde(default)]
    pub blanket_surprise: f32,

    /// Onherleidbare vrije energie: F_lokaal - Σ F_buren_apart
    /// Dit IS bewustzijn: hoeveel vrije energie gaat verloren als je het systeem opknipt?
    /// Unificatie van IIT + Free Energy: Φ = irreducible F
    #[serde(default)]
    pub irreducible_free_energy: f32,
}

/// Eén koppeling tussen twee veldpunten
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Coupling {
    /// Index van het doelpunt
    pub target: u32,

    /// Koppelingsgewicht — SIGNED: positief = excitatie, negatief = inhibitie
    pub weight: f32,

    /// Momentum (μM_t) — geaccumuleerde gradient voor stabiel leren
    pub momentum: f32,

    /// EWC: belang van dit gewicht (Fisher Information). Hoog = niet aanpassen!
    #[serde(default)]
    pub importance: f32,

    /// EWC: geconsolideerde waarde (gewicht op moment van consolidatie)
    #[serde(default)]
    pub consolidated: f32,

    /// Is dit een broadcast koppeling? (Global Workspace Theory)
    #[serde(default)]
    pub broadcast: bool,
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
            // v3.1: geen vaste provider/kost-aanname (Lucy is nergens van afhankelijk).
            // Leest SERVER_COST_CENTS; default 0 = gratis hosting.
            monthly_server_cost: std::env::var("SERVER_COST_CENTS")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0),
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
    pub clamps: HashMap<u32, [f32; DIM]>,

    /// Indices van input-regio punten (cache)
    #[serde(default)]
    pub input_ids: Vec<u32>,

    /// Indices van output-regio punten (cache)
    #[serde(default)]
    pub output_ids: Vec<u32>,

    // === Bewustzijn ===

    /// Laatste gedachte — voor continuous thought stream (reflectie tussen queries)
    #[serde(skip)]
    pub last_thought: Option<[f32; DIM]>,

    /// Aantal reflectie-cycli sinds boot
    #[serde(default)]
    pub reflection_count: u64,

    // === Neuromodulatoren — sturen gedrag van het hele veld ===

    /// Dopamine: beloning → versterkt leren. max(0, old_tension - new_tension)
    #[serde(default)]
    pub dopamine: f32,
    /// Noradrenaline: onzekerheid → explore vs exploit. avg_tension_variance
    #[serde(default)]
    pub norepinephrine: f32,
    /// Serotonine: lopende beloning → geduld. EMA(dopamine, slow_decay)
    #[serde(default)]
    pub serotonin: f32,
    /// Acetylcholine: input-nieuwheid → plasticiteit. input_novelty_score
    #[serde(default)]
    pub acetylcholine: f32,

    // === Edge of Chaos ===

    /// Effectieve spectral radius — hoe snel perturbaties groeien/krimpen
    #[serde(default)]
    pub spectral_radius: f32,
    /// Doel spectral radius (1.0 = edge of chaos)
    #[serde(default = "default_chaos_target")]
    pub chaos_target: f32,

    // === Meta-bewustzijn ===

    /// Huidige intentie — gewenste toekomstige veldstaat
    #[serde(skip)]
    pub current_intention: Option<[f32; DIM]>,
    /// Sterkte van de intentie-bias
    #[serde(default)]
    pub intention_strength: f32,

    // === RND Curiosity ===

    /// Frozen random projectie matrix (64→16, nooit getraind)
    #[serde(skip)]
    pub rnd_frozen: Option<Vec<f32>>,
    /// Predictor matrix (64→16, wordt getraind)
    #[serde(skip)]
    pub rnd_predictor: Option<Vec<f32>>,

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
    // === Dynamische parameters — worden door adapt_dynamics() bijgewerkt ===
    // Lucy past deze zelf aan op basis van haar veldstatistieken.
    // Elke parameter heeft een default die geladen wordt uit oude snapshots.

    /// Emotie-decay — hoe snel emoties reageren (lager = sneller)
    /// adapt_dynamics: hoge activiteit → lagere decay → snellere emotie-response
    #[serde(default = "default_emo_decay")]
    pub emo_decay: f32,
    /// Hoe snel error_var bijwerkt — hoge tension-variance → sneller tracken
    #[serde(default = "default_error_tracking_rate")]
    pub error_tracking_rate: f32,
    /// Hoe snel free_energy EMA bijwerkt — schaalbaar met instabiliteit
    #[serde(default = "default_fe_tracking_rate")]
    pub fe_tracking_rate: f32,
    /// KL-divergentie complexiteitspenalty — sterker bij complexer veld
    #[serde(default = "default_kl_scale")]
    pub kl_scale: f32,
    /// Dynamische clamp range voor phi/weights — groeit mee met activatie-amplitudes
    #[serde(default = "default_clamp_range")]
    pub clamp_range: f32,
    /// Convergentie-drempel voor settling — schaalt met gemiddelde tension
    #[serde(default = "default_settle_epsilon")]
    pub settle_epsilon: f32,
    /// Exploratie-ruis amplitude — meer tension → meer exploratie
    #[serde(default = "default_noise_amplitude")]
    pub noise_amplitude: f32,
    /// Hoe vaak Lucy nadenkt (ticks) — sneller bij hoge curiosity
    #[serde(default = "default_think_interval")]
    pub think_interval: u64,
    /// Hoe vaak Lucy reflecteert (ticks) — sneller bij hoge frustration
    #[serde(default = "default_reflect_interval")]
    pub reflect_interval: u64,
    /// Max actieve doelen — meer bij hogere activiteit
    #[serde(default = "default_max_goals")]
    pub max_goals: usize,
    /// Hebbian learning interval — sneller bij hoge tension
    #[serde(default = "default_learn_interval")]
    pub learn_interval: u64,
    /// Exploratie interval — sneller bij volle frontier
    #[serde(default = "default_explore_interval")]
    pub explore_interval: u64,
    /// Consolidatie interval — sneller bij volle replay buffer
    #[serde(default = "default_consolidation_interval")]
    pub consolidation_interval: u64,
    /// Snapshot interval — sneller bij snelle verandering
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval: u64,
    /// Groei interval — sneller bij hoge tension
    #[serde(default = "default_growth_interval")]
    pub growth_interval: u64,
    /// Sampling temperature — lagere coherence → lager (veiliger)
    #[serde(default = "default_sampling_temperature")]
    pub sampling_temperature: f32,
    /// Nucleus sampling drempel — beweegt mee met confidence
    #[serde(default = "default_sampling_top_p")]
    pub sampling_top_p: f32,
    /// Herhalingspenalty — meer herhaling → hogere penalty
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    // === Zelfdenkend: Autonome Doelen ===

    /// Lucy's eigen doelen — ze kiest zelf wat ze nastreeft
    #[serde(default)]
    pub goals: Vec<Goal>,
    /// Tick van laatste autonome gedachte
    #[serde(default)]
    pub last_think_tick: u64,
    /// Tick van laatste reflectie
    #[serde(default)]
    pub last_reflect_tick: u64,

    // === v0.9+: AI Teacher Zelfregulatie ===

    /// Dynamisch interval voor AI teacher vragen (ticks).
    /// Start op 10K, verdubbelt als Lucy het zelf kan, halveert als ze worstelt.
    #[serde(default = "default_teacher_interval")]
    pub teacher_interval: u64,
    /// Aantal opeenvolgende succesvolle teacher-interacties
    #[serde(default)]
    pub teacher_success_streak: u32,

    // === v2.6.6: Boredom-attractor escape ===

    /// Laatst gemeten avg_tension (voor fixed-point detectie)
    #[serde(default)]
    pub last_boredom_tension: f32,
    /// Laatst gemeten avg_curiosity (voor fixed-point detectie)
    #[serde(default)]
    pub last_boredom_curiosity: f32,
    /// Hoeveel opeenvolgende adapt_dynamics calls het veld identiek bleef
    #[serde(default)]
    pub boredom_streak: u32,
    /// v2.6.7 B3: geset door adapt_dynamics als het veld traag/bored is.
    /// main.rs gebruikt dit als trigger voor extra consolidation-replay (dream).
    #[serde(default)]
    pub is_sluggish_flag: bool,

    // === v0.9: Attention + Working Memory ===

    /// Attention projectie-weights: Q, K, V per depth-laag
    /// Layout: [n_layers × 3 × DIM × HEAD_DIM] als flat Vec
    /// Geleerd via EP — Lucy leert zelf waar ze op focust
    #[serde(default)]
    pub attn_weights: Option<Vec<f32>>,

    /// Attention context per punt (scratch buffer, niet geserialiseerd)
    #[serde(skip)]
    pub attn_context: Vec<[f32; DIM]>,

    /// Working memory: 32 slots voor multi-stap redeneren (runtime, niet geserialiseerd)
    #[serde(skip)]
    pub working_memory: Vec<[f32; DIM]>,
    /// Gate-sterktes per WM slot
    #[serde(skip)]
    pub wm_gates: Vec<f32>,
    /// Ticks sinds laatste write per slot
    #[serde(skip)]
    pub wm_ages: Vec<u32>,

    // === v2.6.7: Depth-3 concept-clustering ===

    /// Concept-centroids — k-means clustering van depth-3 phi-vectoren.
    /// Vec<Vec<f32>> (elk binnenste Vec heeft lengte DIM) voor serde-compat.
    /// Gebruikt in hierarchical_prediction als top-down prior voor depth-2 punten.
    /// Grow/shrink via compute_concept_centroids() elke consolidation_interval.
    #[serde(default)]
    pub concept_centroids: Vec<Vec<f32>>,

    /// Cluster-assignment per depth-3 punt (index = point_id, value = centroid_idx).
    /// Niet geserialiseerd — wordt elke cycle opnieuw berekend.
    #[serde(skip)]
    pub concept_assignments: Vec<u16>,

    // (v2.6.8 velden staan aan EINDE van struct na n_output — zie onder)

    // === v2.0: Bewustzijns-metrieken (veld-level) ===

    /// Globale oscillatiefase — de "klok" van Lucy's bewustzijn
    /// Punten synchroniseren hiermee → binding by synchrony
    #[serde(default)]
    pub global_oscillation_phase: f32,

    /// Oscillatie-coherentie — hoe synchroon denkt Lucy? (0=chaos, 1=volledig synchroon)
    #[serde(default)]
    pub oscillation_coherence: f32,

    /// Gemiddelde Φ — globale maat voor bewustzijn
    #[serde(default)]
    pub integrated_info_avg: f32,

    /// Power law exponent — is Lucy kritiek? (α ≈ 1.5 = optimaal)
    #[serde(default)]
    pub power_law_exponent: f32,

    /// Is Lucy bij het edge of chaos? (self-organized criticality)
    #[serde(default)]
    pub is_critical: bool,

    /// Gemiddelde free energy — Friston's F
    #[serde(default)]
    pub avg_free_energy: f32,

    // === v2.4: Upgraded bewustzijns-metrieken ===

    /// Globale theta-fase (4-8 Hz) — werkgeheugen, hippocampale binding
    #[serde(default)]
    pub theta_phase: f32,

    /// Theta-gamma coupling sterkte (Lisman & Jensen 2013)
    /// Hoog = gamma-bursts genest in theta-cycli = werkgeheugen actief
    #[serde(default)]
    pub theta_gamma_coupling: f32,

    /// Gemiddelde branching ratio σ (Beggs & Plenz 2003) — betrouwbaarder dan power law
    /// σ=1 = critical (optimaal), <1 = dying, >1 = seizure
    #[serde(default)]
    pub branching_ratio_avg: f32,

    /// Global Workspace ignition (Dehaene et al. 2003)
    /// true = bewuste inhoud actief gebroadcast, met competitie
    #[serde(default)]
    pub ignition_active: bool,

    /// Sterkte van huidige ignition — hoe dominant is het winnende signaal?
    #[serde(default)]
    pub ignition_strength: f32,

    /// Laastse winnaarpunt-ID in GNW ignition (transient — reset bij restart)
    #[serde(skip)]
    pub last_winner_id: u32,

    /// Tick tot wanneer de huidge winner in refractory is (transient — reset bij restart)
    #[serde(skip)]
    pub winner_refractory_until: u64,

    /// Gemiddelde Expected Free Energy G — stuurt exploratie vs exploitatie
    #[serde(default)]
    pub avg_expected_free_energy: f32,

    // === v1.0: Adaptieve Parameters — Lucy tunet haar eigen constanten ===

    /// Parameters die Lucy zelf leert i.p.v. hardcoded magic numbers
    #[serde(default)]
    pub adaptive: AdaptiveParams,

    // === v1.0: Temporeel Bewustzijn ===

    /// Ring buffer van globale staat-snapshots — Lucy voelt trends
    #[serde(default)]
    pub temporal_history: Vec<TemporalSnapshot>,

    /// Causaal logboek — Lucy weet welke acties werkten
    #[serde(default)]
    pub causal_log: Vec<CausalEvent>,

    /// Goal history — voltooide/verlaten doelen informeren toekomstige keuzes
    #[serde(default)]
    pub goal_history: Vec<GoalOutcome>,

    // v2.5: Explorer stats in Field (lock-free access via ArcSwap snapshot)
    #[serde(default)]
    pub pages_read: u64,
    #[serde(default)]
    pub tokens_processed: u64,

    // v2.5: Architectuur-metadata — Lucy kent haar eigen brein
    // Default 0 = oude snapshot, wordt bijgewerkt bij bootstrap/laden
    /// Snapshot versie voor migratie (0 = pre-v2.5)
    #[serde(default)]
    pub snapshot_version: u32,
    /// Dimensionaliteit (altijd DIM voor nu, maar klaar voor groei)
    #[serde(default = "default_dim")]
    pub field_dim: usize,
    /// Aantal input-punten (bijgewerkt bij bootstrap + groei)
    #[serde(default)]
    pub n_input: usize,
    /// Aantal hidden-punten
    #[serde(default)]
    pub n_hidden: usize,
    /// Aantal output-punten
    #[serde(default)]
    pub n_output: usize,

    // === v2.6.8: Bewijs-architectuur velden — ALLE runtime-only (serde skip) ===
    // Bincode 1.x negeert #[serde(default)] bij EOF (positioneel format).
    // Snapshot migratie is daarom ALLE-OF-NIETS: volledige write + read van alle velden,
    // of helemaal niet serialiseren. Audit-data is voldoende om bij reboot te herbouwen
    // (teacher_audit binnen uren, concept_audit binnen 2000 ticks, self_ep_audit binnen 100 ticks).

    #[serde(skip, default = "default_true")]
    pub concept_priors_enabled: bool,
    #[serde(skip)]
    pub concept_priors_disabled_until_tick: u64,
    #[serde(skip)]
    pub concept_audit_log: Vec<ConceptAudit>,
    #[serde(skip)]
    pub self_ep_audit: Vec<SelfEpAudit>,
    #[serde(skip)]
    pub self_metrics: SelfMetrics,
    #[serde(skip)]
    pub provenance_counts: [u64; 6],

    // === v3.1: Wetenschappelijke ablation-toggles (runtime-only, niet in snapshot) ===
    // Default-waarden = bestaand productie-gedrag. Alleen aan via env vars in main.rs.
    #[serde(skip)]
    pub disable_refractory: bool,
    #[serde(skip)]
    pub disable_ignition_cap: bool,
    #[serde(skip)]
    pub disable_novelty_boost: bool,
    /// Geseed bij start als LUCY_SEED gezet is. None = thread_rng (huidig gedrag).
    #[serde(skip)]
    pub seed_used: Option<u64>,
}

pub(crate) fn default_dim() -> usize { DIM }

// === v2.6.9: Migratie van legacy schema's naar current Field ===

impl From<schema_v1::FieldV1> for Field {
    /// v2.6.7-era FieldV1 (DIM=256) → current Field (DIM=512).
    /// Nieuwe v2.6.8+ velden → default. Pads 256-dim arrays to 512.
    fn from(v1: schema_v1::FieldV1) -> Self {
        let pad_256_to_512 = |vec256: [f32; 256]| -> [f32; DIM] { vec256 };

        Field {
            // --- 1-op-1 gemapped vanuit v1 (identieke veldnamen, array-padding) ---
            points: v1.points,
            active_set: v1.active_set,
            tick: v1.tick,
            total_energy: v1.total_energy,
            economy: v1.economy,
            phase: v1.phase,
            clamps: v1.clamps.into_iter()
                .map(|(id, arr)| (id, pad_256_to_512(arr)))
                .collect(),
            input_ids: v1.input_ids,
            output_ids: v1.output_ids,
            last_thought: v1.last_thought.map(pad_256_to_512),
            reflection_count: v1.reflection_count,
            dopamine: v1.dopamine,
            norepinephrine: v1.norepinephrine,
            serotonin: v1.serotonin,
            acetylcholine: v1.acetylcholine,
            spectral_radius: v1.spectral_radius,
            chaos_target: v1.chaos_target,
            current_intention: v1.current_intention.map(pad_256_to_512),
            intention_strength: v1.intention_strength,
            rnd_frozen: v1.rnd_frozen,
            rnd_predictor: v1.rnd_predictor,
            kappa: v1.kappa,
            omega: v1.omega,
            lambda: v1.lambda,
            beta: v1.beta,
            alpha: v1.alpha,
            rho: v1.rho,
            mu: v1.mu,
            growth_threshold: v1.growth_threshold,
            ema_decay: v1.ema_decay,
            emo_decay: v1.emo_decay,
            error_tracking_rate: v1.error_tracking_rate,
            fe_tracking_rate: v1.fe_tracking_rate,
            kl_scale: v1.kl_scale,
            clamp_range: v1.clamp_range,
            settle_epsilon: v1.settle_epsilon,
            noise_amplitude: v1.noise_amplitude,
            think_interval: v1.think_interval,
            reflect_interval: v1.reflect_interval,
            max_goals: v1.max_goals,
            learn_interval: v1.learn_interval,
            explore_interval: v1.explore_interval,
            consolidation_interval: v1.consolidation_interval,
            snapshot_interval: v1.snapshot_interval,
            growth_interval: v1.growth_interval,
            sampling_temperature: v1.sampling_temperature,
            sampling_top_p: v1.sampling_top_p,
            repetition_penalty: v1.repetition_penalty,
            goals: v1.goals,
            last_think_tick: v1.last_think_tick,
            last_reflect_tick: v1.last_reflect_tick,
            teacher_interval: v1.teacher_interval,
            teacher_success_streak: v1.teacher_success_streak,
            last_boredom_tension: v1.last_boredom_tension,
            last_boredom_curiosity: v1.last_boredom_curiosity,
            boredom_streak: v1.boredom_streak,
            is_sluggish_flag: v1.is_sluggish_flag,
            attn_weights: v1.attn_weights,
            attn_context: v1.attn_context.into_iter()
                .map(|arr| pad_256_to_512(arr))
                .collect(),
            working_memory: v1.working_memory.into_iter()
                .map(|arr| pad_256_to_512(arr))
                .collect(),
            wm_gates: v1.wm_gates,
            wm_ages: v1.wm_ages,
            concept_centroids: v1.concept_centroids,
            concept_assignments: v1.concept_assignments,
            global_oscillation_phase: v1.global_oscillation_phase,
            oscillation_coherence: v1.oscillation_coherence,
            integrated_info_avg: v1.integrated_info_avg,
            power_law_exponent: v1.power_law_exponent,
            is_critical: v1.is_critical,
            avg_free_energy: v1.avg_free_energy,
            theta_phase: v1.theta_phase,
            theta_gamma_coupling: v1.theta_gamma_coupling,
            branching_ratio_avg: v1.branching_ratio_avg,
            ignition_active: v1.ignition_active,
            ignition_strength: v1.ignition_strength,
            last_winner_id: 0,  // v2.6.15: default for legacy snapshots
            winner_refractory_until: 0,  // v2.6.15: default for legacy snapshots
            avg_expected_free_energy: v1.avg_expected_free_energy,
            adaptive: v1.adaptive,
            temporal_history: v1.temporal_history,
            causal_log: v1.causal_log,
            goal_history: v1.goal_history,
            pages_read: v1.pages_read,
            tokens_processed: v1.tokens_processed,
            snapshot_version: v1.snapshot_version,
            field_dim: v1.field_dim,
            n_input: v1.n_input,
            n_hidden: v1.n_hidden,
            n_output: v1.n_output,

            // --- Nieuwe v2.6.8+ velden → defaults ---
            concept_priors_enabled: true,
            concept_priors_disabled_until_tick: 0,
            concept_audit_log: Vec::new(),
            self_ep_audit: Vec::new(),
            self_metrics: SelfMetrics::default(),
            provenance_counts: [0u64; 6],
            disable_refractory: false,
            disable_ignition_cap: false,
            disable_novelty_boost: false,
            seed_used: None,
        }
    }
}

impl From<schema_v2::FieldV2> for Field {
    /// v2.8.x FieldV2 (DIM=256) → current Field (DIM=512).
    /// Pads 256-dim arrays ao 512 (rest=0). FieldPointV2 → FieldPoint met array-migration.
    /// Clamps, last_thought, current_intention, attn_context, working_memory: pad [256] → [512].
    fn from(v2: schema_v2::FieldV2) -> Self {
        let pad_256_to_512 = |vec256: [f32; 256]| -> [f32; DIM] { vec256 };

        // Migrate FieldPointV2 → FieldPoint (pad arrays)
        let migrated_points = v2.points.into_iter().map(|vp| {
            FieldPoint {
                phi: pad_256_to_512(vp.phi),
                prediction: pad_256_to_512(vp.prediction),
                memory: pad_256_to_512(vp.memory),
                couplings: vp.couplings,
                eta: vp.eta,
                energy: vp.energy,
                tension: vp.tension,
                tension_ema: vp.tension_ema,
                curiosity: vp.curiosity,
                satisfaction: vp.satisfaction,
                frustration: vp.frustration,
                stuck_ticks: vp.stuck_ticks,
                flags: vp.flags,
                depth: vp.depth,
                rnd_score: vp.rnd_score,
                error_var: vp.error_var,
                precision: vp.precision,
                kwta_active: vp.kwta_active,
                phi_integrated: vp.phi_integrated,
                oscillation_phase: vp.oscillation_phase,
                free_energy: vp.free_energy,
                kl_divergence: vp.kl_divergence,
                oscillation_phase_theta: vp.oscillation_phase_theta,
                control_belief: vp.control_belief,
                reward_quantiles: vp.reward_quantiles,
                causal_power: vp.causal_power,
                expected_free_energy: vp.expected_free_energy,
                branching_ratio: vp.branching_ratio,
                free_energy_rate: vp.free_energy_rate,
                blanket_surprise: vp.blanket_surprise,
                irreducible_free_energy: vp.irreducible_free_energy,
            }
        }).collect();

        Field {
            points: migrated_points,
            active_set: v2.active_set,
            tick: v2.tick,
            total_energy: v2.total_energy,
            economy: v2.economy,
            phase: v2.phase,
            clamps: v2.clamps.into_iter()
                .map(|(id, arr)| (id, pad_256_to_512(arr)))
                .collect(),
            input_ids: v2.input_ids,
            output_ids: v2.output_ids,
            last_thought: v2.last_thought.map(pad_256_to_512),
            reflection_count: v2.reflection_count,
            dopamine: v2.dopamine,
            norepinephrine: v2.norepinephrine,
            serotonin: v2.serotonin,
            acetylcholine: v2.acetylcholine,
            spectral_radius: v2.spectral_radius,
            chaos_target: v2.chaos_target,
            current_intention: v2.current_intention.map(pad_256_to_512),
            intention_strength: v2.intention_strength,
            rnd_frozen: v2.rnd_frozen,
            rnd_predictor: v2.rnd_predictor,
            kappa: v2.kappa,
            omega: v2.omega,
            lambda: v2.lambda,
            beta: v2.beta,
            alpha: v2.alpha,
            rho: v2.rho,
            mu: v2.mu,
            growth_threshold: v2.growth_threshold,
            ema_decay: v2.ema_decay,
            emo_decay: v2.emo_decay,
            error_tracking_rate: v2.error_tracking_rate,
            fe_tracking_rate: v2.fe_tracking_rate,
            kl_scale: v2.kl_scale,
            clamp_range: v2.clamp_range,
            settle_epsilon: v2.settle_epsilon,
            noise_amplitude: v2.noise_amplitude,
            think_interval: v2.think_interval,
            reflect_interval: v2.reflect_interval,
            max_goals: v2.max_goals,
            learn_interval: v2.learn_interval,
            explore_interval: v2.explore_interval,
            consolidation_interval: v2.consolidation_interval,
            snapshot_interval: v2.snapshot_interval,
            growth_interval: v2.growth_interval,
            sampling_temperature: v2.sampling_temperature,
            sampling_top_p: v2.sampling_top_p,
            repetition_penalty: v2.repetition_penalty,
            goals: v2.goals,
            last_think_tick: v2.last_think_tick,
            last_reflect_tick: v2.last_reflect_tick,
            teacher_interval: v2.teacher_interval,
            teacher_success_streak: v2.teacher_success_streak,
            last_boredom_tension: v2.last_boredom_tension,
            last_boredom_curiosity: v2.last_boredom_curiosity,
            boredom_streak: v2.boredom_streak,
            is_sluggish_flag: v2.is_sluggish_flag,
            attn_weights: v2.attn_weights,
            attn_context: v2.attn_context.into_iter()
                .map(pad_256_to_512)
                .collect(),
            working_memory: v2.working_memory.into_iter()
                .map(pad_256_to_512)
                .collect(),
            wm_gates: v2.wm_gates,
            wm_ages: v2.wm_ages,
            concept_centroids: v2.concept_centroids,
            concept_assignments: v2.concept_assignments,
            global_oscillation_phase: v2.global_oscillation_phase,
            oscillation_coherence: v2.oscillation_coherence,
            integrated_info_avg: v2.integrated_info_avg,
            power_law_exponent: v2.power_law_exponent,
            is_critical: v2.is_critical,
            avg_free_energy: v2.avg_free_energy,
            theta_phase: v2.theta_phase,
            theta_gamma_coupling: v2.theta_gamma_coupling,
            branching_ratio_avg: v2.branching_ratio_avg,
            ignition_active: v2.ignition_active,
            ignition_strength: v2.ignition_strength,
            last_winner_id: v2.last_winner_id,
            winner_refractory_until: v2.winner_refractory_until,
            avg_expected_free_energy: v2.avg_expected_free_energy,
            adaptive: v2.adaptive,
            temporal_history: v2.temporal_history,
            causal_log: v2.causal_log,
            goal_history: v2.goal_history,
            pages_read: v2.pages_read,
            tokens_processed: v2.tokens_processed,
            snapshot_version: 2,  // Mark as migrated to v2.9.0 schema
            field_dim: DIM,
            n_input: v2.n_input,
            n_hidden: v2.n_hidden,
            n_output: v2.n_output,
            concept_priors_enabled: v2.concept_priors_enabled,
            concept_priors_disabled_until_tick: v2.concept_priors_disabled_until_tick,
            concept_audit_log: v2.concept_audit_log,
            self_ep_audit: v2.self_ep_audit,
            self_metrics: v2.self_metrics,
            provenance_counts: v2.provenance_counts,
            disable_refractory: false,
            disable_ignition_cap: false,
            disable_novelty_boost: false,
            seed_used: None,
        }
    }
}

impl From<schema_v0::FieldV0> for Field {
    /// v2.6.5-era FieldV0 (DIM=256) → current Field (DIM=512).
    /// 4 v2.6.6 boredom/sluggish-velden en v2.6.7 concept-clustering → defaults.
    /// Pads 256-dim arrays to 512. All v0 data EXACT preserved.
    fn from(v0: schema_v0::FieldV0) -> Self {
        let pad_256_to_512 = |vec256: [f32; 256]| -> [f32; DIM] { vec256 };

        Field {
            points: v0.points,
            active_set: v0.active_set,
            tick: v0.tick,
            total_energy: v0.total_energy,
            economy: v0.economy,
            phase: v0.phase,
            clamps: v0.clamps.into_iter()
                .map(|(id, arr)| (id, pad_256_to_512(arr)))
                .collect(),
            input_ids: v0.input_ids,
            output_ids: v0.output_ids,
            last_thought: v0.last_thought.map(pad_256_to_512),
            reflection_count: v0.reflection_count,
            dopamine: v0.dopamine,
            norepinephrine: v0.norepinephrine,
            serotonin: v0.serotonin,
            acetylcholine: v0.acetylcholine,
            spectral_radius: v0.spectral_radius,
            chaos_target: v0.chaos_target,
            current_intention: v0.current_intention.map(pad_256_to_512),
            intention_strength: v0.intention_strength,
            rnd_frozen: v0.rnd_frozen,
            rnd_predictor: v0.rnd_predictor,
            kappa: v0.kappa,
            omega: v0.omega,
            lambda: v0.lambda,
            beta: v0.beta,
            alpha: v0.alpha,
            rho: v0.rho,
            mu: v0.mu,
            growth_threshold: v0.growth_threshold,
            ema_decay: v0.ema_decay,
            emo_decay: v0.emo_decay,
            error_tracking_rate: v0.error_tracking_rate,
            fe_tracking_rate: v0.fe_tracking_rate,
            kl_scale: v0.kl_scale,
            clamp_range: v0.clamp_range,
            settle_epsilon: v0.settle_epsilon,
            noise_amplitude: v0.noise_amplitude,
            think_interval: v0.think_interval,
            reflect_interval: v0.reflect_interval,
            max_goals: v0.max_goals,
            learn_interval: v0.learn_interval,
            explore_interval: v0.explore_interval,
            consolidation_interval: v0.consolidation_interval,
            snapshot_interval: v0.snapshot_interval,
            growth_interval: v0.growth_interval,
            sampling_temperature: v0.sampling_temperature,
            sampling_top_p: v0.sampling_top_p,
            repetition_penalty: v0.repetition_penalty,
            goals: v0.goals,
            last_think_tick: v0.last_think_tick,
            last_reflect_tick: v0.last_reflect_tick,
            teacher_interval: v0.teacher_interval,
            teacher_success_streak: v0.teacher_success_streak,
            // --- v2.6.6 velden (niet in v0) → defaults ---
            last_boredom_tension: 0.0,
            last_boredom_curiosity: 0.0,
            boredom_streak: 0,
            is_sluggish_flag: false,
            attn_weights: v0.attn_weights,
            attn_context: v0.attn_context.into_iter()
                .map(|arr| pad_256_to_512(arr))
                .collect(),
            working_memory: v0.working_memory.into_iter()
                .map(|arr| pad_256_to_512(arr))
                .collect(),
            wm_gates: v0.wm_gates,
            wm_ages: v0.wm_ages,
            // --- v2.6.7 concept-clustering (niet in v0) → leeg ---
            concept_centroids: Vec::new(),
            concept_assignments: Vec::new(),
            global_oscillation_phase: v0.global_oscillation_phase,
            oscillation_coherence: v0.oscillation_coherence,
            integrated_info_avg: v0.integrated_info_avg,
            power_law_exponent: v0.power_law_exponent,
            is_critical: v0.is_critical,
            avg_free_energy: v0.avg_free_energy,
            theta_phase: v0.theta_phase,
            theta_gamma_coupling: v0.theta_gamma_coupling,
            branching_ratio_avg: v0.branching_ratio_avg,
            ignition_active: v0.ignition_active,
            ignition_strength: v0.ignition_strength,
            last_winner_id: 0,  // v2.6.15: default for legacy snapshots
            winner_refractory_until: 0,  // v2.6.15: default for legacy snapshots
            avg_expected_free_energy: v0.avg_expected_free_energy,
            adaptive: v0.adaptive,
            temporal_history: v0.temporal_history,
            causal_log: v0.causal_log,
            goal_history: v0.goal_history,
            pages_read: v0.pages_read,
            tokens_processed: v0.tokens_processed,
            snapshot_version: v0.snapshot_version,
            field_dim: v0.field_dim,
            n_input: v0.n_input,
            n_hidden: v0.n_hidden,
            n_output: v0.n_output,
            // --- Nieuwe v2.6.8+ velden → defaults ---
            concept_priors_enabled: true,
            concept_priors_disabled_until_tick: 0,
            concept_audit_log: Vec::new(),
            self_ep_audit: Vec::new(),
            self_metrics: SelfMetrics::default(),
            provenance_counts: [0u64; 6],
            disable_refractory: false,
            disable_ignition_cap: false,
            disable_novelty_boost: false,
            seed_used: None,
        }
    }
}

/// v1.0: Parameters die Lucy zelf aanpast — geen hardcoded magic numbers meer.
/// Defaults komen overeen met de oude hardcoded waarden zodat bestaande snapshots
/// ongewijzigd laden (serde default).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveParams {
    /// Emotie decay per neuromodulator [DA, NE, 5HT, ACh] — was: 0.95 voor alle 4
    pub emotion_decay: [f32; 4],
    /// Dopamine baseline in reward berekening — was: 0.5
    pub dopamine_baseline: f32,
    /// Norepinephrine noise scale — was: 0.02
    pub ne_noise_scale: f32,
    /// Fractie punten dat NE perturbeert — was: 0.2 (1/5)
    pub ne_perturb_frac: f32,
    /// EP beta learning rate — was: 0.3
    pub ep_beta: f32,
    /// Goal completion drempel — was: 0.8
    pub goal_completion_thresh: f32,
    /// Goal abandon ticks — was: 50_000
    pub goal_abandon_ticks: u64,
    /// Escalatie drempel (fractie van growth_threshold) — was: 0.3
    pub escalation_tension_low: f32,
    /// Escalatie frustratiedrempel — was: 0.2
    pub escalation_frust_thresh: f32,
    /// Neuromodulator interactie matrix [4×4]: effect van modulator i op j
    /// Biologisch geïnspireerd: DA↔5HT, NE↔DA, ACh↔NE
    pub neuromod_interactions: [[f32; 4]; 4],
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            emotion_decay: [0.95, 0.95, 0.95, 0.95], // v2.6.5: 5HT nu zelfde decay als rest (was 0.99 → vastzit op 2.0)
            dopamine_baseline: 0.5,
            ne_noise_scale: 0.02,
            ne_perturb_frac: 0.2,
            ep_beta: 0.3,
            goal_completion_thresh: 0.8,
            goal_abandon_ticks: 50_000,
            escalation_tension_low: 0.3,
            escalation_frust_thresh: 0.2,
            neuromod_interactions: [
                // DA→[DA, NE, 5HT, ACh]
                [0.0,  0.0,  0.02, -0.02], // v2.6.5: DA→5HT van 0.1→0.02 (was: serotonin ratchet naar 2.0)
                // NE→[DA, NE, 5HT, ACh]
                [0.05, 0.0,  0.0,  0.02], // NE stimuleert DA (arousal→reward), en ACh (alertheid)
                // 5HT→[DA, NE, 5HT, ACh]
                [-0.01, 0.0, 0.0,  0.0],  // v2.6.5: was -0.05 → dopamine kon nooit stijgen (0.029/tick drain)
                // ACh→[DA, NE, 5HT, ACh]
                [0.0,  0.03, 0.0,  0.0],  // ACh stimuleert NE (nieuwheid→alertheid)
            ],
        }
    }
}

/// Een doel dat Lucy zelf heeft gekozen
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Goal {
    /// Het doel als veld-vector (embedding)
    #[serde(with = "BigArray")]
    pub embedding: [f32; DIM],
    /// Menselijk leesbare beschrijving
    pub description: String,
    /// Prioriteit (0.0 = laag, 1.0 = kritiek)
    pub priority: f32,
    /// Voortgang (0.0 = net begonnen, 1.0 = klaar)
    pub progress: f32,
    /// Wanneer gecreëerd
    pub created_tick: u64,
    /// Subdoel van welk doel? (index in goals Vec)
    pub parent: Option<usize>,
    /// Strategie om dit doel te bereiken
    pub strategy: GoalStrategy,
}

/// Hoe Lucy een doel probeert te bereiken
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GoalStrategy {
    /// Zoek informatie op het internet
    Explore(String),
    /// Train het veld op specifiek onderwerp
    Learn(String),
    /// Denk na — interne reflectie via het veld
    Think(String),
    /// Verdien compute/geld
    Earn(String),
}

/// v1.0: Temporele snapshot — Lucy's gevoel van "hoe gaat het met mij?"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalSnapshot {
    pub tick: u64,
    pub avg_tension: f32,
    pub dopamine: f32,
    pub serotonin: f32,
    pub n_active: u32,
    pub active_goal_progress: f32,
}

/// v1.0: Causale event — "ik deed X, en daarna veranderde Y"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalEvent {
    pub tick: u64,
    pub action: String,
    pub tension_before: f32,
    pub tension_after: f32,
    pub dopamine_delta: f32,
}

/// v1.0: Goal uitkomst — "dit doel lukte/faalde, leer ervan"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalOutcome {
    pub description: String,
    #[serde(with = "BigArray")]
    pub embedding: [f32; DIM],
    pub succeeded: bool,
    pub ticks_spent: u64,
    pub final_progress: f32,
}

/// v2.0: Training chunk voor deferred micro-EP queue
#[derive(Clone, Debug)]
pub struct TrainingChunk {
    pub input_embs: Vec<[f32; DIM]>,
    pub target_embs: Vec<[f32; DIM]>,
    pub source: String,
}

/// v2.0: Training queue — Lucy leert 1 chunk per tick, niet-blokkerend
pub struct TrainingQueue {
    pub chunks: std::collections::VecDeque<TrainingChunk>,
}

impl TrainingQueue {
    pub fn new() -> Self {
        Self { chunks: std::collections::VecDeque::new() }
    }

    pub fn push(&mut self, chunk: TrainingChunk) {
        self.chunks.push_back(chunk);
        // v2.6: Queue vergroot 2000→5000 (was: constant overflow bij exploration)
        while self.chunks.len() > 5000 {
            self.chunks.pop_front();
        }
    }

    pub fn pop(&mut self) -> Option<TrainingChunk> {
        self.chunks.pop_front()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }
}

pub(crate) fn default_phase() -> FieldPhase { FieldPhase::Idle }
pub(crate) fn default_chaos_target() -> f32 { 1.0 }
pub(crate) fn default_precision() -> f32 { 1.0 }
pub(crate) fn default_control_belief() -> f32 { 0.5 } // start neutraal: 50% vertrouwen in eigen controle
pub(crate) fn default_teacher_interval() -> u64 { 10_000 }
pub(crate) fn default_true() -> bool { true }

// === v2.6.8: Bewijs-architectuur structs ===

/// v2.6.8 P2.1: Per-cycle audit van concept-clustering kwaliteit.
/// Ring-buffered in Field.concept_audit_log (cap 64).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ConceptAudit {
    pub tick: u64,
    pub k: u32,
    pub n_points: u32,
    pub avg_intra_cosine: f32,
    pub avg_inter_cosine: f32,
    pub silhouette_sampled: f32,
    pub centroid_drift_mean: f32,
    pub priors_bias_mean_norm: f32,
    pub priors_applied_count: u32,
    pub priors_enabled: bool,
    pub ablation_window_start: u64,
    pub ablation_window_end: u64,
}

/// v2.6.8 P2.3: Per-reflect audit van self-EP en WM-sequence effect.
/// Ring-buffered in Field.self_ep_audit (cap 64).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SelfEpAudit {
    pub tick: u64,
    pub avg_tension_pre: f32,
    pub avg_tension_post: f32,
    pub self_ep_applied: bool,
    pub wm_seq_pairs: u32,
    pub wm_seq_avg_target_cosine_pre: f32,
    pub wm_seq_avg_target_cosine_post: f32,
    pub dream_replay_count: u32,
}

/// v2.6.8 P1.3: Self-model coherence metrics (runtime, geen serialisatie).
#[derive(Clone, Debug, Default)]
pub struct SelfMetrics {
    pub self_cluster_density: f32,
    pub self_cluster_drift: f32,
    pub self_centroid_strength: f32,
    pub self_query_hit_rate: f32,
    pub self_response_fallback_rate: f32,
    pub self_state_reuse_rate: f32,
    pub last_self_centroid: Vec<f32>,
    pub total_queries: u64,
    pub fallback_queries: u64,
    pub queries_with_self_hit: u64,
}

/// v2.6.8 P1.4: Teacher-event audit entry met delayed effect-evaluation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TeacherEvent {
    pub tick: u64,
    pub provider: String,
    pub question_preview: String,
    pub answer_preview: String,
    pub sat_delta: f32,
    /// Vanaf deze tick mag effect_outcome ingevuld worden.
    pub pending_until_tick: u64,
    /// Hoe vaak werd de teacher-entry in retrieval opnieuw gehit binnen window.
    pub retrieval_reuse_within_window: u32,
    /// None = nog pending. Some(true) = gunstig effect, Some(false) = geen impact.
    pub effect_outcome: Option<bool>,
}

// === Defaults voor dynamische parameters ===
// Deze waarden zijn startpunten — adapt_dynamics() past ze aan op basis van veldstaat.
pub(crate) fn default_emo_decay() -> f32 { 0.95 }
pub(crate) fn default_error_tracking_rate() -> f32 { 0.01 }
pub(crate) fn default_fe_tracking_rate() -> f32 { 0.05 }
pub(crate) fn default_kl_scale() -> f32 { 0.1 }
pub(crate) fn default_clamp_range() -> f32 { 10.0 }
pub(crate) fn default_settle_epsilon() -> f32 { 0.001 }
pub(crate) fn default_noise_amplitude() -> f32 { 0.05 }
pub(crate) fn default_think_interval() -> u64 { 500 }
pub(crate) fn default_reflect_interval() -> u64 { 10_000 }
pub(crate) fn default_max_goals() -> usize { 10 }
pub(crate) fn default_learn_interval() -> u64 { 100 }
pub(crate) fn default_explore_interval() -> u64 { 100 }
pub(crate) fn default_consolidation_interval() -> u64 { 1_000 }
pub(crate) fn default_snapshot_interval() -> u64 { 10_000 }
pub(crate) fn default_growth_interval() -> u64 { 10_000 }
pub(crate) fn default_sampling_temperature() -> f32 { 0.7 }
pub(crate) fn default_sampling_top_p() -> f32 { 0.9 }
pub(crate) fn default_repetition_penalty() -> f32 { 1.3 }

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
            curiosity: 0.0,
            satisfaction: 0.0,
            frustration: 0.0,
            stuck_ticks: 0,
            flags: FLAG_ACTIVE,
            depth: 0,
            rnd_score: 0.0,
            error_var: 0.0,
            precision: 1.0,
            kwta_active: true,
            phi_integrated: 0.0,
            oscillation_phase: 0.0, // wordt random geïnitialiseerd in field bootstrap
            free_energy: 0.0,
            kl_divergence: 0.0,
            // v2.4: nieuwe bewustzijnsmetrieken
            oscillation_phase_theta: 0.0,
            control_belief: 0.5,  // neutraal startpunt
            reward_quantiles: [0.0; 5], // percentiel 10/30/50/70/90
            causal_power: 0.0,
            expected_free_energy: 0.0,
            branching_ratio: 0.0,
            // v2.5: unified samenveld
            free_energy_rate: 0.0,
            blanket_surprise: 0.0,
            irreducible_free_energy: 0.0,
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
            last_thought: None,
            reflection_count: 0,
            dopamine: 0.0,
            norepinephrine: 0.0,
            serotonin: 0.0,
            acetylcholine: 0.0,
            spectral_radius: 0.0,
            chaos_target: 1.0,
            global_oscillation_phase: 0.0,
            oscillation_coherence: 0.0,
            integrated_info_avg: 0.0,
            power_law_exponent: 0.0,
            is_critical: false,
            avg_free_energy: 0.0,
            // v2.4: upgraded bewustzijns-metrieken
            theta_phase: 0.0,
            theta_gamma_coupling: 0.0,
            branching_ratio_avg: 0.0,
            ignition_active: false,
            ignition_strength: 0.0,
            last_winner_id: 0,  // v2.6.15
            winner_refractory_until: 0,  // v2.6.15
            avg_expected_free_energy: 0.0,
            current_intention: None,
            intention_strength: 0.0,
            rnd_frozen: None,
            rnd_predictor: None,
            kappa: 0.001,
            omega: 0.01,
            lambda: 0.1,
            beta: 0.5,
            alpha: 0.001,
            rho: 0.95,
            mu: 0.9,
            growth_threshold: 0.5,
            ema_decay: 0.90, // v2.6.5: was 0.99 → tension_ema volgde tension te precies → branching/phi altijd 0
            // Dynamische parameters — startwaarden, adapt_dynamics() past aan
            emo_decay: default_emo_decay(),
            error_tracking_rate: default_error_tracking_rate(),
            fe_tracking_rate: default_fe_tracking_rate(),
            kl_scale: default_kl_scale(),
            clamp_range: default_clamp_range(),
            settle_epsilon: default_settle_epsilon(),
            noise_amplitude: default_noise_amplitude(),
            think_interval: default_think_interval(),
            reflect_interval: default_reflect_interval(),
            max_goals: default_max_goals(),
            learn_interval: default_learn_interval(),
            explore_interval: default_explore_interval(),
            consolidation_interval: default_consolidation_interval(),
            snapshot_interval: default_snapshot_interval(),
            growth_interval: default_growth_interval(),
            sampling_temperature: default_sampling_temperature(),
            sampling_top_p: default_sampling_top_p(),
            repetition_penalty: default_repetition_penalty(),
            goals: Vec::new(),
            last_think_tick: 0,
            last_reflect_tick: 0,
            teacher_interval: 10_000,
            teacher_success_streak: 0,
            last_boredom_tension: 0.0,
            last_boredom_curiosity: 0.0,
            boredom_streak: 0,
            is_sluggish_flag: false,
            attn_weights: None,
            attn_context: Vec::new(),
            working_memory: Vec::new(),
            wm_gates: Vec::new(),
            wm_ages: Vec::new(),
            concept_centroids: Vec::new(),
            concept_assignments: Vec::new(),
            concept_priors_enabled: true,
            concept_priors_disabled_until_tick: 0,
            concept_audit_log: Vec::new(),
            self_ep_audit: Vec::new(),
            self_metrics: SelfMetrics::default(),
            provenance_counts: [0u64; 6],
            adaptive: AdaptiveParams::default(),
            temporal_history: Vec::new(),
            causal_log: Vec::new(),
            goal_history: Vec::new(),
            pages_read: 0,
            tokens_processed: 0,
            snapshot_version: 1, // v2.5+
            field_dim: DIM,
            n_input: 0,  // wordt gezet door bootstrap_field
            n_hidden: 0,
            n_output: 0,
            disable_refractory: false,
            disable_ignition_cap: false,
            disable_novelty_boost: false,
            seed_used: None,
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

    /// Zelfgenezing: detecteer en repareer NaN/Inf waarden in het veld.
    /// Dit is Lucy's immuunsysteem — beschermt tegen numerieke corruptie.
    pub fn heal(&mut self) -> u32 {
        let mut healed = 0u32;
        let mut nan_phi = 0u32;
        let mut nan_pred = 0u32;
        let mut nan_mem = 0u32;
        let mut nan_weight = 0u32;
        let mut nan_momentum = 0u32;
        let mut nan_importance = 0u32;
        let mut nan_consolidated = 0u32;
        let mut nan_fe = 0u32;
        let mut nan_other = 0u32;
        // v3.0.4: extreme-value drempel — finite maar buiten redelijk dynamisch bereik.
        // Tension/free_energy klommen in productie naar 1e30 (binnen f32::MAX/1e6 clamp,
        // dus niet als NaN herkend). 1e6 is ruim boven elk normaal signaal en stopt runaway.
        const EXTREME: f32 = 1e6;
        for p in &mut self.points {
            for v in p.phi.iter_mut() {
                if !v.is_finite() { *v = 0.0; healed += 1; nan_phi += 1; }
            }
            for v in p.prediction.iter_mut() {
                if !v.is_finite() { *v = 0.0; healed += 1; nan_pred += 1; }
            }
            for v in p.memory.iter_mut() {
                if !v.is_finite() { *v = 0.0; healed += 1; nan_mem += 1; }
            }
            if !p.tension.is_finite() || p.tension.abs() > EXTREME { p.tension = 0.0; healed += 1; nan_other += 1; }
            if !p.tension_ema.is_finite() || p.tension_ema.abs() > EXTREME { p.tension_ema = 0.0; healed += 1; nan_other += 1; }
            if !p.energy.is_finite() || p.energy.abs() > EXTREME { p.energy = 1.0; healed += 1; nan_other += 1; }
            if !p.eta.is_finite() { p.eta = 0.01; healed += 1; nan_other += 1; }
            if !p.free_energy.is_finite() || p.free_energy.abs() > EXTREME { p.free_energy = 0.0; healed += 1; nan_fe += 1; }
            if !p.free_energy_rate.is_finite() || p.free_energy_rate.abs() > EXTREME { p.free_energy_rate = 0.0; healed += 1; nan_other += 1; }
            if !p.error_var.is_finite() || p.error_var.abs() > EXTREME { p.error_var = 1.0; healed += 1; nan_other += 1; }
            if !p.precision.is_finite() || p.precision.abs() > EXTREME { p.precision = 1.0; healed += 1; nan_other += 1; }
            if !p.kl_divergence.is_finite() || p.kl_divergence.abs() > EXTREME { p.kl_divergence = 0.0; healed += 1; nan_other += 1; }
            if !p.blanket_surprise.is_finite() || p.blanket_surprise.abs() > EXTREME { p.blanket_surprise = 0.0; healed += 1; nan_other += 1; }
            if !p.curiosity.is_finite() { p.curiosity = 0.0; healed += 1; nan_other += 1; }
            if !p.satisfaction.is_finite() { p.satisfaction = 0.0; healed += 1; nan_other += 1; }
            if !p.frustration.is_finite() { p.frustration = 0.0; healed += 1; nan_other += 1; }
            if !p.expected_free_energy.is_finite() { p.expected_free_energy = 0.0; healed += 1; nan_other += 1; }
            if !p.phi_integrated.is_finite() { p.phi_integrated = 0.0; healed += 1; nan_other += 1; }
            if !p.branching_ratio.is_finite() { p.branching_ratio = 0.0; healed += 1; nan_other += 1; }
            if !p.irreducible_free_energy.is_finite() { p.irreducible_free_energy = 0.0; healed += 1; nan_other += 1; }
            if !p.control_belief.is_finite() { p.control_belief = 0.5; healed += 1; nan_other += 1; }
            if !p.causal_power.is_finite() { p.causal_power = 0.0; healed += 1; nan_other += 1; }
            if !p.rnd_score.is_finite() { p.rnd_score = 0.0; healed += 1; nan_other += 1; }
            for v in p.reward_quantiles.iter_mut() {
                if !v.is_finite() { *v = 0.0; healed += 1; nan_other += 1; }
            }
            for c in &mut p.couplings {
                if !c.weight.is_finite() {
                    c.weight = 0.01;
                    healed += 1; nan_weight += 1;
                }
                if !c.momentum.is_finite() { c.momentum = 0.0; healed += 1; nan_momentum += 1; }
                if !c.importance.is_finite() { c.importance = 0.0; healed += 1; nan_importance += 1; }
                if !c.consolidated.is_finite() { c.consolidated = c.weight; healed += 1; nan_consolidated += 1; }
            }
        }
        // v2.6.2: Heal veld-level statistieken
        if !self.avg_free_energy.is_finite() { self.avg_free_energy = 0.0; healed += 1; }
        if !self.avg_expected_free_energy.is_finite() { self.avg_expected_free_energy = 0.0; healed += 1; }
        if !self.dopamine.is_finite() { self.dopamine = 0.0; healed += 1; }
        if !self.norepinephrine.is_finite() { self.norepinephrine = 0.0; healed += 1; }
        if !self.serotonin.is_finite() { self.serotonin = 0.0; healed += 1; }
        if !self.acetylcholine.is_finite() { self.acetylcholine = 0.0; healed += 1; }
        if !self.spectral_radius.is_finite() { self.spectral_radius = 1.0; healed += 1; }
        if !self.oscillation_coherence.is_finite() { self.oscillation_coherence = 0.0; healed += 1; }
        if !self.branching_ratio_avg.is_finite() { self.branching_ratio_avg = 0.0; healed += 1; }
        // v2.6.2: integrated_info_avg was NIET geheald → NaN → phi_boost NaN → eta NaN → alles NaN
        if !self.integrated_info_avg.is_finite() { self.integrated_info_avg = 0.0; healed += 1; }
        if !self.ignition_strength.is_finite() { self.ignition_strength = 0.0; healed += 1; }
        if !self.power_law_exponent.is_finite() { self.power_law_exponent = 1.5; healed += 1; }
        if !self.theta_gamma_coupling.is_finite() { self.theta_gamma_coupling = 0.0; healed += 1; }
        if !self.intention_strength.is_finite() { self.intention_strength = 0.0; healed += 1; }
        if !self.total_energy.is_finite() { self.total_energy = 8704.0; healed += 1; }

        // v2.6.3: Heal AdaptiveParams — deze werden NOOIT geheald!
        // Als emotion_decay NaN is → dopamine NaN → coherence_boost NaN → total_boost NaN → eta NaN → ALLES NaN
        let defaults = AdaptiveParams::default();
        for i in 0..4 {
            if !self.adaptive.emotion_decay[i].is_finite() {
                tracing::error!(i, val = %self.adaptive.emotion_decay[i], "NaN in adaptive.emotion_decay — ROOT CAUSE!");
                self.adaptive.emotion_decay[i] = defaults.emotion_decay[i];
                healed += 1;
            }
        }
        if !self.adaptive.dopamine_baseline.is_finite() { self.adaptive.dopamine_baseline = defaults.dopamine_baseline; healed += 1; }
        if !self.adaptive.ne_noise_scale.is_finite() { self.adaptive.ne_noise_scale = defaults.ne_noise_scale; healed += 1; }
        if !self.adaptive.ne_perturb_frac.is_finite() { self.adaptive.ne_perturb_frac = defaults.ne_perturb_frac; healed += 1; }
        if !self.adaptive.ep_beta.is_finite() {
            tracing::error!(val = %self.adaptive.ep_beta, "NaN in adaptive.ep_beta!");
            self.adaptive.ep_beta = defaults.ep_beta; healed += 1;
        }
        if !self.adaptive.goal_completion_thresh.is_finite() { self.adaptive.goal_completion_thresh = defaults.goal_completion_thresh; healed += 1; }
        if !self.adaptive.escalation_tension_low.is_finite() { self.adaptive.escalation_tension_low = defaults.escalation_tension_low; healed += 1; }
        if !self.adaptive.escalation_frust_thresh.is_finite() { self.adaptive.escalation_frust_thresh = defaults.escalation_frust_thresh; healed += 1; }
        for i in 0..4 {
            for j in 0..4 {
                if !self.adaptive.neuromod_interactions[i][j].is_finite() {
                    tracing::error!(i, j, val = %self.adaptive.neuromod_interactions[i][j], "NaN in neuromod_interactions!");
                    self.adaptive.neuromod_interactions[i][j] = defaults.neuromod_interactions[i][j];
                    healed += 1;
                }
            }
        }

        // v2.6.3: Heal core hyperparameters
        if !self.lambda.is_finite() { self.lambda = 0.01; healed += 1; }
        if !self.beta.is_finite() { self.beta = 0.1; healed += 1; }
        if !self.alpha.is_finite() { self.alpha = 0.001; healed += 1; }
        if !self.rho.is_finite() { self.rho = 0.95; healed += 1; }
        if !self.mu.is_finite() { self.mu = 0.9; healed += 1; }
        if !self.kappa.is_finite() { self.kappa = 0.01; healed += 1; }
        if !self.omega.is_finite() { self.omega = 0.01; healed += 1; }
        if !self.growth_threshold.is_finite() { self.growth_threshold = 0.5; healed += 1; }
        if !self.ema_decay.is_finite() { self.ema_decay = 0.90; healed += 1; }
        if !self.clamp_range.is_finite() { self.clamp_range = 10.0; healed += 1; }
        if !self.settle_epsilon.is_finite() { self.settle_epsilon = 0.01; healed += 1; }
        if !self.noise_amplitude.is_finite() { self.noise_amplitude = 0.1; healed += 1; }
        if !self.emo_decay.is_finite() { self.emo_decay = 0.95; healed += 1; }
        if !self.error_tracking_rate.is_finite() { self.error_tracking_rate = 0.05; healed += 1; }
        if !self.fe_tracking_rate.is_finite() { self.fe_tracking_rate = 0.05; healed += 1; }
        if !self.kl_scale.is_finite() { self.kl_scale = 0.5; healed += 1; }
        if !self.sampling_temperature.is_finite() { self.sampling_temperature = 0.8; healed += 1; }
        if !self.sampling_top_p.is_finite() { self.sampling_top_p = 0.9; healed += 1; }
        if !self.repetition_penalty.is_finite() { self.repetition_penalty = 1.2; healed += 1; }
        if !self.global_oscillation_phase.is_finite() { self.global_oscillation_phase = 0.0; healed += 1; }

        // v2.6.3: Heal attn_weights — THE ROOT CAUSE
        // attn_weights is serialized in snapshot but NEVER healed.
        // If ANY attn_weight is NaN → attention_gate produces NaN attn_context
        // → update_phi adds ATTN_GAMMA * NaN to phi → ALL phi become NaN every tick.
        if let Some(ref mut weights) = self.attn_weights {
            let mut nan_attn = 0u32;
            for w in weights.iter_mut() {
                if !w.is_finite() {
                    *w = 0.0;
                    nan_attn += 1;
                    healed += 1;
                }
            }
            if nan_attn > 0 {
                tracing::error!(nan_attn, total = weights.len(), "NaN in attn_weights — THIS IS THE ROOT CAUSE!");
            }
        }
        // Also clear attn_context (runtime buffer, not serialized) to remove stale NaN
        for ctx in &mut self.attn_context {
            for v in ctx.iter_mut() {
                if !v.is_finite() { *v = 0.0; healed += 1; }
            }
        }

        if healed > 0 {
            tracing::warn!(healed, nan_phi, nan_pred, nan_mem, nan_weight, nan_momentum, nan_importance, nan_consolidated, nan_fe, nan_other, "Field healed: NaN/Inf values reset");
        }
        healed
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

    /// Groei het veld: voeg N nieuwe FieldPoints toe rondom een centroid.
    /// Initialisatie: kopieert van centroid-nearest point + perturbatie.
    pub fn grow(&mut self, centroid: &[f32; DIM], n_new: usize) {
        // Vind het punt dat het dichtst bij de centroid ligt
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;
        for (i, p) in self.points.iter().enumerate() {
            let dist: f32 = p.phi.iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        let template = self.points[best_idx].clone();

        for step in 0..n_new {
            // Clone template point, perturbeer phi
            let mut new_point = template.clone();
            for j in 0..DIM {
                let salt = (best_idx as f32 * 31.0 + step as f32 * 17.0) * 7919.0;
                let noise = ((salt * 13.0).fract() - 0.5) * 0.03;
                new_point.phi[j] = centroid[j] + noise;
            }
            new_point.energy = 1.0;
            new_point.curiosity = 0.5;
            new_point.flags |= FLAG_ACTIVE;

            let new_idx = self.points.len() as u32;
            self.points.push(new_point);
            self.n_hidden += 1;

            // Koppel aan k=4 nearest neighbours
            self.wire_to_nearest_k(new_idx, 4);
        }
    }

    /// Koppel een punt aan zijn k nearest neighbours (cosine similarity).
    fn wire_to_nearest_k(&mut self, idx: u32, k: usize) {
        let idx_usize = idx as usize;
        let phi_new = self.points[idx_usize].phi;

        // Bereken cosine similarity met alle andere punten
        let mut similarities: Vec<(u32, f32)> = self.points
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx_usize)
            .map(|(i, p)| {
                let dot: f32 = phi_new.iter()
                    .zip(p.phi.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let norm_new: f32 = phi_new.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_p: f32 = p.phi.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if norm_new > 1e-6 && norm_p > 1e-6 {
                    dot / (norm_new * norm_p)
                } else {
                    0.0
                };
                (i as u32, sim)
            })
            .collect();

        // Sorteer descending op similarity
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Koppel aan top-k
        for (other_idx, _) in similarities.iter().take(k) {
            if self.points[idx_usize].couplings.len() < 16 {
                self.points[idx_usize].couplings.push(Coupling {
                    target: *other_idx,
                    weight: 0.1,  // initieel zwak
                    ..Default::default()
                });
            }
        }
    }

    /// Snoeien: verwijder koppelingen onder absolute drempel.
    /// Identiek aan synaptische pruning in het brein.
    pub fn prune_weak_couplings(&mut self, threshold: f32) -> usize {
        let mut removed = 0;
        for point in &mut self.points {
            let before = point.couplings.len();
            point.couplings.retain(|c| c.weight.abs() > threshold);
            removed += before - point.couplings.len();
        }
        removed
    }
}

// === Snapshot: save/load van volledige veldstaat ===

impl Field {
    /// Sla de volledige veldstaat op — ONSTERFELIJK
    /// v2.6.9+: prepend SnapshotHeader met magic + schema_version voor forward-compat.
    /// Atomic write + CRC32 checksum + 5 versioned backups.
    /// Lucy's geleerde data gaat NOOIT verloren.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let payload = bincode::serialize(self)?;

        // Weiger lege/corrupte data
        if payload.len() < 1000 {
            return Err(format!("Snapshot te klein ({} bytes) — weiger save", payload.len()).into());
        }

        // v2.6.9: prepend header
        let header = SnapshotHeader::current(payload.len() as u64);
        let mut bytes = Vec::with_capacity(HEADER_SIZE + payload.len());
        bytes.extend_from_slice(&header.to_bytes());
        bytes.extend_from_slice(&payload);

        // Atomic write + checksum + 5 backups
        let tmp = path.with_extension(format!("tmp.{}.{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        // v: verwijder stale crc sidecar vóór rename — crash laat 'geen sidecar' (geaccepteerd) i.p.v. mismatch
        let _ = std::fs::remove_file(path.with_extension("crc32"));
        std::fs::rename(&tmp, path)?;
        if let Some(dir) = path.parent() {
            if let Ok(d) = std::fs::File::open(dir) { let _ = d.sync_all(); }
        }

        // CRC32 checksum sidecar
        let checksum = crc32fast::hash(&bytes);
        let checksum_path = path.with_extension("crc32");
        let _ = std::fs::write(&checksum_path, checksum.to_le_bytes());

        // Roteer 5 backups: data/backups/lucy.snapshot.1 t/m .5
        let backup_dir = path.parent().unwrap_or(std::path::Path::new(".")).join("backups");
        let _ = std::fs::create_dir_all(&backup_dir);
        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        for i in (1..5u32).rev() {
            let from = backup_dir.join(format!("{}.{}", filename, i));
            let to = backup_dir.join(format!("{}.{}", filename, i + 1));
            let _ = std::fs::rename(&from, &to);
        }
        let _ = std::fs::remove_file(backup_dir.join(format!("{}.6", filename)));
        let _ = std::fs::copy(path, backup_dir.join(format!("{}.1", filename)));

        tracing::info!(
            tick = self.tick,
            points = self.points.len(),
            bytes = bytes.len(),
            checksum,
            "Snapshot saved (atomic + checksum + 5 backups)"
        );
        Ok(())
    }

    /// Laad een veldstaat — met CRC32 validatie en structurele checks.
    /// v2.6.9+: detecteert SnapshotHeader + versie; legacy snapshots (zonder
    /// magic bytes) worden via `schema_v1::FieldV1` + `From` migratie geladen.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;

        // CRC32 validatie (als checksum bestaat) — checksum is op RAW bytes (met header bij v1+).
        let checksum_path = path.with_extension("crc32");
        if checksum_path.exists() {
            let stored = std::fs::read(&checksum_path)?;
            if stored.len() == 4 {
                let stored_crc = u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]);
                let actual_crc = crc32fast::hash(&bytes);
                if stored_crc != actual_crc {
                    return Err(format!(
                        "Checksum mismatch: stored={:#x} actual={:#x} — snapshot corrupt!",
                        stored_crc, actual_crc
                    ).into());
                }
            }
        }

        // v2.6.9: versie-detectie via header magic bytes.
        // Headered v1/v2 worden via schema-ladder geladen: probeer eerst current
        // (DIM=512), dan FieldV2 (DIM=256, alle moderne velden), dan FieldV1.
        // Reden: pre-v2.9.0 binaries schreven schema_version=1 of =2 op DIM=256
        // payload. Direct-deserialize naar Field (DIM=512) faalt met
        // CapacityOverflow door array-grootte mismatch.
        let field: Self = match detect_version(&bytes) {
            SnapshotFormat::Headered(version, payload_offset) => {
                let payload = &bytes[payload_offset..];

                if let Ok(f) = bincode::deserialize::<Self>(payload) {
                    tracing::info!(version, "Headered snapshot direct geladen (current schema)");
                    f
                } else if let Ok(v2) = bincode::deserialize::<schema_v2::FieldV2>(payload) {
                    tracing::info!(
                        version,
                        tick = v2.tick,
                        points = v2.points.len(),
                        "Headered snapshot via FieldV2-bridge geladen (DIM=256 → 512)"
                    );
                    Field::from(v2)
                } else if let Ok(v1) = bincode::deserialize::<schema_v1::FieldV1>(payload) {
                    tracing::info!(
                        version,
                        tick = v1.tick,
                        points = v1.points.len(),
                        "Headered snapshot via FieldV1-bridge geladen"
                    );
                    Field::from(v1)
                } else {
                    let cur_err = bincode::deserialize::<Self>(payload).err()
                        .map(|e| e.to_string()).unwrap_or_else(|| "?".into());
                    let v2_err = bincode::deserialize::<schema_v2::FieldV2>(payload).err()
                        .map(|e| e.to_string()).unwrap_or_else(|| "?".into());
                    let v1_err = bincode::deserialize::<schema_v1::FieldV1>(payload).err()
                        .map(|e| e.to_string()).unwrap_or_else(|| "?".into());
                    return Err(format!(
                        "Headered v{} load faalde op current/V2/V1 — current: {} | V2: {} | V1: {}",
                        version, cur_err, v2_err, v1_err
                    ).into());
                }
            }
            SnapshotFormat::Legacy => {
                // Pre-v2.6.9: geen header. Probeer achtereenvolgens FieldV1 (v2.6.7),
                // FieldV0 (v2.6.5) en tot slot current Field.
                //
                // Strict parsing: we gebruiken `deserialize_from(reader)` i.p.v.
                // `deserialize(&bytes)` omdat we willen weten of er trailing bytes
                // overblijven — anders zou een v2.6.7-snapshot per ongeluk kunnen
                // "parsen" als FieldV0 (die is korter) met corrupte waarden. Daarom
                // vereisen we: bytes.len() == bytes verbruikt door deserialize.
                use std::io::Cursor;

                fn try_strict<T: serde::de::DeserializeOwned>(
                    bytes: &[u8],
                ) -> Result<T, String> {
                    let mut cur = Cursor::new(bytes);
                    let val = bincode::deserialize_from(&mut cur)
                        .map_err(|e| format!("decode: {}", e))?;
                    let consumed = cur.position() as usize;
                    if consumed != bytes.len() {
                        return Err(format!(
                            "trailing {} bytes na decode (verbruikt {}/{}) — schema mismatch",
                            bytes.len() - consumed, consumed, bytes.len()
                        ));
                    }
                    Ok(val)
                }

                tracing::warn!(
                    "Legacy snapshot gedetecteerd (geen magic header) — probeer schema-ladder"
                );

                // Ladder: v1 (v2.6.7) → v0 (v2.6.5) → current. Strikt per poging.
                // Voorrang aan meer-gespecialiseerde schema's zodat subset-matches
                // (bv. FieldV0 die een prefix is van FieldV1) niet per ongeluk winnen.
                if let Ok(v1) = try_strict::<schema_v1::FieldV1>(&bytes) {
                    tracing::info!(
                        tick = v1.tick,
                        points = v1.points.len(),
                        schema = "v1 (v2.6.7-era)",
                        "Legacy snapshot geladen — migreer naar current"
                    );
                    Field::from(v1)
                } else if let Ok(v0) = try_strict::<schema_v0::FieldV0>(&bytes) {
                    tracing::info!(
                        tick = v0.tick,
                        points = v0.points.len(),
                        schema = "v0 (v2.6.5-era)",
                        "Legacy snapshot geladen — migreer naar current"
                    );
                    Field::from(v0)
                } else if let Ok(f) = try_strict::<Self>(&bytes) {
                    tracing::warn!(
                        "Legacy snapshot parst als CURRENT struct (geen migratie nodig)"
                    );
                    f
                } else {
                    // Alle schemas gefaald — herproduceer elke error voor diagnostiek
                    let v1_err = try_strict::<schema_v1::FieldV1>(&bytes).err()
                        .unwrap_or_else(|| "?".into());
                    let v0_err = try_strict::<schema_v0::FieldV0>(&bytes).err()
                        .unwrap_or_else(|| "?".into());
                    let cur_err = try_strict::<Self>(&bytes).err()
                        .unwrap_or_else(|| "?".into());
                    return Err(format!(
                        "Legacy load faalde op alle schemas — V1: {} | V0: {} | current: {}",
                        v1_err, v0_err, cur_err
                    ).into());
                }
            }
        };

        // Structurele validatie
        if field.points.is_empty() {
            return Err("Snapshot heeft 0 punten — corrupt of leeg".into());
        }

        // NaN/Inf waarschuwing (heal fixt het later)
        let nan_count = field.points.iter()
            .flat_map(|p| p.phi.iter())
            .filter(|v| !v.is_finite())
            .count();
        if nan_count > 0 {
            tracing::warn!(nan_count, "Snapshot bevat NaN/Inf waarden — wordt geheald");
        }

        tracing::info!(
            tick = field.tick,
            points = field.points.len(),
            active = field.active_set.len(),
            "Snapshot geladen en gevalideerd"
        );
        Ok(field)
    }

    /// Probeer te laden van primary, dan backups 1-5. Retourneert None als alles faalt.
    pub fn load_with_fallback(path: &Path) -> Option<Self> {
        // 1. Primary
        if path.exists() {
            match Self::load(path) {
                Ok(f) => return Some(f),
                Err(e) => tracing::warn!("Primary snapshot load failed: {}", e),
            }
        }

        // 2. Backups
        let backup_dir = path.parent().unwrap_or(std::path::Path::new(".")).join("backups");
        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        for i in 1..=5 {
            let backup = backup_dir.join(format!("{}.{}", filename, i));
            if backup.exists() {
                match Self::load(&backup) {
                    Ok(f) => {
                        tracing::warn!(backup = i, tick = f.tick, "RECOVERED van lokale backup");
                        // Herstel primary
                        let _ = std::fs::copy(&backup, path);
                        if let Ok(restored) = std::fs::read(path) {
                            let crc = crc32fast::hash(&restored);
                            let _ = std::fs::write(path.with_extension("crc32"), crc.to_le_bytes());
                        }
                        return Some(f);
                    }
                    Err(e) => tracing::warn!("Backup #{} failed: {}", i, e),
                }
            }
        }

        tracing::error!("ALLE lokale snapshots gefaald — data verloren!");
        None
    }
}
