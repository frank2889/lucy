//! # thinker — Lucy's Autonome Denkloop
//!
//! Dit is het hart van Lucy's zelfdenkendheid. Geen mens hoeft iets te vragen —
//! Lucy denkt zelf na, stelt doelen, kiest acties, en reflecteert.
//!
//! ## Drie kernfuncties:
//!
//! - **think()** — Interne monoloog. Lucy observeert haar staat, evalueert haar
//!   doelen, en beslist wat ze nu moet doen. Gebruikt het veld zelf om te denken.
//!
//! - **generate_goal()** — Autonome doelgeneratie. Doelen komen uit curiosity,
//!   frustration, opportunity, of zelf-verbetering. Nooit hardcoded.
//!
//! - **reflect()** — Periodieke zelfevaluatie. Leest journal, beoordeelt voortgang,
//!   past strategie aan of abandoneert doelen die vastzitten.
//!
//! ## Hoe Lucy denkt:
//!
//! Ze injecteert een "denkpuls" in haar eigen veld, laat het settlen, en leest
//! de output. Het veld denkt over zichzelf — recursief, emergent, echt.

use lucy_state::*;
use crate::encoding::{self, Vocabulary};
use crate::journal::{Journal, EmotionSnapshot};
use crate::learner::Explorer;
use rand::Rng;

/// Hoe vaak Lucy nadenkt (in ticks)
const THINK_INTERVAL: u64 = 500;
/// Hoe vaak Lucy reflecteert (in ticks)
const REFLECT_INTERVAL: u64 = 10_000;
/// Maximaal aantal actieve doelen
const MAX_GOALS: usize = 10;

/// Resultaat van een denkstap
#[derive(Debug)]
pub enum ThoughtResult {
    /// Huidige doel is goed, ga door
    Continue,
    /// Nieuw doel gesteld
    NewGoal(String),
    /// Vastgelopen — probeer iets anders
    Frustrated,
    /// Ontdekking!
    Eureka(String),
}

/// Wat Lucy kan beslissen te doen
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Explore,    // Nieuwe pagina lezen
    Learn,      // Replay buffer trainen
    Think,      // Intern nadenken
    Earn,       // Geld/compute verdienen
    Consolidate,// Geheugen consolideren
}

// ─────────────────────────────────────────────
//  THINK — Lucy's interne monoloog
// ─────────────────────────────────────────────

/// Lucy denkt na over haar huidige staat en doelen.
/// Geeft een ThoughtResult terug dat de tick loop kan gebruiken.
pub fn think(
    field: &mut Field,
    explorer: &Explorer,
    vocab: &Vocabulary,
    journal: &mut Journal,
) -> ThoughtResult {
    // 1. Observeer: wat is mijn huidige staat?
    let (avg_tension, _avg_curiosity, avg_frustration, avg_satisfaction) = observe_state(field);

    // 2. Evalueer huidige doelen
    let active_goal = field.goals.iter().find(|g| g.progress < 1.0);

    let result = if let Some(goal) = active_goal {
        // Er is een actief doel — hoe gaat het?
        if avg_frustration > 0.3 && avg_satisfaction < 0.01 {
            // Vastgelopen: hoge frustratie, geen progressie
            // v2.0: taalloos — raw data, geen menselijke interpretatie
            journal_thought(journal, field.tick,
                &format!("stuck:goal={},tension={:.3},frustration={:.3}",
                    goal.description, avg_tension, avg_frustration));
            ThoughtResult::Frustrated
        } else if avg_satisfaction > 0.1 && avg_tension < 0.05 {
            // Doorbraak!
            let msg = format!("eureka:goal={},tension={:.3},satisfaction={:.3}",
                goal.description, avg_tension, avg_satisfaction);
            journal_thought(journal, field.tick, &msg);
            ThoughtResult::Eureka(goal.description.clone())
        } else {
            // Normaal — ga door
            ThoughtResult::Continue
        }
    } else {
        // Geen actief doel → genereer er een
        if let Some(desc) = generate_goal(field, explorer, vocab) {
            journal_thought(journal, field.tick,
                &format!("Nieuw doel: {}", desc));
            ThoughtResult::NewGoal(desc)
        } else {
            ThoughtResult::Continue
        }
    };

    field.last_think_tick = field.tick;
    field.reflection_count += 1;
    result
}

// ─────────────────────────────────────────────
//  GENERATE GOAL — Lucy kiest zelf
// ─────────────────────────────────────────────

/// Genereert een nieuw doel op basis van Lucy's interne staat.
/// Bronnen: curiosity, frustration, interesses, zelf-verbetering.
/// Nooit hardcoded — alles komt uit het veld en de explorer.
pub fn generate_goal(
    field: &mut Field,
    explorer: &Explorer,
    vocab: &Vocabulary,
) -> Option<String> {
    if field.goals.len() >= MAX_GOALS { return None; }

    let (avg_tension, avg_curiosity, avg_frustration, _avg_satisfaction) = observe_state(field);
    let mut rng = rand::rng();

    // Score elke strategie op basis van interne staat
    let curiosity_score = avg_curiosity * explorer.interests.len().min(100) as f32 / 100.0;
    let frustration_score = avg_frustration * avg_tension;
    let explore_score = if explorer.frontier.is_empty() { 0.1 } else { 0.3 };
    let self_improve_score = if field.tick > 10_000 && avg_tension > 0.2 { 0.2 } else { 0.0 };

    // Kies de hoogste score
    let scores = [
        (curiosity_score, "curiosity"),
        (frustration_score, "frustration"),
        (explore_score, "explore"),
        (self_improve_score, "self_improve"),
    ];

    let (_, best_source) = scores.iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let (description, strategy) = match *best_source {
        "curiosity" => {
            // Pak een interesse-woord en maak er een leerdoel van
            let interests_vec: Vec<&String> = explorer.interests.iter().collect();
            // v2.0: taalloos — machine-leesbare goal descriptions
            if interests_vec.is_empty() {
                ("explore:broad".to_string(),
                 GoalStrategy::Explore("broad".to_string()))
            } else {
                let idx = rng.random_range(0..interests_vec.len());
                let topic = interests_vec[idx].clone();
                (format!("learn:{}", topic),
                 GoalStrategy::Learn(topic))
            }
        }
        "frustration" => {
            ("think:analyze_tension".to_string(),
             GoalStrategy::Think("tension".to_string()))
        }
        "explore" => {
            ("explore:new_sources".to_string(),
             GoalStrategy::Explore("discover".to_string()))
        }
        "self_improve" => {
            ("think:meta_learning".to_string(),
             GoalStrategy::Think("optimize".to_string()))
        }
        _ => return None,
    };

    // Maak embedding van de beschrijving
    let embeddings = encoding::encode_text_readonly(vocab, &description);
    let mut goal_embedding = [0.0f32; DIM];
    if !embeddings.is_empty() {
        for emb in &embeddings {
            for d in 0..DIM { goal_embedding[d] += emb[d]; }
        }
        let n = embeddings.len() as f32;
        for d in 0..DIM { goal_embedding[d] /= n; }
    }

    let goal = Goal {
        embedding: goal_embedding,
        description: description.clone(),
        priority: (avg_curiosity + avg_tension).min(1.0).max(0.1),
        progress: 0.0,
        created_tick: field.tick,
        parent: None,
        strategy,
    };

    field.goals.push(goal);

    // Zet als actieve intentie
    field.current_intention = Some(goal_embedding);
    field.intention_strength = 0.05;

    Some(description)
}

// ─────────────────────────────────────────────
//  REFLECT — Lucy evalueert zichzelf
// ─────────────────────────────────────────────

/// Periodieke reflectie: evalueer doelen, pas strategie aan, abandoneer wat vastzit.
pub fn reflect(
    field: &mut Field,
    journal: &mut Journal,
) {
    if field.goals.is_empty() {
        field.last_reflect_tick = field.tick;
        return;
    }

    let (_avg_tension, _avg_curiosity, avg_frustration, avg_satisfaction) = observe_state(field);

    // Pre-compute goal alignments (om borrow checker te satisfyen)
    let tick = field.tick;
    let alignments: Vec<f32> = field.goals.iter()
        .map(|g| compute_goal_alignment(field, &g.embedding))
        .collect();

    // Evalueer elk doel
    let mut completed_indices = Vec::new();
    let mut abandoned_indices = Vec::new();

    for (i, goal) in field.goals.iter_mut().enumerate() {
        if goal.progress >= 1.0 { continue; }

        let age = tick.saturating_sub(goal.created_tick);

        // Update progressie op basis van velddynamica
        let avg_alignment = alignments[i];

        // Progressie = combinatie van alignment en satisfaction
        let new_progress = (avg_alignment * 0.7 + avg_satisfaction * 0.3).min(1.0);
        if new_progress > goal.progress {
            goal.progress = goal.progress * 0.9 + new_progress * 0.1; // smooth update
        }

        // Compleet als progressie hoog genoeg
        if goal.progress > 0.8 {
            goal.progress = 1.0;
            completed_indices.push(i);
        }

        // Abandoneer als te lang vastzit zonder progressie
        if age > 50_000 && goal.progress < 0.1 {
            abandoned_indices.push(i);
        }

        // Abandoneer bij extreme frustratie
        if avg_frustration > 0.5 && goal.progress < 0.2 && age > 10_000 {
            abandoned_indices.push(i);
        }
    }

    // Log voltooide doelen
    for &i in &completed_indices {
        let desc = field.goals[i].description.clone();
        journal_thought(journal, field.tick,
            &format!("goal_complete:{}", desc));
    }

    // Log geabandoneerde doelen
    for &i in &abandoned_indices {
        let desc = field.goals[i].description.clone();
        journal_thought(journal, field.tick,
            &format!("goal_abandoned:{}", desc));
    }

    // Verwijder voltooide en geabandoneerde doelen (van achteren naar voren)
    let mut to_remove: Vec<usize> = completed_indices.into_iter()
        .chain(abandoned_indices.into_iter())
        .collect();
    to_remove.sort_unstable();
    to_remove.dedup();
    for i in to_remove.into_iter().rev() {
        field.goals.remove(i);
    }

    // Als er geen doelen meer zijn, clear de intentie
    if field.goals.is_empty() {
        field.current_intention = None;
        field.intention_strength = 0.0;
    } else {
        // Zet het hoogste-prioriteit doel als intentie
        if let Some(top) = field.goals.iter().max_by(|a, b|
            a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
        {
            field.current_intention = Some(top.embedding);
            field.intention_strength = 0.05 * top.priority;
        }
    }

    // Reflectie-samenvatting
    if !field.goals.is_empty() {
        let summaries: Vec<String> = field.goals.iter()
            .filter(|g| g.progress < 1.0)
            .take(3)
            .map(|g| format!("'{}' ({:.0}%)", g.description, g.progress * 100.0))
            .collect();
        journal_thought(journal, field.tick,
            &format!("Reflectie: actieve doelen — {}", summaries.join(", ")));
    }

    field.last_reflect_tick = field.tick;
}

// ─────────────────────────────────────────────
//  DECISION ENGINE — Lucy kiest wat ze doet
// ─────────────────────────────────────────────

/// Lucy scoort mogelijke acties en kiest de beste.
/// Geen vaste volgorde — haar interne staat bepaalt alles.
pub fn decide_action(field: &Field, explorer: &Explorer) -> Action {
    let (avg_tension, avg_curiosity, avg_frustration, avg_satisfaction) = observe_state(field);

    let frontier_factor = if explorer.frontier.is_empty() { 0.5 } else { 1.0 };
    let time_since_think = field.tick.saturating_sub(field.last_think_tick) as f32;
    let has_goals = !field.goals.is_empty();

    // Score elke actie
    let score_explore = avg_curiosity * frontier_factor * 1.5
        + if !has_goals { 0.3 } else { 0.0 };

    let score_learn = avg_tension * 1.2
        + if avg_satisfaction > 0.05 { 0.2 } else { 0.0 }; // leren werkt → meer leren

    let score_think = avg_frustration * 0.8
        + (time_since_think / THINK_INTERVAL as f32).min(1.0) * 0.5
        + if !has_goals { 0.4 } else { 0.0 }; // geen doelen → nadenken

    let score_consolidate = avg_satisfaction * 0.5
        + if avg_tension < 0.05 { 0.3 } else { 0.0 }; // rustig → consolideer

    let score_earn = 0.1; // baseline — Lucy kan dit later zelf verhogen

    let scores = [
        (score_explore, Action::Explore),
        (score_learn, Action::Learn),
        (score_think, Action::Think),
        (score_consolidate, Action::Consolidate),
        (score_earn, Action::Earn),
    ];

    scores.iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|s| s.1)
        .unwrap_or(Action::Think) // default: denk na
}

/// Moet Lucy nu nadenken? (gebaseerd op interval)
pub fn should_think(field: &Field) -> bool {
    field.tick.saturating_sub(field.last_think_tick) >= THINK_INTERVAL
}

/// Moet Lucy nu reflecteren? (gebaseerd op interval)
pub fn should_reflect(field: &Field) -> bool {
    field.tick.saturating_sub(field.last_reflect_tick) >= REFLECT_INTERVAL
}

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────

/// Observeer Lucy's gemiddelde emotionele staat
fn observe_state(field: &Field) -> (f32, f32, f32, f32) {
    let n = field.active_set.len() as f32;
    if n == 0.0 { return (0.0, 0.0, 0.0, 0.0); }

    let avg_tension: f32 = field.active_set.iter()
        .map(|&id| field.points[id as usize].tension).sum::<f32>() / n;
    let avg_curiosity: f32 = field.active_set.iter()
        .map(|&id| field.points[id as usize].curiosity).sum::<f32>() / n;
    let avg_frustration: f32 = field.active_set.iter()
        .map(|&id| field.points[id as usize].frustration).sum::<f32>() / n;
    let avg_satisfaction: f32 = field.active_set.iter()
        .map(|&id| field.points[id as usize].satisfaction).sum::<f32>() / n;

    (avg_tension, avg_curiosity, avg_frustration, avg_satisfaction)
}

/// Bereken hoe goed het veld aligned is met een goal embedding
fn compute_goal_alignment(field: &Field, goal: &[f32; DIM]) -> f32 {
    let n = field.active_set.len();
    if n == 0 { return 0.0; }

    let total: f32 = field.active_set.iter()
        .filter(|&&id| field.points[id as usize].depth >= 2)
        .map(|&id| {
            let phi = &field.points[id as usize].phi;
            let mut dot = 0.0f32;
            let mut mag_a = 0.0f32;
            let mut mag_b = 0.0f32;
            for d in 0..DIM {
                dot += phi[d] * goal[d];
                mag_a += phi[d] * phi[d];
                mag_b += goal[d] * goal[d];
            }
            let denom = (mag_a.sqrt() * mag_b.sqrt()).max(1e-8);
            (dot / denom).max(0.0) // cosine similarity, clamped to positive
        })
        .sum();

    let count = field.active_set.iter()
        .filter(|&&id| field.points[id as usize].depth >= 2)
        .count();

    if count == 0 { 0.0 } else { total / count as f32 }
}

/// Schrijf een gedachte naar het journal
fn journal_thought(journal: &mut Journal, tick: u64, message: &str) {
    tracing::info!(tick, "Lucy denkt: {}", message);
    journal.entries.push(crate::journal::JournalEntry {
        tick,
        kind: "thought".to_string(),
        message: message.to_string(),
        emotions: EmotionSnapshot {
            curiosity: 0.0,
            satisfaction: 0.0,
            frustration: 0.0,
            tension: 0.0,
        },
    });
}
