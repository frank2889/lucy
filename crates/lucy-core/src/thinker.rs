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

// v2.3: Geen hardcoded constanten meer.
// THINK_INTERVAL, REFLECT_INTERVAL, MAX_GOALS → field.think_interval, field.reflect_interval, field.max_goals
// Worden dynamisch aangepast door adapt_dynamics() in compute.rs.

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
        // v2.3: Relatieve drempels — geen vaste waarden
        // Frustrated = frustratie overheerst satisfaction (3× ratio)
        // Eureka = satisfaction hoog t.o.v. tension
        if avg_frustration > avg_tension * 3.0 && avg_satisfaction < avg_tension {
            journal_thought(journal, field.tick,
                &format!("stuck:goal={},tension={:.3},frustration={:.3}",
                    goal.description, avg_tension, avg_frustration));
            ThoughtResult::Frustrated
        } else if avg_satisfaction > avg_tension * 2.0 && avg_tension < field.settle_epsilon * 50.0 {
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

/// Update exploration goal progress op basis van echte activiteit.
/// Wordt aangeroepen vanuit de tick loop na exploration.
/// Update exploration goal progress op basis van echte activiteit.
/// v2.3: Progress schaalt relatief — meer frontier = meer nodig voor 100%.
pub fn update_exploration_progress(field: &mut Field, pages_read_delta: u64, tokens_delta: u64, frontier_size: usize) {
    // Progress per pagina schaalt invers met frontier grootte
    let pages_needed = (frontier_size as f32).max(10.0);
    // Progress per tokens schaalt met vocab grootte (hoe meer Lucy weet, hoe meer nodig)
    let token_scale = (field.active_set.len() as f32 * 10.0).max(1000.0);

    for goal in field.goals.iter_mut() {
        match &goal.strategy {
            GoalStrategy::Explore(_) => {
                let page_progress = pages_read_delta as f32 / pages_needed;
                let token_progress = tokens_delta as f32 / token_scale;
                let delta = (page_progress + token_progress).min(0.5);
                goal.progress = (goal.progress + delta).min(1.0);
            }
            _ => {}
        }
    }
}

/// Genereert een nieuw doel op basis van Lucy's interne staat.
/// Bronnen: curiosity, frustration, interesses, zelf-verbetering.
/// Nooit hardcoded — alles komt uit het veld en de explorer.
pub fn generate_goal(
    field: &mut Field,
    explorer: &Explorer,
    vocab: &Vocabulary,
) -> Option<String> {
    // v2.3: max_goals is dynamisch
    if field.goals.len() >= field.max_goals { return None; }

    let (avg_tension, avg_curiosity, avg_frustration, _avg_satisfaction) = observe_state(field);
    let mut rng = rand::rng();

    // === Intrinsic Curiosity Theory (Schmidhuber 2010, Oudeyer & Kaplan 2007) ===
    //
    // Curiosity = LEARNING PROGRESS: hoeveel daalde free energy recent?
    // Exploration = INFORMATION GAIN: hoe nieuw is de frontier?
    // Frustration en self-improve blijven veld-gedreven.

    // Learning progress: verschil tussen tension_ema (oud) en tension (nu), gemiddeld over active set.
    // Positieve learning progress = free energy daalt = we leren effectief.
    let n_active = field.active_set.len().max(1) as f32;
    let avg_learning_progress: f32 = field.active_set.iter()
        .map(|&id| {
            let p = &field.points[id as usize];
            (p.tension_ema - p.free_energy).max(0.0) // hoeveel free energy daalde
        })
        .sum::<f32>() / n_active;

    // Curiosity score = learning progress × interesse-diversiteit
    // Hoge learning progress + breed interesseveld = sterke drive om verder te leren
    let interest_diversity = explorer.interests.len() as f32 / (explorer.interests.len() as f32 + 10.0);
    let curiosity_score = avg_learning_progress * interest_diversity + avg_curiosity * 0.1;

    // Frustration score: frustratie × tension product (onveranderd — werkt goed)
    let frustration_score = avg_frustration * avg_tension;

    // Explore score: information gain van frontier.
    // Hoge frontier ratio = veel onbekend, PLUS gemiddelde novelty (rnd_score) van actieve punten
    // geeft een beeld van hoe "nieuw" de huidige representaties zijn.
    let frontier_ratio = explorer.frontier.len() as f32 / (explorer.frontier.len() as f32 + explorer.pages_read as f32 + 1.0);
    let avg_novelty: f32 = field.active_set.iter()
        .map(|&id| field.points[id as usize].rnd_score)
        .sum::<f32>() / n_active;
    let explore_score = frontier_ratio * (1.0 + avg_novelty);

    // Self-improve score: hoge tension relatief aan eigen gemiddelde = tijd om te optimaliseren
    let self_improve_score = avg_tension / (avg_tension + avg_curiosity + 0.001);

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
    // Intentie sterkte schaalt met noise_amplitude — zelfde schaal als veldactiviteit
    field.intention_strength = field.noise_amplitude;

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

        // Exploration goals: progress wordt extern bijgewerkt via update_exploration_progress()
        // Andere goals: progress op basis van velddynamica
        let is_explore = matches!(&goal.strategy, GoalStrategy::Explore(_));
        if !is_explore {
            let avg_alignment = alignments[i];
            let new_progress = (avg_alignment * 0.7 + avg_satisfaction * 0.3).min(1.0);
            if new_progress > goal.progress {
                goal.progress = goal.progress * 0.9 + new_progress * 0.1;
            }
        }

        // Compleet als progressie hoog genoeg
        if goal.progress > 0.8 {
            goal.progress = 1.0;
            completed_indices.push(i);
        }

        // v2.3: Abandon drempels relatief — geen vaste tick-aantallen
        // Age drempel = 5× reflect_interval (dynamisch)
        let max_age = field.reflect_interval * 5;
        if age > max_age && goal.progress < 0.1 {
            abandoned_indices.push(i);
        }

        // Frustratie-abandon: frustratie dominant, weinig progressie, oud genoeg
        // (niet voor explore — die heeft eigen progress tracking)
        if !is_explore && avg_frustration > avg_satisfaction * 5.0 && goal.progress < 0.2 && age > field.reflect_interval {
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
            field.intention_strength = field.noise_amplitude * top.priority;
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
///
/// Intrinsic motivation signals (Schmidhuber 2010, Oudeyer & Kaplan 2007):
/// - Explore: novelty bonus — hoe anders is de frontier dan wat Lucy al kent?
/// - Learn: prediction error — hoge error = veel te leren
/// - Think: emotie-variantie — verwarring/conflict → tijd om te reflecteren
pub fn decide_action(field: &Field, explorer: &Explorer) -> Action {
    let (avg_tension, avg_curiosity, avg_frustration, avg_satisfaction) = observe_state(field);

    let frontier_factor = explorer.frontier.len() as f32 / (explorer.frontier.len() as f32 + 10.0);
    let time_since_think = field.tick.saturating_sub(field.last_think_tick) as f32;
    let think_urgency = (time_since_think / field.think_interval as f32).min(1.0);
    let has_goals = !field.goals.is_empty();

    // === Novelty bonus voor Explore ===
    // Hoe "nieuw" zijn de huidige actieve representaties? Gemiddelde rnd_score.
    // Hoge novelty + grote frontier = sterke drive om te verkennen.
    let n_active = field.active_set.len().max(1) as f32;
    let avg_novelty: f32 = field.active_set.iter()
        .map(|&id| field.points[id as usize].rnd_score)
        .sum::<f32>() / n_active;
    let novelty_bonus = avg_novelty * frontier_factor;

    // Explore: curiosity × frontier + novelty bonus + bonus als geen doelen
    let score_explore = avg_curiosity * frontier_factor
        + novelty_bonus
        + if !has_goals { avg_curiosity.max(frontier_factor) } else { 0.0 };

    // === Learn: HIGH when prediction error is high (lots to learn) ===
    // avg_tension bevat al free energy, maar we willen specifiek de prediction error component.
    // avg_free_energy is een goede proxy; hoge free energy = veel prediction error = veel te leren.
    let avg_free_energy: f32 = field.active_set.iter()
        .map(|&id| field.points[id as usize].free_energy)
        .sum::<f32>() / n_active;
    let score_learn = avg_free_energy
        + avg_tension
        + if avg_satisfaction > avg_tension { avg_satisfaction } else { 0.0 };

    // === Think: HIGH when emotion variance is high (confusion → need to reflect) ===
    // Hoge variantie in emoties = intern conflict/verwarring → reflectie helpt.
    let emo_mean = (avg_curiosity + avg_frustration + avg_satisfaction) / 3.0;
    let emo_variance = ((avg_curiosity - emo_mean).powi(2)
        + (avg_frustration - emo_mean).powi(2)
        + (avg_satisfaction - emo_mean).powi(2)) / 3.0;
    let score_think = avg_frustration
        + think_urgency * avg_tension
        + emo_variance
        + if !has_goals { think_urgency } else { 0.0 };

    // Consolidate: satisfaction × rust (lage tension)
    let calm = (1.0 - avg_tension.min(1.0)).max(0.0);
    let score_consolidate = avg_satisfaction * calm;

    // Earn: baseline = laagste emotie × noise_amplitude (altijd > 0)
    let score_earn = field.noise_amplitude;

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

/// Moet Lucy nu nadenken? (dynamisch interval uit field)
pub fn should_think(field: &Field) -> bool {
    field.tick.saturating_sub(field.last_think_tick) >= field.think_interval
}

/// Moet Lucy nu reflecteren? (dynamisch interval uit field)
pub fn should_reflect(field: &Field) -> bool {
    field.tick.saturating_sub(field.last_reflect_tick) >= field.reflect_interval
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
