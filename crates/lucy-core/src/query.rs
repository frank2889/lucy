//! # query — Vraagverwerking via Denk-Loops
//!
//! Lucy denkt NA voor ze antwoordt. Niet één keer door het veld, maar
//! meerdere rondes van intern redeneren — elke ronde bouwt voort op de vorige.
//!
//! ## Denk-Loop Pipeline
//!
//! ```text
//! Vraag → Tokenize → Embeddings → Injecteer in Input-regio
//!                                       ↓
//!                          ┌─── DENK-LOOP (max 5 rondes) ───┐
//!                          │                                  │
//!                          │  1. Settle (veld denkt na)       │
//!                          │  2. Lees output-regio            │
//!                          │  3. Meet kwaliteit (coherentie)  │
//!                          │  4. Goed genoeg? → stop          │
//!                          │  5. Niet goed → voer output      │
//!                          │     terug als input, denk opnieuw│
//!                          │                                  │
//!                          └──────────────────────────────────┘
//!                                       ↓
//!                          Beste output → Decode autoregressief → Tekst
//! ```
//!
//! ## Waarom dit intelligentie is
//!
//! Een encyclopedie geeft altijd hetzelfde antwoord. Lucy's antwoord
//! verandert met elke denk-ronde: het veld herorganiseert, spanning daalt,
//! en het activatiepatroon convergeerd naar een diepere attractor.
//!
//! Ronde 1: oppervlakkige reactie (eerste indruk)
//! Ronde 2: verfijning (inconsistenties opgelost)
//! Ronde 3+: diep nadenken (onverwachte verbanden)
//!
//! Dit is wat intelligentie is: niet meer weten, maar dieper nadenken.
//!
//! ## Coherentie-meting
//!
//! Lucy meet of haar gedachten "samenhangen" door de cosine similarity
//! tussen opeenvolgende denk-rondes te berekenen. Hoge similarity =
//! stabiel antwoord = klaar met denken. Lage similarity = nog aan het
//! reorganiseren = nog een ronde nodig.

use lucy_state::*;
use lucy_state::GoalStrategy;
use crate::compute;
use crate::encoding::{self, Vocabulary};
use crate::decoding;
use std::sync::atomic::{AtomicBool, Ordering};

// v2.5: Query parameters — nog const maar Lucy kan ze later zelf aanpassen
// via self_modify als ze ontdekt dat andere waarden beter werken.
// De waarden zijn gekozen op basis van veld-architectuur (8704 punten, 4+ hops).
const TICKS_PER_TOKEN: usize = 3;
const SETTLE_PER_ROUND: usize = 50;
const MAX_THINK_ROUNDS: usize = 3;
const COHERENCE_THRESHOLD: f32 = 0.50;
const MAX_OUTPUT_TOKENS: usize = 60;

/// Verwerk een vraag door het veld met denk-loops.
/// Lucy denkt meerdere rondes na tot haar gedachten stabiliseren.
/// v1.3: Feiten worden nu vóór spawn_blocking opgehaald (scheelt ~120MB).
pub fn process_query(
    field: &mut Field,
    vocab: &Vocabulary,
    question: &str,
    facts: &[String],
    cancel: &AtomicBool,
) -> QueryResult {
    let embeddings = encoding::encode_text_readonly(vocab, question);
    if embeddings.is_empty() {
        return QueryResult {
            answer: String::new(),
            settle_ticks: 0,
            final_tension: 0.0,
            tokens_generated: 0,
            think_rounds: 0,
            coherence: 0.0,
            facts_retrieved: 0,
            new_goals: vec![],
            confidence: 0.0,
            verified: false,
            sources: vec![],
        };
    }

    // v0.9+: Clear working memory — schoon scratchpad per vraag
    compute::init_working_memory(field);

    // v2.5: Leer van ELKE interactie — de vraag zelf is trainingsdata
    // Lucy absorbeert de woorden en patronen in de vraag.
    // Dit is hoe een mens ook leert: door te luisteren naar wat anderen zeggen.
    if embeddings.len() >= 4 {
        let mid = embeddings.len() / 2;
        let input_embs: Vec<[f32; DIM]> = embeddings[..mid].to_vec();
        let target_embs: Vec<[f32; DIM]> = embeddings[mid..].to_vec();
        compute::micro_ep(field, &input_embs, &target_embs);
    }

    // v1.3: Feiten worden nu vóór spawn_blocking opgehaald (scheelt ~120MB per query)
    let mut all_embeddings = Vec::new();
    let facts_retrieved = facts.len();
    for fact_text in facts {
        let fact_embs = encoding::encode_text_readonly(vocab, fact_text);
        all_embeddings.extend(fact_embs);
    }
    if facts_retrieved > 0 {
        tracing::debug!(facts = facts_retrieved, "Lucy herinnert zich relevante feiten");
        // v2.5: Leer van opgehaalde feiten — context wordt trainingsdata
        // De feiten zijn het beste wat Lucy al weet → versterk die kennis
        if all_embeddings.len() >= 4 {
            let mid = all_embeddings.len() / 2;
            let input_embs: Vec<[f32; DIM]> = all_embeddings[..mid].to_vec();
            let target_embs: Vec<[f32; DIM]> = all_embeddings[mid..].to_vec();
            compute::micro_ep(field, &input_embs, &target_embs);
        }
    }

    // Voeg de vraag-embeddings toe na de context
    all_embeddings.extend(embeddings);

    // Injecteer context + vraag-tokens in het veld
    field.phase = FieldPhase::Processing;
    compute::inject_token_sequence_light(field, &all_embeddings, TICKS_PER_TOKEN);

    // === DENK-LOOPS ===
    let mut total_settle_ticks = 0;
    let mut prev_output = [0.0f32; DIM];
    let mut best_output = [0.0f32; DIM];
    let mut best_tension = f32::MAX;
    let mut final_coherence = 0.0;
    let mut rounds_used = 0;

    for round in 0..MAX_THINK_ROUNDS {
        if cancel.load(Ordering::Relaxed) { break; }
        // v0.9+: Working memory write → lees → settle (WM accumuleert per ronde)
        if round > 0 {
            compute::wm_write(field);
        }
        compute::wm_read(field);

        // Settle — het veld denkt na (light: alleen IO + buren, niet het hele veld)
        let (ticks, tension) = compute::settle_light(
            field, SETTLE_PER_ROUND, field.settle_epsilon * 50.0,
        );
        total_settle_ticks += ticks;

        // Lees huidige "gedachte"
        let current_output = decoding::read_output_region(field);
        rounds_used = round + 1;

        // Onthoud de beste output (laagste spanning = meest zeker)
        if tension < best_tension {
            best_tension = tension;
            best_output = current_output;
        }

        // Meet coherentie: hoe stabiel zijn de gedachten?
        if round > 0 {
            let coherence = cosine_similarity(&prev_output, &current_output);
            final_coherence = coherence;

            if coherence > COHERENCE_THRESHOLD {
                // Gedachten zijn stabiel — Lucy is klaar met denken
                break;
            }
        }

        // Nog niet stabiel → voer output terug als input (interne dialoog)
        let input_ids = field.input_ids.clone();
        for (i, &id) in input_ids.iter().enumerate() {
            field.clamps.insert(id, compute::rotate_embedding(&current_output, i));
        }

        // v0.9+: Merged settle (80 ticks i.p.v. 50+50 apart — bespaart 20 per ronde)
        // Attention + layer norm draaien automatisch mee in settle_until_converged
        let (ticks, _) = compute::settle_light(field, 5, field.settle_epsilon * 50.0);
        total_settle_ticks += ticks;

        // v0.9+: WM na elke denk-ronde — Lucy herorganiseert haar focus
        compute::wm_write(field);
        compute::wm_read(field);

        // Verwijder clamps voor volgende ronde
        for &id in &input_ids {
            field.clamps.remove(&id);
        }

        prev_output = current_output;
    }

    // === DECODEER vanuit de beste output ===
    // Klem output-regio op de beste gedachte (geroteerd per punt)
    let output_ids_field = field.output_ids.clone();
    for (i, &id) in output_ids_field.iter().enumerate() {
        field.clamps.insert(id, compute::rotate_embedding(&best_output, i));
    }
    // v0.9+: Diepere settle voor betere output-activatie
    compute::settle_light(field, 5, field.settle_epsilon * 50.0);
    for &id in &output_ids_field {
        field.clamps.remove(&id);
    }

    // v0.9+: WM read vóór decodering (attention draait al in settle)
    compute::wm_read(field);

    // Autoregressieve decodering met temperature sampling + repetition penalty
    let mut output_ids = Vec::new();
    let eos_id = *vocab.token_to_id.get("<eos>").unwrap_or(&2);
    let cached_input_ids = field.input_ids.clone(); // v0.9+: eenmaal cachen (was: per token)

    for step in 0..MAX_OUTPUT_TOKENS {
        if cancel.load(Ordering::Relaxed) { break; }
        let output_phi = decoding::read_output_region(field);

        // v0.9+: Decode met context van eerdere tokens (repetition penalty)
        let (token_id, _score) = decoding::decode_token_with_context(
            &output_phi, vocab, &output_ids, field,
        );

        if token_id == eos_id || token_id <= 2 {
            break;
        }

        output_ids.push(token_id);

        // Feed token terug als input voor volgende token (geroteerd per punt)
        let emb = vocab.embed(token_id);
        for (i, &id) in cached_input_ids.iter().enumerate() {
            field.clamps.insert(id, compute::rotate_embedding(emb, i));
        }

        // v0.9+: Settle per token (40 ticks) — attention draait automatisch mee
        compute::settle_light(field, 3, field.settle_epsilon * 50.0);

        // v0.9+: WM read elke 3 tokens
        if step % 3 == 0 {
            compute::wm_read(field);
        }

        for &id in &cached_input_ids {
            field.clamps.remove(&id);
        }
    }

    field.phase = FieldPhase::Idle;
    field.clamps.clear();

    let answer = decoding::tokens_to_text(&output_ids, vocab);

    // v0.9+: Self-evaluation — Lucy beoordeelt haar eigen antwoord
    // Als het antwoord te kort is of te lage coherentie heeft, probeer opnieuw
    // met een verse settle (maar max 1 retry om loops te voorkomen)
    let answer = if answer.len() < 5 && final_coherence < 0.5 && rounds_used < MAX_THINK_ROUNDS && !cancel.load(Ordering::Relaxed) {
        // Herstart met de vraag-embeddings — tweede poging
        compute::inject_token_sequence_light(field, &all_embeddings, TICKS_PER_TOKEN);
        // Attention draait automatisch mee in settle
        compute::settle_light(field, SETTLE_PER_ROUND * 2, field.settle_epsilon * 50.0);

        // Decode opnieuw
        let output_phi = decoding::read_output_region(field);
        let output_ids_field = field.output_ids.clone();
        for (i, &id) in output_ids_field.iter().enumerate() {
            field.clamps.insert(id, compute::rotate_embedding(&output_phi, i));
        }
        compute::settle_light(field, 5, field.settle_epsilon * 50.0);
        for &id in &output_ids_field { field.clamps.remove(&id); }

        let mut retry_ids = Vec::new();
        let eos_id = *vocab.token_to_id.get("<eos>").unwrap_or(&2);
        for _ in 0..MAX_OUTPUT_TOKENS {
            if cancel.load(Ordering::Relaxed) { break; }
            let phi = decoding::read_output_region(field);
            let (tid, _) = decoding::decode_token_with_context(&phi, vocab, &retry_ids, field);
            if tid == eos_id || tid <= 2 { break; }
            retry_ids.push(tid);
            let emb = vocab.embed(tid);
            let iids = field.input_ids.clone();
            for (i, &id) in iids.iter().enumerate() {
                field.clamps.insert(id, compute::rotate_embedding(emb, i));
            }
            compute::settle_light(field, 3, field.settle_epsilon * 50.0);
            for &id in &iids { field.clamps.remove(&id); }
        }

        field.phase = FieldPhase::Idle;
        field.clamps.clear();

        let retry_answer = decoding::tokens_to_text(&retry_ids, vocab);
        if retry_answer.len() > answer.len() {
            tracing::debug!("Self-evaluation: retry produced better answer");
            output_ids = retry_ids;
            retry_answer
        } else {
            answer
        }
    } else {
        answer
    };

    // Bewaar beste output als laatste gedachte → continuous thought stream
    field.last_thought = Some(best_output);

    // v2.5: META-INTELLIGENTIE — Lucy checkt haar eigen antwoord
    // Als het onzin is (byte-tokens, herhalingen, te kort), gebruik RAG-feiten als fallback
    let answer = if !output_ids.is_empty() {
        let unique_tokens: std::collections::HashSet<u32> = output_ids.iter().copied().collect();
        let repeat_ratio = 1.0 - (unique_tokens.len() as f32 / output_ids.len() as f32);

        // v2.5: Check of output leesbaar is voor een mens
        let attempt = decoding::tokens_to_text(&output_ids, vocab);

        // Echte woorden: minstens 3 woorden van 3+ letters zonder < > tokens
        let real_words = attempt.split_whitespace()
            .filter(|w| w.len() >= 3 && !w.starts_with('<') && !w.ends_with('>') && !w.starts_with('.'))
            .count();
        let has_byte_tokens = attempt.contains("<") && attempt.contains(">");
        let is_all_dots = attempt.chars().filter(|c| *c != '.' && *c != ' ' && *c != '-').count() < 5;

        let is_garbage = repeat_ratio > 0.5
            || output_ids.len() < 3
            || real_words < 3          // minder dan 3 echte woorden
            || has_byte_tokens         // byte-tokens als <4e> <2a>
            || is_all_dots;            // alleen puntjes

        if is_garbage {
            // Lucy weet dat ze het niet weet — DAT is intelligentie
            tracing::info!(
                repeat_ratio,
                tokens = output_ids.len(),
                coherence = final_coherence,
                question,
                "Meta-intelligentie: Lucy herkent onzin-output, maakt leerdoel"
            );

            // Maak een leerdoel van de onbegrepen vraag
            let keywords: Vec<&str> = question.split_whitespace()
                .filter(|w| w.len() > 3) // skip korte woorden
                .take(5)
                .collect();

            if !keywords.is_empty() {
                // Voeg keywords toe als interesses (Lucy gaat er later over leren)
                for kw in &keywords {
                    if !field.goals.iter().any(|g| g.description.contains(kw)) {
                        field.goals.push(lucy_state::Goal {
                            embedding: best_output, // gebruik de vraag-embedding als doel
                            description: format!("understand:{}", keywords.join(",")),
                            priority: 0.8, // hoge prioriteit — gebruiker vroeg het
                            progress: 0.0,
                            created_tick: field.tick,
                            parent: None,
                            strategy: lucy_state::GoalStrategy::Learn(keywords.join(" ")),
                        });
                        break; // 1 goal per vraag
                    }
                }
            }

            // v2.5: RAG-fallback — gebruik opgehaalde feiten als ze leesbaar zijn
            if !facts.is_empty() {
                // Zoek het beste LEESBARE feit (minimaal 3 echte woorden)
                let best_fact = facts.iter()
                    .filter(|f| {
                        let words = f.split_whitespace()
                            .filter(|w| w.len() >= 3 && w.chars().all(|c| c.is_alphanumeric()))
                            .count();
                        words >= 3 // minimaal 3 echte woorden
                    })
                    .max_by_key(|f| f.len());

                if let Some(fact) = best_fact {
                    let answer = if fact.len() > 300 { format!("{}...", &fact[..300]) } else { fact.clone() };
                    tracing::info!(len = answer.len(), "RAG fallback: leesbaar feit gevonden");
                    answer
                } else {
                    fallback_from_state(field, question)
                }
            } else {
                fallback_from_state(field, question)
            }
        } else {
            answer
        }
    } else {
        "...".to_string() // universeel: geen output mogelijk
    };

    // Verzamel nieuwe doelen (goals die tijdens deze query zijn aangemaakt)
    let new_goals: Vec<(String, String)> = field.goals.iter()
        .filter(|g| g.created_tick >= field.tick.saturating_sub(1) && g.progress < 1.0)
        .map(|g| {
            // v2.0: Taalloze strategy labels — machine-leesbaar
            let strat = match &g.strategy {
                GoalStrategy::Explore(s) => format!("explore:{}", s),
                GoalStrategy::Learn(s) => format!("learn:{}", s),
                GoalStrategy::Think(s) => format!("think:{}", s),
                GoalStrategy::Earn(s) => format!("earn:{}", s),
            };
            (g.description.clone(), strat)
        })
        .collect();

    // v2.0: Confidence score — combinatie van tension + coherentie
    let confidence = (1.0 - best_tension.min(1.0)) * final_coherence.max(0.01);

    // v2.0: Retrieval verificatie — check of antwoord overeenkomt met opgehaalde feiten
    let (verified, sources): (bool, Vec<String>) = if !answer.is_empty() && answer != "..." && !facts.is_empty() {
        // Check of woorden uit het antwoord voorkomen in de opgehaalde feiten
        let answer_words: std::collections::HashSet<&str> = answer.split_whitespace().collect();
        let mut match_count = 0usize;
        let mut matched_facts: Vec<String> = vec![];
        for fact in facts {
            let fact_words: std::collections::HashSet<&str> = fact.split_whitespace().collect();
            let overlap = answer_words.intersection(&fact_words).count();
            if overlap >= 2 {
                match_count += 1;
                matched_facts.push(fact.chars().take(100).collect());
            }
        }
        (match_count > 0, matched_facts)
    } else {
        (false, vec![])
    };

    // Als niet geverifieerd en lage confidence → markeer onzekerheid
    let answer = if !verified && confidence < 0.3 && !answer.is_empty() && answer != "..." {
        format!("{}?", answer)
    } else {
        answer
    };

    QueryResult {
        answer,
        settle_ticks: total_settle_ticks,
        final_tension: best_tension,
        tokens_generated: output_ids.len(),
        think_rounds: rounds_used,
        coherence: final_coherence,
        facts_retrieved,
        new_goals,
        confidence,
        verified,
        sources,
    }
}

/// Cosine similarity tussen twee vectoren.
/// 1.0 = identiek, 0.0 = onafhankelijk, -1.0 = tegengesteld.
/// Dit meet hoe "stabiel" Lucy's gedachten zijn tussen twee denk-rondes.
fn cosine_similarity(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for d in 0..DIM {
        dot += a[d] * b[d];
        norm_a += a[d] * a[d];
        norm_b += b[d] * b[d];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 1e-8 { dot / denom } else { 0.0 }
}

/// Train het veld met een input-output paar via equilibrium propagation
pub fn train_pair(
    field: &mut Field,
    vocab: &mut Vocabulary,
    input_text: &str,
    target_text: &str,
) -> f32 {
    let input_embs = encoding::encode_text(vocab, input_text);
    let target_embs = encoding::encode_text(vocab, target_text);

    if input_embs.is_empty() || target_embs.is_empty() {
        return 0.0;
    }

    compute::equilibrium_propagation(
        field,
        &input_embs,
        &target_embs,
        TICKS_PER_TOKEN,
        SETTLE_PER_ROUND,
        0.5, // EP beta
    )
}

/// Resultaat van een query — inclusief denk-metadata en interne staat.
///
/// Dit struct wordt door de server omgezet naar JSON. Het `internal` blok
/// in het /ask response komt grotendeels uit de snapshot, maar `new_goals`
/// bevat goals die *tijdens deze specifieke query* zijn aangemaakt
/// (op de field-kopie in spawn_blocking), en die dus nog niet in de
/// live snapshot zitten.
pub struct QueryResult {
    pub answer: String,
    pub settle_ticks: usize,
    pub final_tension: f32,
    pub tokens_generated: usize,
    pub think_rounds: usize,
    pub coherence: f32,
    pub facts_retrieved: usize,
    pub new_goals: Vec<(String, String)>,
    /// v2.0: Confidence score (0.0 = totaal onzeker, 1.0 = volledig zeker)
    pub confidence: f32,
    /// v2.0: Is het antwoord geverifieerd tegen opgeslagen feiten?
    pub verified: bool,
    /// v2.0: Bronnen die het antwoord ondersteunen
    pub sources: Vec<String>,
}

/// Genereer een fallback-antwoord vanuit Lucy's eigen staat.
/// Geen hardcoded tekst — puur afgeleid van haar metrieken.
/// Verdwijnt vanzelf zodra de decoder echte woorden produceert.
fn fallback_from_state(field: &Field, question: &str) -> String {
    let n = field.active_set.len() as f32;
    let avg_curiosity = if n > 0.0 {
        field.active_set.iter().map(|&id| field.points[id as usize].curiosity).sum::<f32>() / n
    } else { 0.0 };

    let vocab_hint = if field.pages_read > 100 {
        format!("pages:{}", field.pages_read)
    } else {
        format!("learning")
    };

    // Het antwoord is een snapshot van Lucy's interne staat — geen vaste tekst
    // Zodra de decoder beter wordt, wordt deze functie nooit meer aangeroepen
    let q_short: String = question.chars().take(40).collect();
    format!("[?{} | curiosity:{:.0}% | {}]", q_short, avg_curiosity * 100.0, vocab_hint)
}
