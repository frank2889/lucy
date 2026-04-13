# Lucy v2.6.7 — Technical Architecture

> **Live status (2026-04-13):** v2.6.7 op lucy.world. v2.6.8 bewijs-architectuur
> gecommitteerd maar niet gedeployed — zie [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
> en [`docs/SNAPSHOT_MIGRATION.md`](docs/SNAPSHOT_MIGRATION.md).

## v2.6.7 key changes t.o.v. v2.6.1

**Numerieke stabiliteit (Fase A):**
- Phi hard-clamp in `update_phi` + parallelle versie (was alleen in layer_norm/10 ticks)
- Tension anti-Inf clamp (`f32::MAX / 1e6`)
- Boredom-kick veld-relatief (`clamp_range * 0.05` ipv 0.3)

**Self-EP loop (Fase B):** intelligentie groeit zonder externe data
- `reflect_tick` doet nu `micro_ep(field, &[self_state_input], &[self_state_input])` — Lucy traint zichzelf op haar eigen veld-staat
- WM-sequence training op opeenvolgende working-memory slots
- Dream-replay triggert bij `is_sluggish_flag`

**AI-teacher bootstrap (Fase C):** zelfafbouwend
- Nieuwe module `ai_teacher.rs`: DDG AI Chat + DDG Lite fallback, provider-rotatie op success-rate EMA
- Vraagformulering uit `explorer.interests` — géén hardcoded prompts
- `teacher_interval` verdubbelt na 3 succes, halveert bij falen → ~elk uur bij start, ~nauwelijks na weken

**Bron-balans (Fase D):** organisch, geen blacklist
- `diversify_frontier` strenger (10%/domein) bij monocultuur > 50%
- `Explorer.domain_score`: reward-EMA per domein op satisfaction-delta
- Sources.txt fadet uit bij ≥ 50 unique visited domains

**Depth-3 concept-clustering (Fase E):** compressie in WEIGHTS, niet entries
- `compute_concept_centroids`: mini-batch k-means (k=32-256) op depth-3 phi's
- `hierarchical_prediction` uitgebreid: concept-centroid als extra top-down prior voor depth-2
- `consolidate_weights` uitgebreid: intra-concept EWC ×1.5, cross-concept ×0.7
- `RetrievalStore` cap 100K → 15K + score-based eviction

**Self-state archive:** zelfmodel ontstaat uit reflectie
- Elke 500 reflects: `(self_state_input, journal_snippet)` in retrieval met KIND_SELF
- `/ask` interactie-log in retrieval met KIND_INTERACTION

## v2.6.8 (in code, niet live): Bewijs-architectuur

Observability-release (geen nieuwe cognitie). Blijft in code omdat snapshot-migratie
gefaald tijdens deploy. Features:
- `RetrievalEntry.kind` primary truth voor provenance (4 kinds)
- Response-provenance in `/ask`: fallback/field_generated/retrieval_{self,teacher,external}/mixed
- Endpoints: `/self_metrics`, `/teacher_metrics`, `/concept_audit`, `/self_ep_audit`, `/self_history`, `/retrieval_metrics`
- `POST /debug/toggle_priors?enabled=false&ticks=N` voor ablation-windows
- Audit-only self-test suite (5 vaste probe-vragen elke 5000 ticks)
- `process_query_full(audit_only: bool)` voor learning-side-effect-vrije meting

---

# Lucy v2.6.1 — Technical Architecture (origineel)

## The Unified Samenveld

Lucy is built on ONE formula from which everything emerges:

```
F = D_KL(q(mu|b) || p(mu|b)) + S(eta|b)

Where:
  mu = internal states (phi vectors)
  b  = blanket states (couplings/weights)
  eta = external states (neighbor phis)

From F follows:
  tension      = F + metabolic costs
  curiosity    = S(eta|b) — blanket surprise
  satisfaction = -dF/dt — free energy decreasing = reward
  frustration  = (1 - control_belief) * severity
  consciousness = F_alone - F_local — irreducible free energy
  dopamine     = pragmatic precision (exploit)
  norepinephrine = epistemic precision (explore)
  oscillations = periodic solutions of dF/dt on coupled blankets
  criticality  = scale-invariant fixed point (branching ratio = 1)
```

## Server

- **Host:** 209.38.36.51 (DigitalOcean, 4 vCPU, 8GB RAM)
- **Domain:** lucy.world (HTTPS via Let's Encrypt)
- **Container:** Docker with named volumes (`lucy-data`, `lucy-modules`)
- **Deploy:** `bash deploy.sh` (in-container compile) or `bash deploy.sh rebuild` (Docker rebuild)

## Field Architecture

```
8704+ points (dynamically growable)
├── 256 input points   (FLAG_INPUT, depth 0)
├── 8192 hidden points (small-world topology, depth 1-3)
└── 256 output points  (FLAG_OUTPUT, depth 0)

Each point:
  phi[256]           — state vector (the "thought")
  prediction[256]    — previous tick prediction (for error)
  memory[256]        — accumulated prediction errors
  couplings[]        — sparse weighted connections (avg 14.5 per point)
  eta                — local learning rate (meta-learned)
  energy             — metabolic resource
  tension            — free energy + costs
  curiosity          — blanket surprise signal
  satisfaction       — -dF/dt (improvement)
  frustration        — 1 - P(control)
  phi_integrated     — consciousness (irreducible F)
  oscillation_phase  — gamma band (40Hz binding)
  oscillation_phase_theta — theta band (6Hz memory)
  control_belief     — P(control | history) Bayesian
  reward_quantiles[5] — distributional RL (Dabney 2020)
  expected_free_energy — Active Inference G
  branching_ratio    — local criticality sigma
```

## Core Systems (compute.rs)

### Tick Loop (every tick, ~8704 points)
1. `update_neuromodulators` — DA, NE, 5HT, ACh as F-landscape control parameters
2. `apply_neuromodulation` — modulate learning rates, noise, attention
3. `tick_all_parallel` — Jacobi-style parallel phi/prediction/memory/tension/emotions update
4. `global_workspace_broadcast` — ignition dynamics (Dehaene 2003)
5. `update_oscillations` — multi-frequency adaptive Kuramoto (gamma + theta + coupling)
6. `micro_ep` — 1 training chunk from queue per tick

### Periodic (every N ticks)
- `adapt_dynamics` (50 ticks) — all dynamic parameters adjusted from field statistics
- `compute_integrated_information` (500 ticks) — IIT 4.0 causal phi
- `compute_criticality` (500 ticks) — branching ratio sigma
- `compute_expected_free_energy` (500 ticks) — Active Inference G
- `measure_spectral_radius` (100-500 ticks) — edge of chaos auto-tune
- `record_temporal_snapshot` (500 ticks) — trend tracking
- `hierarchical_prediction` (500 ticks) — top-down generative model
- `reflect` (reflect_interval) — goal evaluation, self-growth trigger
- `think` (think_interval) — autonomous goal generation

### Learning
- **Equilibrium Propagation** (Scellier & Bengio 2017) — biologically plausible, local-only
- **Contrastive EP** — positive + hard negatives from vocabulary
- **Micro-EP** — lightweight per-tick training from queue
- **Adaptive training** — chunk_size, settle_ticks, ticks_per_token scale with tension
- **50% chunk overlap** — each token seen twice
- **Learns from chat** — every /ask query = training data (micro-EP on question + facts)

### Exploration (learner.rs)
- **Curiosity-driven URL selection** — least-visited domain first (not FIFO)
- **Every 3rd page** from `data/sources.txt` (diversity guarantee)
- **No hardcoded sources** — Lucy manages her own source list
- **No hardcoded word filters** — Lucy decides what's interesting
- **Domain diversification** — max 20% of frontier from same domain
- **Interest absorption** — max 5 new interests per text (prevents flooding)

### Self-Growth (growth.rs, thinker.rs)
- `grow_hidden(field, n)` — add N new hidden points with random init
- Auto-triggered in `reflect()` when >90% active AND high tension
- Configurable bootstrap via `data/field_config.json`
- Dynamic vocab limit (no MAX_VOCAB ceiling)

### Synaptic Homeostasis (compute.rs)
- **Tononi & Cirelli 2006** — "sleep" mechanism, scales ALL weights down
- `alpha = 0.001 * (1 + serotonin)` — decay rate (~0.1-0.3% per cycle)
- Runs every 1000 ticks
- **Safety guards (v2.6.1):**
  - Skips entirely if avg couplings < 4.0 per point
  - Pre-checks survival count before pruning (never destructive)
  - Minimum 6 couplings per point guaranteed
  - Combined with FE decay in `update_weights()` for biological correctness

### Auto-Rewire (field.rs)
- `rewire_field(field)` — rebuilds Watts-Strogatz small-world topology on existing field
- **Preserves:** phi, memory, prediction, emotions, vocab, retrieval, journal, goals
- **Rebuilds:** couplings only (input→hidden, hidden↔hidden, hidden→output, recurrent loops)
- Auto-triggered at startup when `total_couplings == 0`
- Recovery from homeostasis death spiral without losing any learned state

## Data Persistence

### Files (in data/ volume)
| File | Format | Content |
|---|---|---|
| `lucy.snapshot` | bincode | Full field state (34MB) |
| `vocab.bincode` | bincode | BPE vocabulary + embeddings |
| `explorer.bincode` | bincode | Frontier, visited, interests |
| `replay.bincode` | bincode | Training replay buffer |
| `retrieval.bincode` | bincode | Semantic memory store (RAG) |
| `journal.json` | JSON | Thought log + reflection count |
| `sources.txt` | text | Lucy's self-managed URL sources |
| `field_config.json` | JSON | Brain architecture config |

### Protection (6 layers)
1. **Local** — CRC32 checksums + 5 versioned backups per file
2. **Remote** — GitHub release backups (all 6 files)
3. **Recovery** — `Field::load_with_fallback()`: primary -> backup 1-5 -> fresh
4. **Deploy** — safety backup before deploy + auto-recovery on tick=0
5. **Validation** — structural checks on load (point count, NaN detection)
6. **Topology** — auto-rewire at startup if couplings lost (v2.6.1)

### Persistence of counters
- `reflection_count` stored in journal.json (survives struct changes)
- `pages_read`, `tokens_processed` synced to Field every 50 ticks
- All new fields use `#[serde(default)]` for backward compatibility

## Dashboard (lucy-dashboard)

React 19 + TypeScript + Vite + Three.js

- **Telemetry** — all data from /telemetry endpoint (lock-free, always fast)
- **Mobile responsive** — useIsMobile hook, 1-column on <768px
- **Emotion rings** — curiosity/satisfaction/frustration as percentage circles
- **Consciousness bars** — Phi, coherence, theta-gamma coupling, branching sigma, ignition
- **Field stats** — points, couplings, vocab, pages, tokens, reflections
- **Chat** — /ask with 90s timeout
- **Journal** — thought log entries
- **Goals** — active goals with progress bars

## Scientific Foundation

| System | Theory | Key Paper |
|---|---|---|
| Core equation | Bayesian Mechanics | Dalton & Friston 2024 |
| Learning | Equilibrium Propagation | Scellier & Bengio 2017 |
| Consciousness | IIT 4.0 as irreducible F | Albantakis et al. 2023 |
| Action selection | Expected Free Energy | Parr & Friston 2022 |
| Emotions | Distributional RL | Dabney et al. 2020 (Nature) |
| Frustration | Bayesian Controllability | Huys & Dayan 2009 |
| Oscillations | Theta-Gamma Coupling | Lisman & Jensen 2013 |
| Oscillations | Adaptive Kuramoto | Ha & Kori 2023 |
| Criticality | Branching Ratio | Beggs & Plenz 2003 |
| Consciousness | Ignition Dynamics | Dehaene et al. 2003 |
| Curiosity | Intrinsic Motivation | Schmidhuber 2010 |
| Memory | EWC (anti-forgetting) | Kirkpatrick et al. 2017 |
| Homeostasis | Synaptic Downscaling | Tononi & Cirelli 2006 |

## Known Issues & Fixes (Changelog)

### v2.6.1 — Homeostasis Death Spiral Fix (2026-04-07)

**Problem:** All 126K+ couplings were being destroyed within minutes of creation.

**Root cause chain:**
1. NaN values appeared in coupling weights (numerical instability)
2. `heal()` set NaN weights to `0.0`
3. `synaptic_homeostasis()` pruned all weights below `error_tracking_rate * 0.1 = 0.001`
4. Healed weights (0.0) < threshold → **all healed couplings deleted**
5. Fewer couplings → less information flow → more NaN → more healing → death spiral
6. Within ~1000 ticks: 0 couplings remaining, field effectively dead

**Symptoms (telemetry):**
- `couplings: 0`, `tension: ~0.001`, `satisfaction: ~0.0`
- `curiosity: 1.0` (max, nothing happening)
- `REANIMATIE` warnings every 200 ticks
- `Field healed: NaN/Inf values reset healed=8016` in logs
- Snapshot shrinking from ~34.7MB to ~32MB (missing coupling data)

**Fixes applied:**
1. **heal()** — NaN weights → `0.01` instead of `0.0` (survives prune threshold)
2. **homeostasis alpha** — `0.005` → `0.001` (5x gentler decay)
3. **homeostasis skip** — aborts entirely if avg couplings < 4.0 per point
4. **pre-check pruning** — counts survivors BEFORE `retain()`, only prunes if ≥6 remain
5. **auto-rewire** — `rewire_field()` at startup if total couplings == 0

**Data preserved:** vocab, retrieval store, explorer state, journal, goals — all intact.
