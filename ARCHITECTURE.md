# Lucy v2.5 — Technical Architecture

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

### Protection (5 layers)
1. **Local** — CRC32 checksums + 5 versioned backups per file
2. **Remote** — GitHub release backups (all 6 files)
3. **Recovery** — `Field::load_with_fallback()`: primary -> backup 1-5 -> fresh
4. **Deploy** — safety backup before deploy + auto-recovery on tick=0
5. **Validation** — structural checks on load (point count, NaN detection)

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
