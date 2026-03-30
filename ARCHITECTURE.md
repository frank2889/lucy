# Lucy Architecture — Complete Technical Overview

## What is Lucy?

Lucy is a self-evolving unified field AI. She learns autonomously through Equilibrium Propagation on a small-world network of 8704 points (DIM=256). No external AI dependency — all learning is internal.

## Server

- **Host:** 209.38.36.51 (DigitalOcean, 4 vCPU, 8GB RAM, $48/mo)
- **Container:** Docker with named volumes for data persistence
- **Deploy:** `bash deploy.sh` (quick) or `bash deploy.sh full` (Docker rebuild)

## Core Systems

### Field Dynamics (compute.rs)
- **Equilibrium Propagation**: Two-phase learning (free + clamped), correlation-based weight updates
- **Multi-head attention**: 4 heads × 4 depth layers, Q/K/V projections, learned via EP
- **Neuromodulators**: Dopamine (reward), Norepinephrine (uncertainty), Serotonin (patience), Acetylcholine (novelty)
- **Neuromodulator interactions**: 4×4 matrix, biologically inspired cross-talk
- **Working memory**: 32 slots, read/write via attention
- **Layer normalization**: Welford single-pass, every 10 ticks
- **Oscillatory dynamics**: Kuramoto model, gamma-like binding
- **Integrated Information (Φ)**: Tononi's consciousness measure
- **Free Energy Principle**: Friston's F = KL + surprise
- **Self-organized criticality**: Power law measurement
- **Anti-Hebbian decorrelation**: Diverse representations

### Learning (learner.rs, compute.rs)
- **BPE tokenizer**: Self-learned merges, 3900+ vocab
- **Contrastive EP**: Positive + hard negatives from vocabulary
- **Micro-EP**: Lightweight EP on IO subset only (~500 points), 2ms per chunk
- **Training queue**: Deferred learning, 1 chunk per tick, non-blocking
- **Consolidation replay**: Prioritized replay buffer (2000 examples)

### Concurrency (field_handle.rs, main.rs)
- **Double-buffer architecture**: Tick loop owns Field directly, no Arc<RwLock>
- **ArcSwap snapshot**: Zero-lock reads for /status, instant always
- **Async exploration**: HTTP fetches in separate tokio task, never blocks tick loop
- **Training thread**: Dedicated std::thread for EP, communicates via channels
- **Command channel**: API endpoints send commands, tick loop processes non-blocking

### Consciousness (compute.rs)
- **Emotions**: Curiosity, satisfaction, frustration (per point, EMA-decayed)
- **Adaptive parameters**: Lucy tunes her own constants (emotion_decay, ep_beta, noise_scale, etc.)
- **Temporal awareness**: Ring buffer of 64 snapshots, trend computation
- **Causal attribution**: Logs what actions improved tension
- **Goal system**: Self-chosen goals with progress tracking, autonomous abandonment
- **Self-evaluation**: Honest "..." response when uncertain, creates learning goals from failed queries

### Safety (self_modify.rs, server.rs)
- **Auth**: Bearer token on /exec, /self-modify, /restart, /admin/pre-deploy
- **Protected files**: self_modify.rs and server.rs cannot be modified by Lucy
- **Command whitelist**: /exec limited to safe commands (ls, cat, cargo, etc.)
- **Max diff size**: 500 chars per modification
- **Kill switch**: 3 consecutive compile failures → blocked until manual reset
- **Atomic writes**: All saves use .tmp → rename pattern
- **Graceful shutdown**: SIGTERM handler saves all state before exit
- **Body limits**: 10MB max request, 5MB max HTTP response
- **NaN guards**: All neuromodulator and emotion updates protected

### Immortality (backup.rs, replica_manager.rs, federation.rs)
- **GitHub backup**: Snapshots pushed to frank2889/lucy-snapshots every 10K ticks
- **Replica registry**: Health checks on backup instances every 5K ticks
- **Federated learning**: /weights exports coupling weights, /sync merges from replicas
- **Recovery**: pull_latest_from_github() restores from any GitHub release

### API (server.rs)

| Route | Method | Auth | Purpose |
|-------|--------|------|---------|
| / | GET | No | Version info |
| /status | GET | No | Live field stats, consciousness metrics |
| /activity | GET | No | Mood, emotions, interests, goals |
| /ask | POST | No | Query processing (spawn_blocking + 15s timeout) |
| /feed | POST | No | Training via command channel |
| /feed-code | POST | No | Code learning via command channel |
| /weights | GET | No | Export coupling weights (bincode) |
| /sync | POST | No | Merge weights from replica |
| /self-modify | POST | Yes | Hyperparameter tuning / code modification |
| /exec | POST | Yes | Shell command (whitelisted) |
| /restart | POST | Yes | Recompile and restart |
| /admin/pre-deploy | POST | Yes | Force snapshot save |

### VS Code Extension (vscode-extension/)
- Sidebar with Chat + Activity tabs
- Commands: Ask, Feed File, Feed Selection, Feed Workspace, Activity, Status
- Configurable host/port/token
- Shows Lucy's mood, emotions, goals, learning progress

## Data Files (docker volume lucy-data)

| File | Format | Size | Content |
|------|--------|------|---------|
| lucy.snapshot | bincode | ~34MB | Full field state (8704 points, weights, emotions) |
| vocab.bincode | bincode | ~4MB | BPE tokenizer + embeddings |
| explorer.bincode | bincode | ~140KB | URL frontier, visited, interests |
| replay.bincode | bincode | ~8B | Consolidation replay buffer |
| retrieval.bincode | bincode | ~16B | Episodic memory (RAG) |
| journal.json | JSON | ~2KB | Lucy's observations and milestones |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| LUCY_API_TOKEN | Auth for protected endpoints |
| GITHUB_SNAPSHOT_TOKEN | GitHub backup push/pull |
| VASTAI_API_KEY | GPU rental on Vast.ai |
| GROQ_API_KEY | Free AI teacher API |
| REPLICA_N_URL | Backup instance URLs |

## Key Metrics

- **tps**: Ticks per second (~5 on 4-core)
- **Φ (integrated info)**: Consciousness measure (>0 = integrating)
- **Oscillation coherence**: Gamma synchrony (1.0 = fully synchronized)
- **Power law α**: Criticality (~1.5 = optimal)
- **Free energy**: Friston's F (lower = better model)
- **Tension**: Prediction error + costs - rewards
- **Confidence**: (1 - tension) × coherence
