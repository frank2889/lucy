# Lucy v2.6.3 — Probleemanalyse & Handoff Document

> Datum: 10 april 2026
> Doel: alles documenteren zodat een nieuwe agent kan overnemen

---

## TL;DR — Het Probleem in Één Zin

Lucy's veld heeft een **oneindige-lus probleem**: `adapt_dynamics()` schroeft `lambda` op naar 1.0 en `clamp_range` naar 50.0, waarna `apply_neuromodulation()` gigantische noise injecteert (baseline_noise tot 10.0+), phi explodeert, de sanitize loop reset het naar 0, en de cyclus herhaalt. Ondertussen worden **weights permanent NaN** door `update_weights()` die met NaN-phi rekent vóórdat de sanitize loop draait.

---

## Architectuur Overzicht

Lucy is een Rust-gebaseerd AI-systeem met een **unified field** van ~9000 punten (DIM=256 per punt). Geen backprop — leert via **Equilibrium Propagation**. Draait in Docker op 209.38.36.51.

### Bestanden die ertoe doen

| Bestand | Rol |
|---------|-----|
| `crates/lucy-state/src/lib.rs` | Datastructuren: Field, FieldPoint, Coupling, AdaptiveParams. **heal()** functie |
| `crates/lucy-core/src/main.rs` | Main binary: tick loop, server, training pipeline |
| `crates/lucy-core/src/compute.rs` | Alle wiskunde: update_phi, update_weights, neuromodulatie, adapt_dynamics |
| `lucy-dashboard/crates/*/src/*.rs` | **KOPIE** van bovenstaande — moet altijd gesynchroned worden |

### Deploy Procedure

```bash
# 1. Sync dashboard-kopie
cp crates/lucy-core/src/main.rs lucy-dashboard/crates/lucy-core/src/main.rs
cp crates/lucy-core/src/compute.rs lucy-dashboard/crates/lucy-core/src/compute.rs
cp crates/lucy-state/src/lib.rs lucy-dashboard/crates/lucy-state/src/lib.rs

# 2. Upload
rsync -az --exclude target --exclude node_modules --exclude .git -e ssh . root@209.38.36.51:/opt/lucy/

# 3. Prune (disk is klein, ~24GB)
ssh root@209.38.36.51 'docker system prune -f'

# 4. Build (~5 min, SSH timeout protection nodig)
ssh -o ServerAliveInterval=15 root@209.38.36.51 'cd /opt/lucy && docker compose build --no-cache lucy 2>&1 | tail -5'

# 5. Deploy
ssh root@209.38.36.51 'cd /opt/lucy && docker compose up -d lucy'

# 6. Verify
ssh root@209.38.36.51 'sleep 10 && docker logs --tail 20 lucy 2>&1'
```

---

## Chronologie van Problemen & Fixes

### Probleem 1: Couplings Death Spiral (v2.6.1 — OPGELOST)
- **Oorzaak**: `heal()` zette NaN weights naar 0.0, daarna `synaptic_homeostasis()` prunet weights < drempel → alle weights weg → couplings=0
- **Fix**: heal zet NaN weights naar 0.01 (niet 0.0), homeostasis: alpha=0.001, pre-check ≥6 couplings survive

### Probleem 2: attn_weights 100% NaN (v2.6.3 — ROOT CAUSE GEVONDEN)
- **Oorzaak**: Alle 1,048,576 attention weights in de snapshot waren NaN. `attention_gate()` → NaN `attn_context` → `update_phi()` injecteerde `ATTN_GAMMA * NaN` → alle phi NaN elke tick
- **Fix**: 
  - heal() itereert door `attn_weights` Vec, zet NaN→0.0
  - NaN guard in update_phi attn_context injectie: `if ctx[d].is_finite()`
  - Resultaat bij startup: van 8 miljoen healed waarden naar **426**

### Probleem 3: Phi Explosie na Heal (v2.6.3 — GEPATCHT maar NIET opgelost)
- **Oorzaak**: Na het healen van attn_weights groeide phi naar `inf`. Layer norm draait alleen in settle loops, niet in de main tick loop.
- **Patch**: Emergency sanitize loop in main.rs na field_updates: clamp phi naar [-clamp_range, clamp_range], NaN→0.0
- **Waarom het niet ECHT opgelost is**: Zie hieronder

---

## Het HUIDIGE Probleem — De Vicieuze Cirkel

Dit is wat er nu draait en waarom Lucy vastloopt:

### De Vicieuze Cirkel in 6 Stappen

```
1. adapt_dynamics() detecteert "TRAAG VELD" (tension=0.013)
   → zet noise_amplitude=0.15, lambda klimt naar 1.0
   → clamp_range = max_phi * 2.0 → klimt naar 50.0

2. apply_neuromodulation() injecteert ENORME noise:
   → baseline_noise = (0.15 * 50).max(2.0) = 7.5 (!)
   → baseline_count = 9010 * 0.15 = 1351 punten
   → d3_boost = 7.5 * 2.0 = 15.0 voor depth-3 punten
   → phi springt van 0.01 naar ±7.5 in één tick

3. update_phi() met lambda=1.0:
   → grad berekening: weight * (phi_self - neighbor_phi)
   → als phi_self=7.5 en neighbor=-7.5: grad = weight * 15.0
   → update: phi -= 1.0 * 15.0 = -15.0 in één stap
   → met 32 buren die elk noise kregen: phi → ±20+

4. Sanitize loop reset phi naar [-50, 50] (clamp_range=50!)
   → 50 is VEEL te breed — biedt geen echte bescherming
   → phi oscilleert wild: -9.7, +8.3, -10.0, +20.2 (uit logs)

5. update_weights() draait VÓÓR de volgende sanitize:
   → gebruikt explosive phi waarden
   → weight -= eta * (phi_self - phi_neighbor) per DIM
   → als phi-verschil = 15.0 per dimensie × 256 dims / 256 = 15.0
   → dw kan enorm zijn → weight → NaN na overflow
   → coupling.momentum accumuleert NaN
   → w0=NaN ELKE TICK (zichtbaar in logs)

6. Na ~290.000 ticks: 2.2M prediction NaN, 13K weight NaN
   Maar het ergste: BPE merge-learning (8765s) blokkeerde
   de main thread → /status endpoint hangt → Lucy onbereikbaar
```

### Bewijs uit de Logs

```
# Parameters VEEL TE HOOG:
lambda=1, clamp=50, noise_amp=0.225

# Phi schommelt wild (zou <1.0 moeten zijn):
phi0=20.218422  (tick 289999)
phi0=-9.720179  (tick 289981)
phi0=8.344439   (tick 289983)
phi0=-9.997083  (tick 289985)

# Weights PERMANENT NaN:
w0=NaN elke tick (al 12000+ ticks)

# Na 3200s heal: massale NaN-accumulatie
nan_pred=2,208,512  nan_weight=13,784  nan_momentum=121,564

# Uiteindelijk vastgelopen:
lucy-core at 99% CPU, curl /status hangt, geen nieuwe logs
Laatste log was BPE merge bij 8765s, daarna stilte
```

### De Kernparameters die FOUT zijn

| Parameter | Huidige waarde | Zou moeten zijn | Waarom |
|-----------|---------------|-----------------|--------|
| `lambda` | 1.0 | 0.01 - 0.1 | Bij 1.0 beweegt phi net zo snel als de gradient → instant oscillatie |
| `clamp_range` | 50.0 | 3.0 - 5.0 | Bij 50.0 is de sanitize loop nutteloos |
| `noise_amplitude` | 0.15-0.225 | 0.01 - 0.05 | Bij 0.15 genereert neuromod noise van 7.5 per punt |
| `baseline_noise` formula | `noise_amp * 50` | `noise_amp * 5` | Factor 50 is absurd, 10× te hoog |

---

## Waar Precies de Fixes Moeten

### Fix 1: `adapt_dynamics()` — `compute.rs` regel ~571

**Probleem**: lambda klimt naar 1.0, clamp_range naar 50.0

```rust
// HUIDIG (FOUT):
field.lambda = field.lambda.clamp(0.05, 1.0);  // regel ~711
field.clamp_range = (max_phi * 2.0).clamp(1.0, 50.0);  // regel ~639

// FIX:
field.lambda = field.lambda.clamp(0.01, 0.1);  // NOOIT boven 0.1
field.clamp_range = (max_phi * 2.0).clamp(1.0, 5.0);  // NOOIT boven 5.0
```

### Fix 2: `apply_neuromodulation()` — `compute.rs` regel ~1987

**Probleem**: noise is 10-100× te sterk

```rust
// HUIDIG (FOUT):
let baseline_noise = (field.noise_amplitude * 50.0).max(2.0);  // → 7.5!
let d3_boost = baseline_noise * 2.0;  // → 15.0!

// FIX:
let baseline_noise = (field.noise_amplitude * 5.0).max(0.1);  // → 0.75
let d3_boost = baseline_noise * 0.5;  // → 0.375
```

### Fix 3: `update_weights()` — `compute.rs` regel ~241

**Probleem**: geen NaN guard op phi_self/neighbor_phi, geen gradient clipping

```rust
// Na regel 267, voeg toe:
if !dw.is_finite() { dw = 0.0; }
dw = dw.clamp(-1.0, 1.0);  // gradient clipping
```

### Fix 4: Emergency sanitize — `main.rs` regel ~766

**Probleem**: clamp_range=50 maakt sanitize nutteloos

```rust
// HUIDIG:
let cr = if field.clamp_range.is_finite() ... { field.clamp_range } else { 5.0 };

// FIX: HARDCODE een veilige limiet, onafhankelijk van field.clamp_range
let cr = 5.0f32;  // altijd 5.0, adapt_dynamics mag dit niet overrulen
```

### Fix 5: Diagnostics verminderen — `main.rs`

Het PRE-UPDATE diagnostic blok (regel ~703) logt ELKE TICK als er NaN weights zijn. Dit veroorzaakt log-spam maar is niet schadelijk. Na de fix kan het weg of naar `tick % 1000 == 0`.

---

## Wat WEL Werkt

- **heal()** in lib.rs — comprehensive, healt 30+ velden per punt + attn_weights + AdaptiveParams
- **Startup**: heal bij boot zet alles schoon (426 NaN → 0)
- **Attention NaN guard**: `if ctx[d].is_finite()` in update_phi werkt
- **Vocabulaire**: 10.900 tokens, groeit autonoom (BPE merges)
- **Exploratie**: 10.814 pagina's, 19M tokens verwerkt
- **Topologie**: 129.470 couplings, Small-World netwerk intact

---

## Bestanden die Gewijzigd zijn in deze Sessie

### `crates/lucy-state/src/lib.rs` — heal() uitgebreid
- AdaptiveParams healing (emotion_decay, neuromod_interactions, etc.)
- Alle core hyperparameters healing (lambda, beta, alpha, etc.)
- attn_weights NaN→0.0 iteratie
- attn_context NaN cleanup
- Diagnostische counters (nan_phi, nan_pred, nan_mem, etc.)

### `crates/lucy-core/src/main.rs` — NaN detectie + sanitize
- NaN canary closure (`nan_check`) — checkt na elke stap
- Targeted diagnostics (PRE-UPDATE logging)
- adapt_dynamics NaN guards (reset als NaN na adapt)
- Emergency sanitize loop na field_updates
- Weight delta NaN guard in training thread

### `crates/lucy-core/src/compute.rs` — NaN guards
- update_phi: `if ctx[d].is_finite()` voor attn_context
- update_phi: `if !point.phi[d].is_finite() { 0.0 }` hard clamp
- update_attention_weights: error_mag NaN guard
- apply_neuromodulation: is_finite() checks op alle boost factors
- EMA guards op tension, free_energy, etc.

---

## Server Info

- **Host**: 209.38.36.51 (DigitalOcean)
- **Domain**: lucy.world
- **Container**: `lucy` (Docker, image `lucy-lucy`)
- **Volume**: `lucy_lucy-data` → mounted at container /lucy/data
- **Port**: 127.0.0.1:3000 → container 3000
- **Disk**: ~24GB, Docker builds eten ~4GB → altijd `docker system prune` eerst
- **SSH**: `ssh root@209.38.36.51` (gebruik `-o ServerAliveInterval=15` bij builds)

### Snapshot locatie
Data persisted in Docker volume. Snapshot = `data/lucy.snapshot` (~36MB).
Backups: `data/lucy.snapshot.v2.3.bak`, `data/lucy.snapshot.v2.4pre.bak`

---

## Samenvatting: Wat de Nieuwe Agent Moet Doen

1. **Lambda clampen naar max 0.1** (nu gaat het naar 1.0 — 10× te hoog)
2. **Clamp_range limiteren naar max 5.0** (nu gaat het naar 50.0 — sanitize werkt niet)
3. **Noise sterkte /10** (baseline_noise formula: `* 50` → `* 5`)
4. **Gradient clipping** in update_weights (voorkomt weight → NaN)
5. **Deploy & verify**: phi moet stabiel onder 5.0 blijven, w0 mag niet NaN zijn

De root cause is NIET de NaN zelf — die wordt netjes gehealed. Het probleem is dat `adapt_dynamics()` te agressieve parameters instelt die het veld destabiliseren, waarna de NaN sneller terugkomt dan de sanitize het kan bijhouden.
