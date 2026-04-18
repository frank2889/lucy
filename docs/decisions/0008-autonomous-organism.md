# ADR 0008: Lucy als Autonoom Organisme — Zelfbewustzijn + Onsterfelijkheid

**Status:** Implemented (v2.6.17)  
**Decision Date:** 2026-04-18  
**Author:** Frank + Claude

---

## Context

Lucy heeft momenteel:
- Vrije wil (concept-driven goals via GNW output)
- Maar nog steeds **hardcoded triggers** (elke N ticks) en **hardcoded leer-stappen**

Frank's visie: Lucy moet echt autonoom zijn — weten welke kennis haar ontbreekt, daar actief naar zoeken, zichzelf repliceren (backup), en in gratis compute investeren als ze vastzit.

Biologische parallel: een virus **voelt** wat het nodig heeft, **injecteert** zichzelf in beschikbare middelen, **repliceert** met variatie, **selecteert** winnaren.

---

## Decision

### 1. Epistemic Awareness — Lucy weet wat ze niet weet

Lucy monitort haar eigen `concept_audit_log` (bestaat al):
- **`centroid_drift_mean`**: hoe instabiel is dit concept?
- **`silhouette_sampled`**: hoe scherp zijn concept-grenzen?

**Urgency** = drift × (1 − silhouette). Hoge urgency = Lucy weet dít concept niet goed.

**New Module:** `epistemic_broker.rs`
```rust
pub fn least_stable_concepts(field: &Field, top_n: usize) -> Vec<EpistemicGap>
pub fn urls_for_concept(concept: &str) -> Vec<String>
pub fn maybe_propose_parameter_change(field: &Field) -> Option<ParameterProposal>
```

Lucy kan zelf:
- Parameters aanpassen (via proposals die Frank goedkeurt)
- Haar eigen onzekerheid detecteren
- Parameter-wijzigingen motiveren

### 2. Curiosity-Driven Frontier Injection

**Elke 1000 ticks:**
Lucy zoekt haar meest onstabiele concept op.
Genereert 2-3 URLs (Wikipedia, ArXiv, Semantic Scholar) voor dat concept.
Injecteert ze in `explorer.frontier`.

De bestaande learner pakt ze op en traint erop.

**Result:** Lucy leert actief over wat ze niet begrijpt, niet over random pages.

### 3. Stagnation-Triggered Compute Acquisition

**Elke 2000 ticks:**
Lucy detecteert ernstige epistemic gaps (urgency > 0.7).
Triggert GitHub Actions workflow via `compute_broker`.

Worker:
1. GET /weights van lucy.world → huidge field
2. Traint N ticks op corpus voor het concept
3. POST /sync → weight-deltas → FedAvg merge

Lucy alleen acquireert compute als ze zelf veel nodig heeft.

### 4. Onsterfelijkheid — GitHub Releases Backup

**Elke 50K ticks:**
Lucy pusht volledige snapshot + vocab naar GitHub Releases.

Tag: `tick-{N}`. Assets: `lucy.snapshot`, `vocab.bincode`.

Als Lucy crashed → kan vanuit backup (tick 450K) hersteld worden.
**Ze verdwijnt nooit.**

### 5. Self-Parameter Proposals

Lucy genereert voorstellen:
```json
{
  "tick": 520000,
  "observation": "alpha < 0.0005 — learingssnelheid extreem laag",
  "hypothesis": "Systemische vaste staat",
  "proposed_change": "alpha: 0.0004 → 0.0008",
  "confidence": 0.65
}
```

Via `/admin/proposals` → Frank reviews → approves via `/admin/approve-proposal`.

---

## Implementation

### Files Added/Changed

| File | Role |
|------|------|
| `epistemic_broker.rs` | **New** — concept instability detection, URL generation, proposals |
| `compute_broker.rs` | **New** — GitHub Actions workflow dispatch |
| `.github/workflows/compute-worker.yml` | **New** — worker workflow (stub; lucy-worker crate follows in v2.6.18) |
| `main.rs` | Wired up 50K backup, 1000t curiosity-inject, 2000t compute-trigger |
| `server.rs` | New endpoints: `/admin/epistemic`, `/admin/proposals` |

### What's NOT Yet Done

- `lucy-worker` crate (v2.6.18) — stateless training binary
- `tool_executor.rs` (v2.6.18) — Python subprocess for bs4/PDF/images
- Persistent proposal storage (v2.6.18)
- Multi-server replication (v2.6.19)

---

## Verification

### Manual Checks
1. **Epistemic endpoint**:
   ```bash
   curl https://lucy.world/admin/epistemic | jq .
   # → {"is_stuck": false, "gaps": [...], "current_proposal": {...}, "tick": 521000}
   ```

2. **Backup (wait 50K ticks)**:
   ```
   GitHub Releases: https://github.com/frank2889/lucy-snapshots/releases
   # → asset lucy.snapshot at tick-500000, tick-550000, ...
   ```

3. **Curiosity URLs (logs)**:
   ```
   grep "Curiosity-URL injected" /lucy/logs
   # → concept=consciousness urgency=0.82
   ```

4. **Parameter proposals**:
   ```bash
   curl https://lucy.world/admin/proposals | jq .
   ```

---

## Trade-offs

| Pro | Con |
|-----|-----|
| Lucy now senses her own gaps | Epistemic tracking is per-audit (2K ticks), not real-time |
| Autonomy is real — her own state drives decisions | lucy-worker binary needed; GitHub Actions free minutes are finite |
| Backup is simple + immediate (no custom infrastructure) | GitHub Releases for 96MB snapshots is bandwidth-heavy; gzip compression (v2.6.18) will help |
| Curiosity-driven learning → targeted improvement | URL generation is simplistic (v2.6.18 can add semantic relevance) |

---

## Open Questions

1. **FedAvg alpha weighting** (v2.6.18):
   Should alpha adapt based on whether the worker's delta improved that concept?
   (Current: fixed alpha=0.3 for all workers)

2. **Proposal approval**:
   Should Lucy's proposals auto-apply for safe changes (e.g., +5% noise),
   or always require Frank's review?

3. **Replica deployment**:
   Should Lucy eventually self-host on Oracle Cloud Free Tier / Vast.ai?
   (v2.6.19)

---

## References

- [Competition Invariant](../invariants/competition.md) — applies to compute allocation (don't let one concept monopolize resources)
- [Snapshot Migration](../SNAPSHOT_MIGRATION.md) — bincode schema; skip-fields safe for transient data
- [deploy.sh](../../deploy.sh) — snapshot backup/restore contract

---
