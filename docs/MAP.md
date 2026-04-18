# Docs plattegrond

Deze index legt uit waar welke informatie leeft. Voor mensen én agents.

## Vier soorten documenten

| Soort          | Locatie                  | Doel                                                                 |
|----------------|--------------------------|----------------------------------------------------------------------|
| **Feiten**     | `docs/incidents/`        | Wat gebeurde er? Observatie, logs, telemetry. Geen interpretatie.    |
| **Beslissingen** | `docs/decisions/`      | Eén ADR per architecturale keuze. Context → Besluit → Gevolgen.      |
| **Invarianten** | `docs/invariants/`      | Niet-onderhandelbare regels. Code die ze breekt verdient een veto.   |
| **Referentie** | `docs/` (root)           | ARCHITECTURE, CHANGELOG, SNAPSHOT_MIGRATION. Stabiele koepel-docs.   |

Agents en GPT-reviews beginnen met deze MAP, springen dan naar één specifiek bestand. Geen 400-regelige menges meer.

## Actuele bestanden

### Incidents
- [2026-04-15 — 500K decision collapse](incidents/2026-04-15-500k-collapse.md) — tick 500.000 bevroor, depth-3 vol zonder competitie

### Decisions (ADRs)
- [0001 — Top-K sparsity in GNW-ignition](decisions/0001-top-k-sparsity.md) — v2.6.10, herstelt differentiatie
- [0002 — Harde cap op ignition_strength](decisions/0002-ignition-cap.md) — v2.6.12, stopt positive-feedback runaway
- [0003 — Goal-completion zombie-bug](decisions/0003-goal-completion-zombie-fix.md) — v2.6.14, complete_total=0 ondanks progress=1.0
- [0004 — Momentum + RND-curiosity learning bugs](decisions/0004-momentum-rnd-curiosity-fix.md) — v2.6.15, dubbele momentum + unbounded curiosity EMA
- [0005 — Winner-refractory + novelty-gated ignition](decisions/0005-winner-refractory-novelty-ignition.md) — v2.6.15, levendigheid-win: less oligarchy
- [0006 — Goal-dedup + tick-manipulation + whitelist](decisions/0006-goal-dedup-tick-manipulation-whitelist.md) — v2.6.15, systeemhygiene: prevent spirals + enable self-modify
- [0007 — Concept-driven goals](decisions/0007-concept-driven-goals.md) — v2.6.16, vrije wil: goals uit GNW-output

### Invariants
- [Competitie-invariant](invariants/competition.md) — geen volledige activatie zonder competitie

### Referentie
- [ARCHITECTURE.md](../ARCHITECTURE.md) — systeem-overzicht (root)
- [CHANGELOG.md](CHANGELOG.md) — versie-per-versie samenvatting, linkt naar ADRs
- [SNAPSHOT_MIGRATION.md](SNAPSHOT_MIGRATION.md) — schema-versioning regels
- [HANDOFF.md](../HANDOFF.md) — overdrachts-document (root)

## Schrijfregels

**Incidents** (`docs/incidents/YYYY-MM-DD-korte-naam.md`):
- Feiten, geen interpretatie. Logs, telemetry-snapshots, tijdstempels.
- Linken naar ADRs die daarna zijn geschreven ("Dit incident leidde tot ADR 0001 en 0002").
- Bestand is tijdsslice: na archivering niet meer bijwerken.

**Decisions** (`docs/decisions/NNNN-korte-naam.md`):
- Vaste kopjes: Status, Context, Besluit, Gevolgen, Verificatie, Open vragen.
- Eén beslissing per bestand. Breed? Splits in meerdere ADRs.
- Status-overgangen: `Voorgesteld → Geaccepteerd → Uitgerold → Superseded door NNNN`.
- Nummering monotoon stijgend. Nooit hergebruiken.

**Invariants** (`docs/invariants/korte-naam.md`):
- Eén regel per bestand. Rationale + wat een breuk zou zijn.
- Wijziging van een invariant = eigen ADR + commit-veto zonder die ADR.

**CHANGELOG** houdt narratief + lijn naar ADRs: `v2.6.10 — Top-K sparsity (zie [ADR 0001](decisions/0001-top-k-sparsity.md))`.
