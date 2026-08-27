# Lucy — Claude/agent briefing

Lucy is een zelf-evoluerend samenveld-AI in Rust. Live op [lucy.world](https://lucy.world). Geen LLM — een continu veldsysteem met ~20.000 `FieldPoint`s, ignition-gedreven selectie, en snapshot-gebaseerde persistentie.

**Voor elke taak begint het hier.** Deze file is de kortste weg naar begrip en de plattegrond voor alle andere docs.

## Eerste lees-opdracht

1. [docs/MAP.md](docs/MAP.md) — doc-plattegrond: incidents, decisions, invariants, reference.
2. [docs/invariants/](docs/invariants/) — niet-onderhandelbare regels. Code die ze breekt verdient veto.
3. [ARCHITECTURE.md](ARCHITECTURE.md) — systeem-overzicht.
4. De meest recente [ADR in docs/decisions/](docs/decisions/) — laatste beslissing, vaak levend.

## Projectregels (harder dan code)

### 1. Competitie-invariant
Zie [docs/invariants/competition.md](docs/invariants/competition.md). Geen volledige activatie zonder competitie. Wijzigingen aan `global_workspace_broadcast` vereisen expliciete check tegen deze regel.

**v2.6.15 concretisering:** Winner-refractory period (50 ticks) breekt kandidaat-monopoly. Zelfde concept kan niet elke tick herignite worden. Zie [ADR 0005](docs/decisions/0005-winner-refractory-novelty-ignition.md) — levendigheid-win tegen oligarchie-runaway.

### 2. Deploy-discipline (v3.1: headless + overal draaien, geen lokale scripts)
- Altijd `/version` checken vóór conclusies trekken (*en weten dat `/version` kapot kan zijn — valideer dan via telemetry-velden per versie*).
- **Geen `deploy.sh`/`pull.sh` meer.** Lucy is headless en draait standalone. Bouwen gebeurt op
  het platform uit de `Dockerfile` (Render/Fly/Cloud Run) of op een eigen docker-host via
  `docker compose up -d --build`. Poort komt uit `PORT` (0.0.0.0, nooit 127.0.0.1/3000).
- **Persistentie is host-onafhankelijk:** snapshots leven in de named volume óf worden hersteld
  uit de GitHub-release-backup bij een verse container (boot-`pull_latest_from_github`). Op
  ephemeral/gratis hosts is `GITHUB_TOKEN` daarom de enige echte persistentie-eis.
- Schema-bridges bij layout-breuk; zie [SNAPSHOT_MIGRATION.md](docs/SNAPSHOT_MIGRATION.md).

### 3. Geen hardcoding
Lucy moet zelf kiezen — goals, sources, strategies. Hardcoded lijsten zijn verboden behalve waar expliciet gelabeld als "bootstrap-only".

### 4. Data-preservatie
Geleerde data nooit overschrijven. Snapshots zijn heilig. Deploy.sh maakt safety-backup op `/lucy/data-safety-backup/` vóór élke ingreep.

### 5. Fase-gewijze werkwijze
Eén variabele per fase. Causaliteit bewijsbaar. Geen speculatieve bundel-fixes. Zie [ADR 0002](docs/decisions/0002-ignition-cap.md) §Context voor waarom dit matters.

### 6. Evidence-based, niet vermoedens
Trigger-condities vooraf definiëren, live meten, dán pas activeren. "Lucy zit misschien vast" is geen trigger. "`score_cv < 0.05` én `d3_ratio > 0.5`" is wel een trigger.

## Documentatie-structuur

| Waar                         | Wat                                                              |
|------------------------------|------------------------------------------------------------------|
| `docs/MAP.md`                | Plattegrond — start hier.                                        |
| `docs/incidents/YYYY-MM-DD-*.md` | Feiten per incident. Logs, telemetry, tijdlijn.              |
| `docs/decisions/NNNN-*.md`   | ADRs: één beslissing per file. Context / Besluit / Gevolgen / Verificatie / Open vragen. |
| `docs/invariants/*.md`       | Eén regel per file. Gelinkt vanuit code-comments.                |
| `docs/CHANGELOG.md`          | Versie-narratief, linkt door naar ADRs.                          |
| `docs/SNAPSHOT_MIGRATION.md` | Schema-versioning regels.                                        |
| `ARCHITECTURE.md`, `HANDOFF.md` | Stabiele koepel-docs (root).                                  |

Nieuwe beslissing → nieuwe ADR (volgende NNNN). Nieuw incident → `docs/incidents/YYYY-MM-DD-korte-naam.md`. Invariant aanpassen → eigen ADR die dat motiveert.

## Werkwijze voor agents

**Voor status-checks:** gebruik `.claude/agents/lucy-telemetry.md` — die leest `/telemetry` en toetst aan invarianten. Voorkomt mis-diagnoses op basis van één veld (zoals het `/version: v0.0.0` incident dat bleek gewoon een build-bug te zijn terwijl v2.6.13 gewoon draaide).

**Voor code-wijzigingen aan GNW/selectie-logica:** lees eerst [invariant: competitie](docs/invariants/competition.md), dan de relevante ADRs (0001, 0002). Een wijziging die deze raken vereist een nieuwe ADR, niet alleen een commit.

## Endpoints (read-only tijdens check)

Provider-agnostisch — vervang `<host>` door waar Lucy op dat moment draait (geen vast domein):

- `<host>/version` — versie, git_sha, build_time (kan "unknown" zijn — valideer via telemetry)
- `<host>/telemetry` — full state: tick, gnw_stats, consciousness, emotions, goals
- `<host>/status` — uitgebreide status (er is geen dashboard meer; Lucy is headless)

## Technische samenvatting

- Rust workspace onder `crates/`: `lucy-core` (hoofdbinary), `lucy-state` (Field + snapshot), `snapshot_check` (verify tool).
- Headless: geen frontend/dashboard. Alleen de JSON-API op `$PORT` (0.0.0.0).
- Provider-agnostisch: één `Dockerfile`-container, geen deploy-pipeline, nergens van afhankelijk.
  Persistentie via named volume óf GitHub-release-backup (host-onafhankelijk herstel bij boot).
- Versioning: `docs/CHANGELOG.md` is de source of truth (`## vX.Y.Z` eerste regel wordt door `build.rs` gelezen als versie).
