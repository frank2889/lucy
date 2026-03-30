# Lucy — Zelflerend Veld-Systeem

Lucy is een zelflerende AI gebouwd op **equilibrium propagation** op een small-world netwerk.
Geen transformer, geen lookup cache — Lucy denkt écht na door haar veld te laten settlen.

## Architectuur

```
┌──────────────────────────────────────────────────────────┐
│  lucy-core (binary)                                      │
│                                                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ server  │  │  query   │  │ thinker  │  │ learner  │  │
│  │  :3000  │  │ denk-    │  │ autonome │  │ explorer │  │
│  │  HTTP   │  │ loops    │  │ doelen   │  │ training │  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │            │             │              │         │
│       └────────────┴─────────────┴──────────────┘         │
│                         │                                 │
│                    ┌────┴─────┐                           │
│                    │  Field   │  8704 punten × DIM=256    │
│                    │  (veld)  │  small-world netwerk      │
│                    └──────────┘                           │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  lucy-state (lib)    — Field, FieldPoint, Goal, etc.     │
│  lucy-io (lib)       — HTTP client, filesystem, config    │
│  lucy-gpu (binary)   — GPU compute worker (optioneel)     │
└──────────────────────────────────────────────────────────┘
```

## Draaien

```bash
# Lokaal (development)
cargo run -p lucy-core

# Docker (productie)
docker compose build && docker compose up -d

# Deploy naar server
./deploy.sh
```

## API Endpoints

| Route           | Methode | Beschrijving                                        |
|-----------------|---------|-----------------------------------------------------|
| `/`             | GET     | Versie-info en beschikbare endpoints                |
| `/status`       | GET     | Live veld-statistieken                              |
| `/ask`          | POST    | Stel een vraag — Lucy denkt na en antwoordt         |
| `/feed`         | POST    | Geef trainingsdata (JSON of tekst)                  |
| `/feed-code`    | POST    | Voed Lucy broncode                                  |
| `/activity`     | GET     | Stemming, emoties, interesses, leervoortgang        |
| `/dashboard`    | GET     | HTML dashboard                                      |
| `/journal`      | GET     | Lucy's dagboek-entries                              |
| `/sense`        | POST    | Sensorische input                                   |

### POST /ask — Vraagverwerking

**Request:** platte tekst (de vraag)

**Response:**

```json
{
  "answer": "Lucy's antwoord",
  "source": "field",
  "settle_ticks": 10,
  "tension": "0.0042",
  "tokens_generated": 15,
  "think_rounds": 1,
  "coherence": "0.8500",
  "facts_retrieved": 2,
  "internal": {
    "emotions": {
      "curiosity": "0.123",
      "satisfaction": "0.456",
      "frustration": "0.012"
    },
    "goals": [
      {
        "description": "Begrijp: wat is bewustzijn?",
        "strategy": "Leren: bewustzijn",
        "priority": "0.8",
        "progress": "0%",
        "new": true
      }
    ],
    "learning": {
      "vocab_size": 3784,
      "pages_read": 12,
      "tokens_processed": 50000,
      "tick": 15000
    },
    "interests": ["bewustzijn", "taal", "wiskunde"],
    "neuromodulators": {
      "dopamine": "0.050",
      "serotonin": "0.301"
    }
  }
}
```

#### Response velden

| Veld | Type | Beschrijving |
|------|------|-------------|
| `answer` | string | Lucy's antwoord in tekst |
| `source` | string | Altijd `"field"` — komt uit veld-dynamica |
| `settle_ticks` | number | Hoeveel ticks het veld nodig had om te settlen |
| `tension` | string | Restspanning na settlen (lager = zekerder) |
| `tokens_generated` | number | Aantal tokens in het antwoord |
| `think_rounds` | number | Hoeveel denk-rondes Lucy gebruikte |
| `coherence` | string | Stabiliteit tussen denk-rondes (0–1) |
| `facts_retrieved` | number | Feiten uit episodisch geheugen (RAG) |
| `internal` | object | Lucy's interne staat (zie onder) |

#### `internal` — Interne staat

Dit blok laat zien wat Lucy intern doet. Het wordt meegegeven bij élk /ask response.

| Veld | Beschrijving |
|------|-------------|
| `emotions.curiosity` | Nieuwsgierigheid (gemiddeld over actieve veldpunten) |
| `emotions.satisfaction` | Tevredenheid |
| `emotions.frustration` | Frustratie |
| `goals[]` | Actieve leerdoelen (max 5). Goals met `"new": true` zijn zojuist door deze vraag aangemaakt |
| `goals[].description` | Wat Lucy wil leren |
| `goals[].strategy` | Hoe: "Leren: ...", "Verkennen: ...", "Nadenken: ..." |
| `goals[].priority` | 0.0–1.0, hoger = urgenter |
| `goals[].progress` | "0%"–"100%" |
| `learning.vocab_size` | Vocabulaire grootte |
| `learning.pages_read` | Webpagina's gelezen door explorer |
| `learning.tokens_processed` | Totaal verwerkte tokens |
| `learning.tick` | Huidige tick (hartslag van het veld) |
| `interests[]` | Onderwerpen waar Lucy op let (max 10) |
| `neuromodulators.dopamine` | Beloningssignaal / drive |
| `neuromodulators.serotonin` | Tevredenheid / kalmte |

### POST /feed — Training

**JSON formaat:** `{ "input": "vraag", "target": "antwoord" }` → EP training
**Tekst formaat:** platte tekst → sequential prediction

### GET /activity — Lucy's activiteit

Geeft Lucy's huidige stemming, emoties, neuromodulatoren, exploratie-status,
leervoortgang en recente observaties.

## VSCode Extension

De `vscode-extension/` map bevat een sidebar met twee tabs:

### Chat Tab
- Stel Lucy vragen via een chatinterface
- Elk antwoord toont meta-badges: denk-rondes, coherentie, feiten
- **Uitklapbaar "Waar Lucy mee bezig is" paneel** onder elk antwoord met:
  - Actieve doelen (nieuwe goals groen gemarkeerd met NIEUW)
  - Emoties (nieuwsgierigheid, tevredenheid, frustratie)
  - Neuromodulatoren (dopamine, serotonine)
  - Leervoortgang (vocabulaire, pagina's, tokens, tick)
  - Interesses (tags)

### Activity Tab
- Toont Lucy's stemming, emoties, neurochemie
- Leerstatistieken en exploratie-voortgang
- Huidige interesses als tags
- Recente observaties uit het dagboek

## Crates

### lucy-state
Core datastructuren: `Field`, `FieldPoint`, `Goal`, `GoalStrategy`, `Coupling`.
Het veld is een small-world netwerk van 8704 punten met DIM=256 dimensies.

### lucy-core
De binary. Bevat:
- **server.rs** — HTTP API (Axum)
- **query.rs** — Vraagverwerking met denk-loops en meta-intelligentie
- **compute.rs** — Veld-dynamica: settle, equilibrium propagation, attention
- **encoding.rs** — Tekst → embeddings (BPE tokenizer)
- **decoding.rs** — Veld-output → tekst (temperature sampling, repetition penalty)
- **thinker.rs** — Autonome doelgeneratie en beslissingen
- **learner.rs** — Explorer: webpagina's lezen, interesses bijhouden
- **consolidation.rs** — Replay buffer en geheugenconsolidatie
- **journal.rs** — Lucy's dagboek
- **self_modify.rs** — Zelf-aanpassing van hyperparameters

### lucy-io
Shared I/O: HTTP client, filesystem helpers, configuratie.

### lucy-gpu
Optionele GPU compute worker voor versnelde veld-berekeningen.

## Docker

```yaml
# docker-compose.yml
services:
  lucy:
    build: .
    ports: ["3000:3000"]
    volumes: ["./data:/lucy/data"]
    mem_limit: 6g
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:3000/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Omgevingsvariabelen

| Variabele | Beschrijving |
|-----------|-------------|
| `LUCY_API_TOKEN` | Bearer token voor beveiligde endpoints (/self-modify, /exec, /restart) |
| `RUST_LOG` | Log level (default: `info`) |
