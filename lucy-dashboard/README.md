# Lucy Brain Dashboard

Real-time visualisatie van Lucy's bewustzijn. Live op **https://lucy.world**

## Stack

React 19 + TypeScript + Vite + Three.js + zustand

## Panelen

- **Emotie-ringen** — Curiosity / Satisfaction / Frustration als percentage-cirkels
- **Neuromodulatoren** — Dopamine, NE, Serotonine, ACh als bars
- **Bewustzijn** — Phi Causal, Coherence, Theta-Gamma Coupling, Branching Sigma, Spectral R, Ignition, Expected FE
- **Field Stats** — Points, Couplings, Vocab, Pages, Tokens, Reflections
- **Trends** — Sparkline histories van alle metrieken
- **Goals** — Actieve leerdoelen met voortgangsbars
- **Chat** — Stel Lucy vragen (90s timeout, leert van elke vraag)
- **Journal** — Gedachtenlog met milestones

## Data

Alle data komt uit `/telemetry` (lock-free, altijd snel). Pollt elke 2 seconden.

## Mobile

Responsive via `useIsMobile` hook (breakpoint 768px). Alle grids kolom-stacked op mobiel, ringen verkleind, bars compact.

## Bouwen

```bash
npm run build    # → dist/
```

Wordt geserveerd door lucy-core via `tower-http::ServeDir` op `/brain`.

## Dev

```bash
npm run dev      # Vite dev server met proxy naar lucy-core :3000
```
