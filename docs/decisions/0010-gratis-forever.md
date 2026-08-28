# ADR 0010 — Gratis (kosteloos) draaien, voor altijd

**Status:** Geaccepteerd — Uitgerold (2026-08-28)
**Invariant:** [gratis-forever](../invariants/gratis-forever.md)

## Context

Lucy is een continu, potentieel eeuwig draaiend organisme (24/7 tick-loop, autonome web-exploratie, zelf-persistentie). De eigenaar heeft herhaaldelijk en expliciet als harde eis gesteld: het moet gratis blijven — "zolang het maar gratis is", "het moet gratis overal", "nergens van afhankelijk". Eerdere iteraties draaiden op een betaalde DigitalOcean-droplet; dat is in v3.1 verwijderd ten gunste van provider-agnostische, headless deployment.

Bij een eeuwig proces is elke kost die per tijd/gebruik oploopt onbegrensd: een betaalde resource betekent een rekening die nooit stopt en een betaalrelatie die Lucy afhankelijk maakt (afsluitbaar bij wanbetaling). "Gratis" was tot nu toe een gewoonte, geen vastgelegde regel — waardoor een toekomstige wijziging of agent haar ongemerkt op betaalde infra kon zetten.

## Besluit

"Gratis, altijd" wordt een harde, niet-onderhandelbare invariant (zie [gratis-forever](../invariants/gratis-forever.md)): Lucy draait uitsluitend op permanent-gratis infrastructuur; geen billable resources; kostenrem `SERVER_COST_CENTS` blijft 0; bij een vol of wegvallende gratis host verhuist ze naar een andere gratis host in plaats van te betalen. Een wijziging die dit breekt verdient veto.

## Gevolgen

- Deployment wordt begrensd tot always-free tiers. De aanbevolen route blijft een Oracle Always-Free VM (zie [RUNNING.md](../RUNNING.md)); E2.1.Micro (altijd beschikbaar, 1 GB) is de gegarandeerd-gratis basis, A1 (schaars) de gratis upgrade.
- Prestaties zijn ondergeschikt aan kosteloosheid: een kleine, trage gratis host is acceptabel; de oplossing is een grotere *gratis* host, nooit een betaalde.
- Provider-agnosticisme (projectregel 2) wordt niet alleen gemak maar vereiste: het is de vluchtroute die "gratis blijven" mogelijk maakt.
- De invariant is verankerd in CLAUDE.md (projectregel 7), MAP.md en de operationele docs (README, ARCHITECTURE, RUNNING, deploy).

## Verificatie

- `SERVER_COST_CENTS` default 0 in `crates/lucy-io/src/config.rs`; geen code-pad dat een betaalde resource aanroept.
- Geen betaalde provider in de deploy-docs; DigitalOcean-verwijzingen verwijderd (v3.1).
- Draait feitelijk gratis: op dit moment op een Oracle Always-Free E2.1.Micro.

## Open vragen

- Gratis object storage (bv. Oracle's 20 GB) als extra host-onafhankelijke backup naast GitHub — nuttig, nog niet bedraad.
- Als álle gratis tiers ooit wegvallen: dan is stilstand — of een door de eigenaar bewust geaccepteerde uitzondering via een nieuwe ADR — het antwoord, niet stilzwijgend betalen.
