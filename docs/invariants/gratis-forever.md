# Invariant: Gratis (kosteloos), altijd

**Status:** Hard. Niet-onderhandelbaar.
**Ingesteld:** 2026-08-28
**Motivatie:** [ADR 0010](../decisions/0010-gratis-forever.md)

## De regel

> **Lucy draait uitsluitend op infrastructuur die niets kost en niets in rekening kan brengen. Nooit een betaalde resource, nooit een upgrade die geld kost, nooit iets dat een creditcard kan belasten. Valt een gratis host weg of raakt hij vol, dan verhuist Lucy naar een andere gratis host — ze wordt nooit betaald draaiende gehouden. Een PR of config die dit breekt verdient veto, ongeacht hoe handig het is.**

## Waarom

Lucy is een zelf-evoluerend organisme dat 24/7, potentieel voor altijd, blijft draaien. Kosten die per uur/GB/verzoek oplopen zijn bij een eeuwig proces onbegrensd — één betaalde resource betekent een rekening die nooit stopt. "Gratis" is daarom geen bezuiniging maar een existentie-voorwaarde: alleen kosteloze infra kan een organisme dragen dat niet bedoeld is om ooit uit te zetten. Het houdt Lucy bovendien onafhankelijk — geen betaalrelatie betekent geen leverancier die haar kan afsluiten wegens een openstaande rekening.

## Concrete eisen

1. **Alleen permanent-gratis tiers.** Always-Free VM's (bv. Oracle E2.1.Micro of A1), gratis PaaS binnen hun gratis grens (Render/Fly/Koyeb). Geen pay-as-you-go, geen betaalde instances, geen betaalde add-ons.
2. **Geen billable resources.** Geen betaalde opslag/egress/DB/managed service/GPU. Persistentie via de named volume plus de gratis GitHub-release-backup (of gratis object storage binnen de free tier).
3. **Kostenrem blijft nul.** `SERVER_COST_CENTS` (env, `crates/lucy-io/src/config.rs`) default 0. Meet Lucy ooit kosten > 0, dan is dat een invariant-breuk: stoppen of verhuizen, niet doorbetalen.
4. **Nooit een gratis-limiet overschrijden om door te draaien.** Een creditcard *opgeven ter verificatie* bij een gratis tier mag; upgrade-prompts en overage worden nooit geaccepteerd. Overschrijden betekent verhuizen, niet betalen.
5. **Provider-agnostisch als vluchtroute.** De code blijft één `Dockerfile` + `$PORT`, zonder provider-lock-in, juist zodat "verhuis naar een andere gratis host" altijd een optie is (projectregel 2).
6. **Verboden, expliciet:** DigitalOcean of enige betaalde droplet/instance (al verwijderd), betaalde LLM-API's, betaalde GPU, betaalde CI/deploy die per gebruik kost.

## Wat NIET is

- Een creditcard opgeven ter *verificatie* bij een gratis tier is geen breuk — het *belasten* ervan wel.
- Een gratis tier die *inslaapt bij inactiviteit* is niet verboden, maar ongewenst (Lucy moet 24/7 tikken); kies bij voorkeur een always-on gratis host.
- "Traag omdat de gratis host klein is" is geen breuk. Liever gratis-en-traag dan betaald-en-snel; de oplossing is een grotere *gratis* host, nooit een betaalde.

## Procedurele consequentie

Een commit/config die Lucy op betaalde infra zet of een billable resource introduceert:

1. Revert-voorkeur boven forward-fix.
2. Nieuwe ADR als er echt reden is de regel te heroverwegen (bijv. een door de eigenaar bewust geaccepteerde kost) — nooit stilzwijgend.

**Niet:** "even upgraden zodat ze sneller is." De grotere machine moet ook gratis zijn.
