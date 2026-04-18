# ⚠ Dit bestand is gesplitst

Op 2026-04-18 is dit 444-regelige document opgedeeld in losse bestanden per soort informatie. Zie [docs/MAP.md](MAP.md) voor de plattegrond.

## Waar staat wat nu?

| Oude sectie in dit bestand              | Nieuwe locatie                                                  |
|-----------------------------------------|-----------------------------------------------------------------|
| §1 TL;DR, §3 Observatie, §10.3 Data     | [incidents/2026-04-15-500k-collapse.md](incidents/2026-04-15-500k-collapse.md) |
| §4 Fase 1 (top-K sparsity)              | [decisions/0001-top-k-sparsity.md](decisions/0001-top-k-sparsity.md) |
| §6 Fase 2 (ignition cap), §9.1-9.3 open vragen | [decisions/0002-ignition-cap.md](decisions/0002-ignition-cap.md) |
| §7 Architectuur-invariant, §10.4 Filosofie | [invariants/competition.md](invariants/competition.md)        |
| §5 Infrastructuur-hardening             | [CHANGELOG.md](CHANGELOG.md) v2.6.11 + [deploy.sh](../deploy.sh) docstring |
| §8 Risk-zone logging                    | Code-comment in `crates/lucy-core/src/compute.rs:3411-3434`     |
| §9.4 Goal-spiral                        | [decisions/0003-goal-completion-zombie-fix.md](decisions/0003-goal-completion-zombie-fix.md) (gedeeltelijk beantwoord) |
| §10 Bijlagen                            | Opgegaan in de relevante ADR's                                  |
| §11 Methodologische noot                | [CLAUDE.md](../CLAUDE.md) §Projectregels #5 + #6                |

**Waarom gesplitst?** Eén monolithisch document mengde feiten, beslissingen en invarianten. Agents en GPT-reviews vonden de ADR-kern niet terug omdat hij vastzat in een 400-regelig verhaal. ADR-per-beslissing is de natuurlijke vorm voor Lucy's fase-gewijze werkwijze.
