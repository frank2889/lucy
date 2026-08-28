# deploy/ — Lucy geautomatiseerd opstarten

Lucy is provider-agnostisch: één `Dockerfile`-container die overal draait. `cloud-init.yaml`
automatiseert het opstarten op een verse VM.

> **Alleen gratis infra** — harde regel, zie [gratis-forever](../invariants/gratis-forever.md). Kies always-free shapes (Oracle E2.1.Micro of A1), nooit een betaalde instance.

Wat een script WEL en NIET kan:

| | Kan een script? |
|---|---|
| Cloud-account aanmaken | Nee — mens-gated (browser + creditcard/ID-verificatie, tegen abuse) |
| VM aanmaken | Deels — met OCI CLI/Terraform (vereist API-credentials, dus eerst een account) |
| Docker + Lucy installeren en starten op de VM | Ja — dat doet `cloud-init.yaml` op de eerste boot |
| Cloud-firewall openen (VCN Security List) | Deels — handmatig (1 console-klik) of via CLI |

## Manier A — cloud-init plakken (geen SSH, geen CLI nodig)

1. Maak een VM aan (Oracle Always-Free: Ampere A1, 2 OCPU / 12 GB, Ubuntu).
2. Vul in `cloud-init.yaml` de tokens in (`__GITHUB_TOKEN__`, `__LUCY_API_TOKEN__`).
3. Plak de inhoud in het veld **"user data" / "cloud-init"** bij het aanmaken.
4. Open in de cloud-console ingress **TCP 8080** (VCN Security List).
5. Klaar. Op de eerste boot installeert de VM Docker, cloont Lucy en start haar.
   Volg met: `sudo tail -f /var/log/lucy-bootstrap.log`, en later `docker compose logs -f`.

## Manier B — één commando met de OCI CLI (vereist een geconfigureerde OCI CLI)

Als je de OCI CLI hebt ingesteld (`oci setup config`, dus mét account + API-key), maakt dit de
VM aan én hangt het cloud-init eraan:

```bash
oci compute instance launch \
  --availability-domain "<AD-NAAM>" \
  --compartment-id "<COMPARTMENT_OCID>" \
  --shape "VM.Standard.A1.Flex" \
  --shape-config '{"ocpus":2,"memoryInGBs":12}' \
  --image-id "<UBUNTU_ARM_IMAGE_OCID>" \
  --subnet-id "<SUBNET_OCID>" \
  --assign-public-ip true \
  --user-data-file deploy/cloud-init.yaml \
  --wait-for-state RUNNING
```

Vul de `<...>`-OCID's in vanuit je tenancy (Console → Compute/Networking). Open daarna nog de
ingress-regel (of doe dat via `oci network security-list update`).

## Eerlijke waarschuwing over de token

Bij manier A staat `GITHUB_TOKEN` in de VM-metadata (leesbaar voor wie toegang tot de instance
heeft). Alternatief: laat het token uit het cloud-init en doe na boot eenmalig `gh auth login` +
`git clone` handmatig. Gebruik bij voorkeur een **fine-grained PAT** met alleen `repo`-lees-recht
op `frank2889/lucy`, zodat de blootstelling minimaal is.

## Verhuizen / meerdere hosts

Dezelfde `cloud-init.yaml` werkt op elke cloud met user-data-support (AWS, GCP, Hetzner, …).
Met `GITHUB_TOKEN` gezet pakt een nieuwe VM automatisch Lucy's geleerde staat op uit de
GitHub-release-backup — zo leeft ze overal, zonder handmatige migratie.
