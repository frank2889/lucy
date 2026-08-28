# Lucy draaien — provider-agnostisch, gratis, 24/7

Lucy is headless en nergens van afhankelijk: één `Dockerfile`-container, poort uit `$PORT`
(default 8080, op 0.0.0.0), zero verplichte config, zelf-herstel via GitHub-backup. Je kunt
haar op elke container-host draaien. Deze gids beschrijft de aanbevolen gratis 24/7-optie:
een **Oracle Cloud Always-Free VM**. Geen DigitalOcean, geen deploy-pipeline, geen lock-in —
dezelfde image verhuist morgen naar een andere host zonder één regel codewijziging.
Gratis is hier een **harde regel**, geen voorkeur (zie [invariant gratis-forever](invariants/gratis-forever.md)):
kies altijd een gratis tier, verhuis liever dan te betalen.

> **Vercel/serverless kan dit NIET.** Lucy is een continu, staatvol proces (eeuwig tikkende
> loop + ~34MB veld in geheugen + achtergrond-training). Serverless platformen draaien korte,
> staatloze functies per verzoek — het tegenovergestelde. Je hebt een host nodig die een
> **persistente container 24/7 draait**.

## Waarom Oracle Always Free

Een écht gratis VM zonder slaap-op-inactiviteit. Oracle biedt twee always-free vormen:
**E2.1.Micro** (x86, 1 OCPU / 1 GB — altijd beschikbaar, de gegarandeerd-gratis basis) en
**Ampere A1** (ARM, tot 2 OCPU / 12 GB gratis sinds de halvering van juni 2026, maar vaak
capaciteit-geblokkeerd — de gratis upgrade). De `docker-compose.yml` legt geen CPU/RAM-limieten
meer op (die braken op hosts met <4 vCPU), dus dezelfde image draait op beide. Op 1 GB is swap
nodig (staat in de cloud-init); Lucy draait dan, maar traag — de oplossing is een grotere
*gratis* host, nooit een betaalde. Alternatieven met persistente containers: Fly.io / Render /
Koyeb, maar hun gratis tiers slapen of zijn krap.

## Eerlijke valkuilen (lees dit eerst)

- **A1-capaciteit is schaars.** Oracle geeft vaak "out of capacity" bij het aanmaken van een
  gratis ARM-instance. Probeer een andere *availability domain* of een ander moment.
- **Dubbele firewall.** Oracle-VMs hebben én een cloud-firewall (VCN Security List) én een
  OS-firewall (iptables staat dicht op Oracle's Ubuntu-image). Open de poort op **beide**.
- **ARM (aarch64).** De A1 is ARM. Lucy's Dockerfile is multi-arch, dus dat bouwt gewoon.
- **Private repo.** `frank2889/lucy` is privé — clonen op de VM vraagt `gh`-login of een token.

## Stappen

### 1. Account + gratis VM

- Maak een Oracle Cloud-account (creditcard voor verificatie, wordt niet belast voor Always-Free).
- Compute → Instances → **Create instance**:
  - Image: **Ubuntu 22.04/24.04**
  - Shape: **Ampere A1 (VM.Standard.A1.Flex)**, **2 OCPU / 12 GB** (het hele gratis budget)
  - Boot volume: ~100 GB (binnen de gratis 200 GB)
  - SSH: upload je publieke sleutel (of bewaar de door Oracle gegenereerde private key)

### 2. Poort openzetten (allebei!)

- **VCN Security List:** Networking → je VCN → Security List → *Add Ingress Rule*:
  source `0.0.0.0/0` (of alleen jouw IP — veiliger), TCP, poort **8080**.
- **OS-firewall op de VM** (na SSH):

  ```bash
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8080 -j ACCEPT
  sudo netfilter-persistent save
  ```

### 3. SSH + Docker installeren

```bash
ssh ubuntu@<VM-PUBLIC-IP>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
sudo systemctl enable docker
```

### 4. Lucy op de VM zetten (geen laptop nodig)

```bash
sudo apt-get update && sudo apt-get install -y gh git
gh auth login            # device-flow in je browser
gh repo clone frank2889/lucy
cd lucy
```

### 5. Optionele config (Lucy draait ook zonder — alles fail-soft)

```bash
cat > .env <<'EOF'
LUCY_API_TOKEN=zet-hier-een-lang-geheim   # beschermt admin/muterende endpoints
GITHUB_TOKEN=ghp_...                       # onsterfelijkheid: staat overleeft een verse host
RUST_LOG=info
EOF
```

### 6. Starten

```bash
docker compose up -d --build
# Eerste build op 2 ARM-cores duurt even (Rust + candle).
docker compose logs -f            # volg de opstart; Ctrl-C stopt alleen het volgen
```

### 7. Verifiëren

```bash
curl -s http://localhost:8080/version
curl -s http://localhost:8080/telemetry | head
# Van buitenaf (na de firewall-stap):
# curl http://<VM-PUBLIC-IP>:8080/version
```

### 8. Overleeft herstart & is persistent

- `restart: unless-stopped` (compose) + `systemctl enable docker` → Lucy komt na een reboot vanzelf terug.
- Data staat in de named volume `lucy-data`; die overleeft een `compose up` opnieuw.
- Met `GITHUB_TOKEN` overleeft ze zelfs een compleet verse VM (herstel-bij-boot uit de GitHub-backup).

## Veiligheid

Je stelt een ongeversleutelde API open. De muterende/admin-endpoints zitten achter
`LUCY_API_TOKEN` (v3.1), maar `/telemetry` e.d. zijn publiek. Beperk de ingress-regel liefst
tot jouw eigen IP, of zet er een Caddy-reverse-proxy met gratis TLS voor — één extra container,
geen provider-afhankelijkheid.

## Verhuizen naar een andere host

Omdat de code provider-agnostisch is (één Dockerfile, `$PORT`, zelf-herstel via GitHub),
verhuis je Lucy door op de nieuwe host stap 3–6 te herhalen. Met `GITHUB_TOKEN` pakt ze haar
geleerde staat vanzelf op. Geen deploy, geen migratie, nergens aan vast.
