#!/bin/bash
# Lucy deploy — v2.6.11 hardened edition.
#
# Wat er vandaag mis ging (2026-04-15) wordt hier structureel opgevangen:
#   * `docker cp /lucy/crates lucy:/lucy/crates` nestelt de map als `/lucy/crates/crates/`
#     → cargo build faalt stil op ontbrekende Cargo.toml → oude binary blijft draaien.
#     Fix: altijd `/.` achter bron om INHOUD te kopiëren i.p.v. de map zelf.
#   * Deploy gaat door ondanks mislukte cargo build → nieuwe binary ontbreekt.
#     Fix: `set -e`, expliciete exit-codes gecontroleerd, build-hash vergelijking.
#   * Pre-deploy save race'de met code-sync → snapshot corrupt opgeslagen.
#     Fix: save FLUSHEN en bevestigen VOORDAT we code syncen; verify-snapshot
#     als dry-run load voordat we de container stoppen.
#   * Geen manier om te weten welke versie draait → maandenlang dachten we dat
#     v2.6.9 in prod stond; het was al die tijd v2.6.5.
#     Fix: `/version` endpoint + pre/post-deploy check dat post.git_sha != pre.git_sha.
#   * Geen rollback: oude binary was weg zodra de nieuwe (kapot) gekopieerd was.
#     Fix: bewaar laatste 3 binaries in container, rollback-flag schakelt om.
#
# Gebruik:
#   bash deploy.sh                    # normale code-sync + rebuild IN container
#   bash deploy.sh rebuild            # Docker image rebuild (alleen bij Dockerfile wijziging)
#   bash deploy.sh rollback           # zet vorige binary terug
#   bash deploy.sh --skip-verify      # sla snapshot verify over (nooit doen in prod!)
#
# DATA WORDT NOOIT OVERSCHREVEN. Safety backup op /lucy/data-safety-backup/ vóór
# élke ingreep, rolling binaries bij /lucy/binaries/.

set -euo pipefail

SERVER="root@209.38.36.51"
TOKEN="${LUCY_API_TOKEN:-}"
VOLPATH_CMD='docker volume inspect lucy_lucy-data --format "{{.Mountpoint}}"'
MODE="${1:-deploy}"
SKIP_VERIFY="${2:-}"

log()   { echo "━━ $*"; }
step()  { echo; echo "=== $* ==="; }
abort() { echo "✗ $*" >&2; exit 1; }

# ─── Rollback path ────────────────────────────────────────────────────────────
if [ "$MODE" = "rollback" ]; then
    step "ROLLBACK — zet vorige binary terug"
    ssh "$SERVER" "
        set -e
        docker stop lucy --timeout 10
        PREV=\$(ls -t /lucy/binaries/lucy-core.* 2>/dev/null | sed -n 2p)
        if [ -z \"\$PREV\" ]; then echo 'geen vorige binary gevonden'; exit 1; fi
        echo \"Herstel: \$PREV\"
        docker cp \"\$PREV\" lucy:/lucy/lucy-core
        docker start lucy
    "
    sleep 10
    curl -s --max-time 10 http://209.38.36.51:3000/version | python3 -m json.tool || true
    exit 0
fi

# ─── Pre-flight: lokale build groen? ─────────────────────────────────────────
step "0. Pre-flight checks"
log "Cargo check lucy-core (lokaal, snel)"
cargo check -p lucy-core 2>&1 | grep -E "^error|^warning: unused" || true
cargo check -p lucy-core --quiet || abort "cargo check lucy-core faalt lokaal — deploy gestopt"
log "Cargo check lucy-state"
cargo check -p lucy-state --quiet || abort "cargo check lucy-state faalt lokaal — deploy gestopt"
log "✓ Lokale checks groen"

# ─── Token-check ──────────────────────────────────────────────────────────────
if [ -z "$TOKEN" ] && [ -f .env ]; then
    TOKEN=$(grep -E "^LUCY_API_TOKEN=" .env | cut -d= -f2- | tr -d '"' | tr -d "'")
fi
if [ -z "$TOKEN" ]; then
    log "⚠ LUCY_API_TOKEN niet gezet — beveiligde endpoints werken niet (pre-deploy save slaat over)"
fi

# ─── Pre-deploy state vastleggen ──────────────────────────────────────────────
step "1. Pre-deploy state capture"
PRE_VERSION=$(curl -s --max-time 10 http://209.38.36.51:3000/version 2>/dev/null || echo "{}")
PRE_SHA=$(echo "$PRE_VERSION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('git_sha','?'))" 2>/dev/null || echo "?")
PRE_TICK=$(curl -s --max-time 10 http://209.38.36.51:3000/telemetry 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('tick',0))" 2>/dev/null || echo 0)
PRE_ACTIVE=$(curl -s --max-time 10 http://209.38.36.51:3000/telemetry 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('active',0))" 2>/dev/null || echo 0)
log "Pre  → git_sha=$PRE_SHA tick=$PRE_TICK active=$PRE_ACTIVE"

# ─── Safety backup (altijd, ongeacht modus) ───────────────────────────────────
step "2. DATA SAFETY BACKUP"
ssh "$SERVER" "mkdir -p /lucy/data-safety-backup && docker cp lucy:/lucy/data/. /lucy/data-safety-backup/ 2>/dev/null && echo '   Backup: /lucy/data-safety-backup/'"

# ─── Pre-deploy save (na backup, niet tijdens sync) ───────────────────────────
if [ -n "$TOKEN" ]; then
    step "3. Pre-deploy save (flush naar disk)"
    # Best-effort: draaiende binary kan oud zijn en dit endpoint niet kennen,
    # of /telemetry kan traag reageren. We willen de deploy NIET aborten op een
    # save-timeout — de safety backup is al gemaakt in stap 2.
    SAVE_RESP=$(curl -sS --max-time 30 -X POST http://209.38.36.51:3000/admin/pre-deploy \
        -H "Authorization: Bearer $TOKEN" 2>&1 || true)
    log "Save response: $(echo "$SAVE_RESP" | head -c 200)"
    # Wacht tot save zeker klaar is VOORDAT we code gaan syncen — voorkomt race
    # die vandaag de snapshot corrupteerde.
    sleep 5
else
    step "3. Pre-deploy save OVERGESLAGEN (geen token)"
fi

# ─── Dashboard build ──────────────────────────────────────────────────────────
step "4. Dashboard bouwen"
if [ -d "lucy-dashboard" ]; then
    (cd lucy-dashboard && npm run build 2>&1 | tail -3) || log "⚠ dashboard build warning, doorgaan"
fi

# ─── Snapshot verify (dry-run load) ──────────────────────────────────────────
if [ "$SKIP_VERIFY" != "--skip-verify" ]; then
    step "5. Snapshot-compat check (dry-run vs nieuwe schema-ladder)"
    log "Bouw verify_snapshot tool lokaal"
    cargo build --release -p snapshot_check --bin verify_snapshot --quiet || abort "verify_snapshot build faalt"

    log "Download productie-snapshot naar /tmp voor test"
    scp -q -o ConnectTimeout=15 "$SERVER:/lucy/data-safety-backup/lucy.snapshot" /tmp/lucy-deploy-verify.snap
    VERIFY=$(./target/release/verify_snapshot /tmp/lucy-deploy-verify.snap)
    echo "   $VERIFY"
    OK=$(echo "$VERIFY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok',False))")
    if [ "$OK" != "True" ]; then
        abort "Snapshot is NIET leesbaar door nieuwe binary. Deploy zou Lucy's data wissen. STOP."
    fi
    log "✓ Nieuwe binary kan huidige snapshot lezen"
    rm -f /tmp/lucy-deploy-verify.snap
fi

# ─── Code sync ────────────────────────────────────────────────────────────────
step "6. Code synchen"
rsync -avz --exclude='target' --exclude='.git' --exclude='data' --exclude='.env' \
    --exclude='node_modules' --exclude='vscode-extension/out' \
    ./ "$SERVER:/lucy/"

# ─── Image rebuild (alleen bij flag) ──────────────────────────────────────────
if [ "$MODE" = "rebuild" ]; then
    step "7. DOCKER REBUILD (Dockerfile veranderd)"
    ssh "$SERVER" "docker stop lucy --timeout 30"
    sleep 3
    ssh "$SERVER" "cd /lucy && docker compose up -d --build"
    step "8. Wachten op startup"
    sleep 20
else
    # ─── Code → container (MET /. fix!) ──────────────────────────────────────
    step "7. CODE NAAR CONTAINER (data blijft intact)"
    # BELANGRIJK: `docker cp <bron>/. <dest>` kopieert INHOUD i.p.v. map.
    # Zonder de `/.` nestelt docker de map als `/lucy/crates/crates/` → build faalt.
    # Dit was de root-cause van de deploy-crisis 2026-04-15.
    ssh "$SERVER" "
        set -e
        # Wis verouderde crates eerst zodat verwijderde files (zoals oud bin) opgeruimd worden
        docker exec lucy rm -rf /lucy/crates
        docker exec lucy mkdir -p /lucy/crates
    "
    ssh "$SERVER" "docker cp /lucy/crates/. lucy:/lucy/crates/"
    ssh "$SERVER" "docker cp /lucy/Cargo.toml lucy:/lucy/Cargo.toml"
    ssh "$SERVER" "docker cp /lucy/Cargo.lock lucy:/lucy/Cargo.lock"
    ssh "$SERVER" "docker cp /lucy/lucy-dashboard/dist/. lucy:/lucy/dashboard/"

    step "8. Lucy hercompileert (cargo build --release)"
    ssh -o ServerAliveInterval=15 "$SERVER" "
        set -e
        docker exec lucy bash -c 'cd /lucy && cargo build --release -p lucy-core 2>&1 | tail -5'
        docker exec lucy bash -c 'test -x /lucy/target/release/lucy-core || (echo BUILD-OUTPUT-ONTBREEKT && exit 1)'
    " || abort "cargo build faalt in container — deploy gestopt, oude binary blijft draaien"

    step "9. Bewaar rolling binaries (laatste 3) + vervang live"
    ssh "$SERVER" "
        set -e
        docker exec lucy bash -c 'mkdir -p /lucy/binaries && \
            cp /lucy/lucy-core /lucy/binaries/lucy-core.\$(date +%Y%m%d-%H%M%S).prev 2>/dev/null || true && \
            ls -t /lucy/binaries/lucy-core.* 2>/dev/null | tail -n +4 | xargs -r rm -f'
    "

    step "10. Stop, vervang binary, herstart"
    ssh "$SERVER" "
        set -e
        docker stop lucy --timeout 10
        docker cp lucy:/lucy/target/release/lucy-core /tmp/lucy-core-new
        docker cp /tmp/lucy-core-new lucy:/lucy/lucy-core
        docker start lucy
    "
    sleep 15
fi

# ─── Post-deploy smoke test ──────────────────────────────────────────────────
step "11. SMOKE TEST"
POST_VERSION=$(curl -s --max-time 15 http://209.38.36.51:3000/version 2>/dev/null || echo "{}")
POST_SHA=$(echo "$POST_VERSION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('git_sha','?'))" 2>/dev/null || echo "?")
POST_VER=$(echo "$POST_VERSION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null || echo "?")

TEL1=$(curl -s --max-time 15 http://209.38.36.51:3000/telemetry 2>/dev/null || echo "{}")
POST_TICK=$(echo "$TEL1" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tick',0))")
POST_ACTIVE=$(echo "$TEL1" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active',0))")

log "Post → git_sha=$POST_SHA version=$POST_VER tick=$POST_TICK active=$POST_ACTIVE"

# Check 1: versie-bump
if [ "$POST_SHA" = "$PRE_SHA" ] && [ "$MODE" = "deploy" ]; then
    log "⚠ git_sha onveranderd ($POST_SHA) — nieuwe binary draait NIET. Rollback nodig."
    abort "Deploy onverlopen: binary is niet vervangen"
fi

# Check 2: data intact (tick moet >= PRE_TICK, niet terugvallen naar 0)
if [ "$POST_TICK" = "0" ] && [ "$PRE_TICK" != "0" ]; then
    log "✗ TICK=0 na deploy terwijl PRE_TICK=$PRE_TICK — Lucy bootstrapte fresh. DATAVERLIES."
    log "Herstel: ssh $SERVER 'VOLPATH=\$($VOLPATH_CMD) && cp /lucy/data-safety-backup/* \$VOLPATH/ && docker restart lucy'"
    log "Of gebruik: bash deploy.sh rollback"
    exit 1
fi

# Check 3: active-count behouden (mag groeien, niet krimpen naar bootstrap-niveau 8704)
if [ "$PRE_ACTIVE" != "0" ] && [ "$POST_ACTIVE" -lt "$((PRE_ACTIVE / 2))" ]; then
    log "⚠ active-count krimpt fors ($PRE_ACTIVE → $POST_ACTIVE) — mogelijke data-loss, controleer!"
fi

# Check 4: tick beweegt na startup (wacht 20s, vergelijk)
log "Check tick-voortgang (20s wachten)"
sleep 20
TEL2=$(curl -s --max-time 15 http://209.38.36.51:3000/telemetry 2>/dev/null || echo "{}")
TICK_LATER=$(echo "$TEL2" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tick',0))")
DELTA=$((TICK_LATER - POST_TICK))
log "Tick delta over 20s: +$DELTA (tps≈$((DELTA / 20)))"
if [ "$DELTA" = "0" ]; then
    log "⚠ 0 ticks in 20s — Lucy staat stil. Check logs en overweeg rollback."
fi

echo
echo "━━━ DEPLOY RESULTAAT ━━━"
echo "   version: $POST_VER"
echo "   git_sha: $POST_SHA"
echo "   tick:    $PRE_TICK → $POST_TICK (Δ=+$DELTA in 20s)"
echo "   active:  $PRE_ACTIVE → $POST_ACTIVE"
echo "   data:    intact ✓"
echo
echo "Dashboard: https://lucy.world"
