#!/bin/bash
# Lucy v2.5 — Data-Safe Deploy naar DigitalOcean
#
# Gebruik:
#   bash deploy.sh           (standaard — code sync + hercompileer IN container, DATA INTACT)
#   bash deploy.sh rebuild   (Docker image rebuild — alleen als Dockerfile veranderd is)
#
# DATA WORDT NOOIT OVERSCHREVEN. De standaard deploy:
# 1. Syncht code naar server
# 2. Kopieert code IN de draaiende container
# 3. Lucy hercompileert zichzelf (via /restart of cargo build)
# 4. Container herstart met DEZELFDE data volume
#
# `rebuild` is alleen nodig bij Dockerfile wijzigingen (nieuwe apt packages, etc.)
# en maakt EERST een safety backup.

SERVER="root@209.38.36.51"
TOKEN="${LUCY_API_TOKEN:-}"
VOLPATH_CMD='docker volume inspect lucy_lucy-data --format "{{.Mountpoint}}"'

if [ -z "$TOKEN" ]; then
    echo "  LUCY_API_TOKEN niet gezet. Beveiligde endpoints niet beschikbaar."
    echo "   export LUCY_API_TOKEN=jouw-token"
fi

echo "=== 0. DATA SAFETY BACKUP ==="
ssh $SERVER "mkdir -p /lucy/data-safety-backup && \
    docker cp lucy:/lucy/data/. /lucy/data-safety-backup/ 2>/dev/null && \
    echo '   Backup: /lucy/data-safety-backup/'" 2>/dev/null
echo ""

echo "=== 1. Pre-deploy save ==="
if [ -n "$TOKEN" ]; then
    curl -s --max-time 15 -X POST http://209.38.36.51:3000/admin/pre-deploy \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null
fi
sleep 3
echo ""

echo "=== 2. Dashboard bouwen ==="
if [ -d "lucy-dashboard" ]; then
    (cd lucy-dashboard && npm run build 2>&1 | tail -3)
fi
echo ""

echo "=== 3. Code synchen ==="
rsync -avz --exclude='target' --exclude='.git' --exclude='data' --exclude='.env' \
    --exclude='node_modules' --exclude='vscode-extension/out' ./ $SERVER:/lucy/
echo ""

if [ "$1" = "rebuild" ]; then
    echo "=== 4. DOCKER REBUILD (Dockerfile veranderd) ==="
    echo "   LET OP: data volume wordt NIET aangeraakt dankzij named volume"

    # Stop container graceful (triggert shutdown save)
    ssh $SERVER "docker stop lucy --time 30"
    sleep 3

    # Rebuild image + herstart met DEZELFDE named volume
    ssh $SERVER "cd /lucy && docker compose up -d --build"

    echo "=== 5. Wachten op startup ==="
    sleep 20

    # Controleer of data intact is
    VERIFY=$(curl -s --max-time 10 http://209.38.36.51:3000/status 2>/dev/null)
    TICK=$(echo "$VERIFY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tick',0))" 2>/dev/null)

    if [ "$TICK" = "0" ] || [ -z "$TICK" ]; then
        echo ""
        echo "   TICK=0 GEDETECTEERD — data herstellen van backup..."
        ssh $SERVER "VOLPATH=\$($VOLPATH_CMD) && cp /lucy/data-safety-backup/* \$VOLPATH/ && docker restart lucy"
        sleep 15
        echo "   Data hersteld, Lucy herstart"
    fi
else
    echo "=== 4. CODE NAAR CONTAINER (data blijft intact) ==="
    ssh $SERVER "docker cp /lucy/crates lucy:/lucy/crates && \
        docker cp /lucy/Cargo.toml lucy:/lucy/Cargo.toml && \
        docker cp /lucy/Cargo.lock lucy:/lucy/Cargo.lock && \
        docker cp /lucy/lucy-dashboard/dist lucy:/lucy/dashboard"

    echo ""
    echo "=== 5. Lucy hercompileert ==="
    ssh -o ServerAliveInterval=15 $SERVER "docker exec lucy bash -c 'cd /lucy && cargo build --release -p lucy-core 2>&1 | tail -3'"

    echo ""
    echo "=== 6. Stop, vervang binary, herstart ==="
    ssh $SERVER "docker stop lucy --timeout 10 && \
        docker cp lucy:/lucy/target/release/lucy-core /tmp/lucy-core-new && \
        docker cp /tmp/lucy-core-new lucy:/lucy/lucy-core && \
        docker start lucy"
    sleep 15
fi

echo ""
echo "=== VERIFICATIE ==="
VERIFY=$(curl -s --max-time 10 http://209.38.36.51:3000/status 2>/dev/null)
TICK=$(echo "$VERIFY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tick',0))" 2>/dev/null)
VOCAB=$(echo "$VERIFY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('vocabulary',{}).get('size',0))" 2>/dev/null)
PAGES=$(echo "$VERIFY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('explorer',{}).get('pages_read',0))" 2>/dev/null)

if [ -z "$TICK" ]; then
    echo "   Lucy reageert niet — check logs: ssh $SERVER 'docker logs lucy --tail 20'"
elif [ "$TICK" = "0" ]; then
    echo "   TICK=0 — herstel: ssh $SERVER 'VOLPATH=\$($VOLPATH_CMD) && cp /lucy/data-safety-backup/* \$VOLPATH/ && docker restart lucy'"
else
    echo "   Data intact: tick=$TICK vocab=$VOCAB pages=$PAGES"
fi

echo ""
echo "Lucy: https://lucy.world"
