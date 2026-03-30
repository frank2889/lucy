#!/bin/bash
# Lucy v0.9+ — Veilige Deploy naar DigitalOcean
# Gebruik: bash deploy.sh          (snel — alleen code sync + hercompileer)
#          bash deploy.sh full      (volledig — Docker rebuild)
#
# ALTIJD veilig: triggert pre-save voor deploy, atomic writes, graceful shutdown

SERVER="root@209.38.36.51"
TOKEN="${LUCY_API_TOKEN:-}"

if [ -z "$TOKEN" ]; then
    echo "⚠️  LUCY_API_TOKEN niet gezet. Beveiligde endpoints niet beschikbaar."
    echo "   export LUCY_API_TOKEN=jouw-token"
fi

echo "=== 1. Pre-deploy save triggeren ==="
if [ -n "$TOKEN" ]; then
    SAVE_RESULT=$(curl -s --max-time 15 -X POST http://209.38.36.51:3000/admin/pre-deploy \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    echo "   $SAVE_RESULT"
else
    echo "   (geen token — skip pre-save, data wordt opgeslagen bij shutdown)"
fi

echo ""
echo "=== 2. Wachten op save ==="
sleep 3

echo ""
echo "=== 3. Dashboard bouwen ==="
if [ -d "lucy-dashboard" ]; then
    (cd lucy-dashboard && npm run build 2>&1)
    echo "   Dashboard gebouwd"
fi

echo ""
echo "=== 4. Code synchen ==="
rsync -avz --exclude='target' --exclude='.git' --exclude='data' --exclude='.env' --exclude='node_modules' --exclude='vscode-extension/out' ./ $SERVER:/lucy/

if [ "$1" = "full" ]; then
    echo ""
    echo "=== 5. VOLLEDIGE DEPLOY (Docker rebuild) ==="
    ssh $SERVER "cd /lucy && docker compose up -d --build"
    echo ""
    echo "=== 6. Wachten op startup ==="
    sleep 15
else
    echo ""
    echo "=== 5. SNELLE DEPLOY (code naar container + hercompileer) ==="
    ssh $SERVER "docker cp /lucy/crates lucy:/lucy/crates && docker cp /lucy/Cargo.toml lucy:/lucy/Cargo.toml && docker cp /lucy/Cargo.lock lucy:/lucy/Cargo.lock && docker cp /lucy/lucy-dashboard/dist lucy:/lucy/dashboard"

    echo ""
    echo "=== 6. Lucy hercompileert zichzelf ==="
    if [ -n "$TOKEN" ]; then
        curl -s -X POST http://209.38.36.51:3000/restart \
            -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
    else
        curl -s -X POST http://209.38.36.51:3000/restart | python3 -m json.tool
    fi
fi

echo ""
echo "=== 7. Verify ==="
sleep 3
curl -s http://209.38.36.51:3000/status | python3 -m json.tool

echo ""
echo "Lucy draait op: http://209.38.36.51:3000"
echo "  /brain        — 3D brain dashboard"
echo "  /status       — live stats + emoties"
echo "  /telemetry    — compact real-time data"
echo "  /topology     — network topology"
echo "  /ask          — stel een vraag"
echo "  /feed         — trainingsdata"
echo "  /self-modify  — zelf-aanpassing (auth vereist)"
echo "  /exec         — shell commando (auth vereist, whitelist)"
echo "  /restart      — hercompileren (auth vereist)"
