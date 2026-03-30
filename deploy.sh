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
echo "=== 3. Code synchen ==="
rsync -avz --exclude='target' --exclude='.git' --exclude='data' --exclude='.env' --exclude='node_modules' --exclude='vscode-extension/out' ./ $SERVER:/lucy/

if [ "$1" = "full" ]; then
    echo ""
    echo "=== 4. VOLLEDIGE DEPLOY (Docker rebuild) ==="
    ssh $SERVER "cd /lucy && docker compose up -d --build"
    echo ""
    echo "=== 5. Wachten op startup ==="
    sleep 15
else
    echo ""
    echo "=== 4. SNELLE DEPLOY (code naar container + hercompileer) ==="
    ssh $SERVER "docker cp /lucy/crates lucy:/lucy/crates && docker cp /lucy/Cargo.toml lucy:/lucy/Cargo.toml && docker cp /lucy/Cargo.lock lucy:/lucy/Cargo.lock"

    echo ""
    echo "=== 5. Lucy hercompileert zichzelf ==="
    if [ -n "$TOKEN" ]; then
        curl -s -X POST http://209.38.36.51:3000/restart \
            -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
    else
        curl -s -X POST http://209.38.36.51:3000/restart | python3 -m json.tool
    fi
fi

echo ""
echo "=== 6. Verify ==="
sleep 3
curl -s http://209.38.36.51:3000/status | python3 -m json.tool

echo ""
echo "Lucy draait op: http://209.38.36.51:3000"
echo "  /status       — live stats + emoties"
echo "  /ask          — stel een vraag"
echo "  /feed         — trainingsdata"
echo "  /self-modify  — zelf-aanpassing (auth vereist)"
echo "  /exec         — shell commando (auth vereist, whitelist)"
echo "  /restart      — hercompileren (auth vereist)"
