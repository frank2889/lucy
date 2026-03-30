#!/bin/bash
# Lucy — Pull haar server-aanpassingen terug naar lokaal
# Gebruik: bash pull.sh

SERVER="root@209.38.36.51"

echo "=== Lucy's code ophalen van server ==="

# 1. Kopieer code UIT de container naar de server
ssh $SERVER "docker cp lucy:/lucy/crates /lucy/crates-live && docker cp lucy:/lucy/Cargo.toml /lucy/Cargo.toml.live"

# 2. Sync terug naar lokaal
rsync -avz $SERVER:/lucy/crates-live/ ./crates/
rsync -avz $SERVER:/lucy/Cargo.toml.live ./Cargo.toml

# 3. Toon wat Lucy veranderd heeft
echo ""
echo "=== Lucy's wijzigingen ==="
git diff --stat
echo ""
echo "Details:"
git diff --name-only

echo ""
echo "Review met: git diff"
echo "Accepteer met: git add -A && git commit -m 'Lucy self-modification'"
echo "Verwerp met: git checkout ."
