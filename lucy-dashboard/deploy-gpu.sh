#!/bin/bash
# Deploy lucy-gpu naar Vast.ai GPU instance
# Gebruik: bash deploy-gpu.sh [instance-id]
#
# Stappen:
# 1. Cross-compile lucy-gpu voor Linux
# 2. Upload binary + shader naar Vast.ai instance
# 3. Start de GPU service

set -e

# Vast.ai instance info
INSTANCE_ID=${1:-""}

if [ -z "$INSTANCE_ID" ]; then
    echo "Zoek actieve Vast.ai instance..."
    INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
instances = json.load(sys.stdin)
running = [i for i in instances if i.get('cur_state') == 'running']
if running:
    print(running[0]['id'])
" 2>/dev/null)

    if [ -z "$INSTANCE_ID" ]; then
        echo "Geen actieve Vast.ai instance gevonden."
        echo ""
        echo "Start een instance:"
        echo "  vastai search offers 'gpu_name=RTX_3060 num_gpus=1 rentable=true' --order 'dph_total' --limit 5"
        echo "  vastai create instance <ID> --image 'nvidia/cuda:12.4.1-devel-ubuntu22.04' --disk 20"
        exit 1
    fi
fi

echo "=== Instance: $INSTANCE_ID ==="

# Haal SSH info op
SSH_INFO=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
instances = json.load(sys.stdin)
for i in instances:
    if i['id'] == $INSTANCE_ID:
        print(f\"{i['ssh_host']} {i['ssh_port']}\")
        break
")

SSH_HOST=$(echo $SSH_INFO | cut -d' ' -f1)
SSH_PORT=$(echo $SSH_INFO | cut -d' ' -f2)

if [ -z "$SSH_HOST" ]; then
    echo "Kan SSH info niet vinden voor instance $INSTANCE_ID"
    exit 1
fi

echo "SSH: $SSH_HOST:$SSH_PORT"

# Cross-compile voor Linux (als we op macOS zitten)
if [[ "$(uname)" == "Darwin" ]]; then
    echo "=== Cross-compiling voor Linux x86_64 ==="
    # Check of cross-compile target beschikbaar is
    if ! rustup target list --installed | grep -q x86_64-unknown-linux-gnu; then
        echo "Installeer Linux cross-compile target..."
        rustup target add x86_64-unknown-linux-gnu
    fi

    # Gebruik cross als beschikbaar, anders cargo
    if command -v cross &> /dev/null; then
        cross build --release --package lucy-gpu --target x86_64-unknown-linux-gnu
        BINARY="target/x86_64-unknown-linux-gnu/release/lucy-gpu"
    else
        echo "OPMERKING: cross-compilation op macOS vereist 'cross' tool"
        echo "  cargo install cross"
        echo ""
        echo "Alternatief: compileer op de Vast.ai machine zelf"
        echo "Upload source en compileer remote..."

        # Upload source en compileer remote
        echo "=== Upload source naar Vast.ai ==="
        rsync -avz --progress \
            -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
            --exclude target \
            --exclude data \
            --exclude .git \
            . root@$SSH_HOST:/root/lucy/

        echo "=== Compileer op Vast.ai ==="
        ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST \
            "source \$HOME/.cargo/env 2>/dev/null; cd /root/lucy && cargo build --release --package lucy-gpu 2>&1" | tail -20

        echo "=== Start lucy-gpu service ==="
        ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST \
            "source \$HOME/.cargo/env 2>/dev/null; cd /root/lucy && pkill -f lucy-gpu 2>/dev/null; nohup target/release/lucy-gpu > /tmp/lucy-gpu.log 2>&1 &"

        sleep 2

        # Test
        echo "=== Test GPU service ==="
        ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST \
            "curl -s http://localhost:3001/status"

        echo ""
        echo "=== GPU service draait op Vast.ai ==="
        echo "SSH tunnel voor toegang vanaf DO:"
        echo "  ssh -L 3001:localhost:3001 -p $SSH_PORT root@$SSH_HOST"
        echo ""
        echo "Of stel GPU_URL in op Lucy (DO):"
        echo "  GPU_URL=http://$SSH_HOST:3001"
        exit 0
    fi
else
    echo "=== Building lucy-gpu ==="
    cargo build --release --package lucy-gpu
    BINARY="target/release/lucy-gpu"
fi

echo "=== Upload binary ==="
scp -o StrictHostKeyChecking=no -P $SSH_PORT \
    $BINARY root@$SSH_HOST:/root/lucy-gpu

scp -o StrictHostKeyChecking=no -P $SSH_PORT \
    crates/lucy-gpu/src/gpu_compute.wgsl root@$SSH_HOST:/root/

echo "=== Start lucy-gpu service ==="
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST \
    "pkill -f lucy-gpu 2>/dev/null; chmod +x /root/lucy-gpu && nohup /root/lucy-gpu > /tmp/lucy-gpu.log 2>&1 &"

sleep 2

echo "=== Test ==="
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST \
    "curl -s http://localhost:3001/status"

echo ""
echo "Done! GPU service draait op $SSH_HOST:$SSH_PORT"
