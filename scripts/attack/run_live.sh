#!/bin/bash
# Runs on the GCP NVIDIA T4 VM (eyes-attacknet-live): live AttackNet behind a
# public wss:// address. Prints the address the laptop connects to.
set -euo pipefail
B=gs://eyes-hacklanta-media
W=~/eyes
mkdir -p $W/scripts/attack $W/data/attack/out
cd $W
gcloud storage cp "$B/jobs/attack/extract.py" "$B/jobs/attack/train.py" "$B/jobs/attack/live_server.py" scripts/attack/ >/dev/null
gcloud storage cp "$B/attack/out/attacknet.pt" data/attack/out/ >/dev/null
sudo apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
python3 -m pip install -q ultralytics==8.3.57 "websockets>=12"
if [ ! -x ~/cloudflared ]; then
  curl -sL -o ~/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x ~/cloudflared
fi
pkill -f live_server.py || true
pkill -f "cloudflared tunnel" || true
cd scripts/attack
nohup python3 -u live_server.py > ~/live.log 2>&1 < /dev/null &
until grep -q "AttackNet live" ~/live.log; do sleep 1; if ! pgrep -f live_server.py >/dev/null; then cat ~/live.log; exit 1; fi; done
nohup ~/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765 > ~/tunnel.log 2>&1 < /dev/null &
until grep -qo "https://[a-z0-9-]*\.trycloudflare\.com" ~/tunnel.log; do sleep 1; done
echo "ATTACKNET_URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" ~/tunnel.log | head -1 | sed 's#^https#wss#')"
