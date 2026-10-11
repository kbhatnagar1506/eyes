#!/bin/bash
# Runs on the GCP NVIDIA T4 VM (eyes-gpu-train): pulls clips from the private
# media bucket into the repo layout, detects + tracks people with
# YOLOv8x-pose on the GPU, trains AttackNet, and pushes results back.
set -euo pipefail
B=gs://eyes-hacklanta-media
W=~/eyes
mkdir -p $W/public/demo-clips $W/sessions $W/densepose-output $W/densepose-multi $W/data/attack $W/scripts/attack
cd $W
gcloud storage cp "$B/jobs/attack/*" scripts/attack/
mv scripts/attack/labels.json data/attack/labels.json
gcloud storage cp "$B/clips/stock/*" "$B/clips/ucf-crime-private/*" public/demo-clips/
gcloud storage cp "$B/sessions-private/gf-punch-1.mov" sessions/
gcloud storage cp "$B/densepose/stock/*" densepose-output/
gcloud storage cp "$B/densepose/cctv/*_multi_tracked.mp4" densepose-multi/
# OpenCV needs libGL even headless.
sudo apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
python3 -m pip install -q ultralytics==8.3.57 scikit-learn
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u scripts/attack/extract.py
python3 -u scripts/attack/train.py
gcloud storage cp data/attack/*_tracks.json "$B/attack/tracks/"
gcloud storage cp data/attack/out/* "$B/attack/out/"
echo JOB_DONE
