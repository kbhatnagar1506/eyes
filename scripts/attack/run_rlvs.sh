#!/bin/bash
# Runs on the GCP NVIDIA T4 VM (eyes-gpu-train): AttackNet v2.
# Downloads Real Life Violence Situations (public, Hugging Face mirror),
# tracks every person in it with YOLOv8x-pose on the GPU, pretrains on it,
# fine-tunes on our clips, and pushes results to the private bucket.
set -euo pipefail
B=gs://eyes-hacklanta-media
W=~/eyes
mkdir -p $W/scripts/attack $W/data/attack/out ~/rlvs
cd $W
gcloud storage cp "$B/jobs/attack/*" scripts/attack/
mv scripts/attack/labels.json data/attack/labels.json
gcloud storage cp "$B/attack/tracks/*" data/attack/
gcloud storage cp "$B/attack/out-v1/metrics.json" data/attack/out/
sudo apt-get install -y -qq libgl1 libglib2.0-0 unzip >/dev/null
python3 -m pip install -q ultralytics==8.3.57 scikit-learn "huggingface_hub[cli]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f ~/rlvs/.unzipped ]; then
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('34data/real-life-violence', repo_type='dataset', local_dir='$HOME/rlvs')"
  (cd ~/rlvs && for z in *.zip; do unzip -q -o "$z"; done && touch .unzipped)
fi
python3 -u scripts/attack/extract_rlvs.py ~/rlvs
cd scripts/attack && python3 -u train_rlvs.py && cd $W
gcloud storage cp data/attack/out/* "$B/attack/out/"
gcloud storage cp -r data/attack/rlvs "$B/attack/rlvs-tracks" >/dev/null 2>&1 || true
echo JOB_DONE
