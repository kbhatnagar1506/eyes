#!/bin/zsh
# Signs 7-day read links for the hosted site's media (map datasets + stock
# clips) in the private bucket and writes them to public/media.json, the only
# media file that ships with the Vercel deployment. Re-run (then redeploy)
# before the links expire. Research clips and recordings are never signed.
set -euo pipefail
source ~/.zshrc >/dev/null 2>&1 || true
cd "$(dirname "$0")/.."
BUCKET=gs://eyes-hacklanta-media
OBJECTS=(
  data/marta.json data/places.json
  clips/stock/sparring_6296158.mp4 clips/stock/woman_sparring_6296385.mp4 clips/stock/boxing_6296696.mp4
  clips/stock/sparring2_6296587.mp4 clips/stock/busstop_calm_36467681.mp4
  # NVIDIA T4 DensePose renders (stock + real CCTV), detector scores, tracking summaries
  $(gcloud storage ls "$BUCKET/densepose/stock/" "$BUCKET/densepose/cctv/" "$BUCKET/densepose/scores/" | grep -E '\.(mp4|json)$' | sed "s#^$BUCKET/##")
  # AttackNet output per clip + evaluation metrics (GPU training run)
  $(gcloud storage ls "$BUCKET/attack/out/" 2>/dev/null | grep -E '\.json$' | sed "s#^$BUCKET/##")
)
URLS=()
for o in $OBJECTS; do
  URLS+=("$o" "$(gcloud storage sign-url "$BUCKET/$o" --duration=7d --format='value(signed_url)')")
done
python3 - "$URLS[@]" <<'PY'
import json, sys, datetime
args = sys.argv[1:]
pairs = dict(zip(args[0::2], args[1::2]))
out = {
  "expires": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)).isoformat(timespec="minutes"),
  "data": {k.split("/")[-1].removesuffix(".json"): v for k, v in pairs.items() if k.startswith("data/")},
  "clips": {k.split("/")[-1]: v for k, v in pairs.items() if k.startswith("clips/")},
  "densepose": {k.split("/")[-1]: v for k, v in pairs.items() if k.startswith("densepose/")},
  "attack": {k.split("/")[-1]: v for k, v in pairs.items() if k.startswith("attack/")},
}
open("public/media.json", "w").write(json.dumps(out, indent=1))
print(f"public/media.json: {len(out['data'])} datasets, {len(out['clips'])} clips, {len(out['densepose'])} DensePose files, links expire {out['expires']}")
PY
