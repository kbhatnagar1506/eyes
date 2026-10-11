// Where the hosted site's media lives. Locally: files in public/. On Vercel:
// the private media bucket, through the signed links in /media.json
// (scripts/sign_media.sh). The deployment itself carries no data or video.

interface Media {
  expires: string;
  data: Record<string, string>;
  clips: Record<string, string>;
  densepose?: Record<string, string>;
  attack?: Record<string, string>;
}

let manifest: Promise<Media | null> | null = null;

function media(): Promise<Media | null> {
  // Never cached: re-signed links must reach every device right away.
  return (manifest ??= fetch("/media.json", { cache: "no-store" })
    .then((r) => (r.ok ? (r.json() as Promise<Media>) : null))
    .catch(() => null));
}

export async function dataUrl(name: "marta" | "places"): Promise<string> {
  if (!import.meta.env.PROD) return `/data/${name}.json`;
  return (await media())?.data[name] ?? `/data/${name}.json`;
}

// Hosted: only clips that have a signed link (stock footage). Research clips
// and recordings never leave the laptop.
export async function clipUrl(file: string): Promise<string | null> {
  if (!import.meta.env.PROD) return `/demo-clips/${file}`;
  return (await media())?.clips[file] ?? null;
}

// NVIDIA GPU DensePose renders, detector scores and tracking summaries: always
// from the media bucket (they never lived in the repo's public folder).
export async function denseposeUrl(file: string): Promise<string | null> {
  return (await media())?.densepose?.[file] ?? null;
}

// AttackNet output per clip (people, boxes, attack probabilities) and its
// evaluation metrics, from the GPU training run.
export async function attackUrl(file: string): Promise<string | null> {
  return (await media())?.attack?.[file] ?? null;
}
