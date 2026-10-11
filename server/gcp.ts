// GCP access for the laptop services (agents, recorder). gcloud's default
// account here is the project's service account, so this never prompts.

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// A shell started without the user's profile may not have gcloud on PATH.
const GCLOUD = [process.env.GCLOUD, join(homedir(), "google-cloud-sdk/bin/gcloud"), "/opt/homebrew/bin/gcloud", "/usr/local/bin/gcloud"].find(
  (p) => p && existsSync(p),
) ?? "gcloud";

let token: { value: string; until: number } | null = null;

export function accessToken(): string {
  if (token && token.until > Date.now()) return token.value;
  const out = Bun.spawnSync([GCLOUD, "auth", "print-access-token"]);
  if (out.exitCode !== 0) throw new Error(`gcloud token: ${out.stderr.toString().trim()}`);
  token = { value: out.stdout.toString().trim(), until: Date.now() + 45 * 60_000 };
  return token.value;
}

export const PROJECT_ID = process.env.EYES_GCP_PROJECT ?? "patchguard-reakon";
export const MEDIA_BUCKET = process.env.EYES_MEDIA_BUCKET ?? "eyes-hacklanta-media";

// Private upload to Cloud Storage (the bucket enforces public access prevention).
export async function uploadObject(name: string, body: Blob, contentType: string): Promise<string> {
  const url = `https://storage.googleapis.com/upload/storage/v1/b/${MEDIA_BUCKET}/o?uploadType=media&name=${encodeURIComponent(name)}`;
  const resp = await fetch(url, { method: "POST", headers: { Authorization: `Bearer ${accessToken()}`, "Content-Type": contentType }, body });
  if (!resp.ok) throw new Error(`GCS upload ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
  return `gs://${MEDIA_BUCKET}/${name}`;
}
