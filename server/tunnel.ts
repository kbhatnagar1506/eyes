// Gives the laptop's mesh hub a public, secure (wss://) address through a
// Cloudflare quick tunnel, so the hosted site (Vercel) and phones anywhere
// can reach it. Prints the join links; the hub address rides in them as
// ?hub=…, and every device remembers it.
//
// Run: npm run tunnel   (with the hub already running: npm run hub)

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const SITE = (process.env.EYES_SITE ?? "https://eyes-hacklanta.vercel.app").replace(/\/+$/, "");
const HUB_PORT = Number(process.env.EYES_HUB_PORT ?? 8787);

const candidates = [process.env.CLOUDFLARED, join(homedir(), ".local/bin/cloudflared"), "/opt/homebrew/bin/cloudflared", "/usr/local/bin/cloudflared"];
const bin = candidates.find((p) => p && existsSync(p)) ?? "cloudflared";

const proc = Bun.spawn([bin, "tunnel", "--no-autoupdate", "--url", `http://127.0.0.1:${HUB_PORT}`], { stdout: "pipe", stderr: "pipe" });

let printed = false;
async function watch(stream: ReadableStream<Uint8Array>) {
  const decoder = new TextDecoder();
  for await (const chunk of stream) {
    const text = decoder.decode(chunk);
    const m = text.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/);
    if (m && !printed) {
      printed = true;
      const hub = m[0].replace(/^https/, "wss");
      const link = (q: string) => `${SITE}/?${q}hub=${encodeURIComponent(hub)}`;
      console.log(`\nMesh hub is public at ${hub}\n`);
      console.log(`Console (laptop): ${link("")}`);
      console.log(`Police phone:     ${link("role=police&")}`);
      console.log(`Driver phone:     ${link("role=driver&")}`);
      console.log(`Rider phone:      ${link("role=rider&")}`);
      console.log(`\nOr open the console and use Join on the map for QR codes. Ctrl-C stops the tunnel.\n`);
    }
  }
}
void watch(proc.stdout);
void watch(proc.stderr);

process.on("SIGINT", () => {
  proc.kill();
  process.exit(0);
});
await proc.exited;
