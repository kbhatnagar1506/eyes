// Everything the laptop runs for the hosted demo, in one terminal:
//   db     Cloud SQL proxy for eyes-db (127.0.0.1:5434)
//   hub    mesh hub (presence, signaling, relay) on :8787
//   tunnel public wss:// address for the hub (prints the join links)
//   agents Dispatch AI: agents + Photon iMessage + recorder
// Ctrl-C stops all of them.
//
// Run: npm run up

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const PROXY = join(homedir(), ".gbrain/cloud-sql-proxy");
const COLORS: Record<string, string> = { db: "\x1b[90m", hub: "\x1b[36m", tunnel: "\x1b[35m", agents: "\x1b[33m" };

const procs: ReturnType<typeof Bun.spawn>[] = [];

function run(name: string, cmd: string[]) {
  const p = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe", env: process.env });
  procs.push(p);
  const prefix = `${COLORS[name] ?? ""}${name.padEnd(6)}\x1b[0m │ `;
  const pipe = async (s: ReadableStream<Uint8Array>) => {
    const decoder = new TextDecoder();
    let buf = "";
    for await (const chunk of s) {
      buf += decoder.decode(chunk);
      const lines = buf.split("\n");
      buf = lines.pop()!;
      for (const line of lines) if (line.trim()) console.log(prefix + line);
    }
  };
  void pipe(p.stdout);
  void pipe(p.stderr);
  void p.exited.then((code) => console.log(`${prefix}exited (${code})`));
}

// Something already listening (a hub or proxy left running elsewhere) is
// reused rather than crashing on the busy port.
async function busy(port: number): Promise<boolean> {
  try {
    const sock = await Bun.connect({ hostname: "127.0.0.1", port, socket: { data() {} } });
    sock.end();
    return true;
  } catch {
    return false;
  }
}

if (await busy(5434)) console.log("db     │ port 5434 already has a proxy running: using it");
else if (!existsSync(PROXY)) console.log(`db     │ no Cloud SQL proxy at ${PROXY}: recorder will be off`);
else run("db", [PROXY, "--address", "127.0.0.1", "--port", "5434", "patchguard-reakon:us-central1:eyes-db"]);
if (await busy(8787)) console.log("hub    │ port 8787 already has a hub running: using it (stop it first if it's a stale one)");
else run("hub", ["bun", "server/hub.ts"]);
await Bun.sleep(1500);
run("tunnel", ["bun", "server/tunnel.ts"]);
run("agents", ["bun", "server/agents.ts"]);

process.on("SIGINT", () => {
  for (const p of procs) p.kill();
  process.exit(0);
});
await Promise.all(procs.map((p) => p.exited));
