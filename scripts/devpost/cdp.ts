// A tiny Chrome DevTools Protocol driver for the Devpost gallery: launches
// the local Chrome headless, opens pages at an exact size and pixel density,
// runs page scripts, and saves screenshots. No extra packages.

import { mkdirSync } from "node:fs";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

export async function launch(profileDir: string, port = 9333) {
  mkdirSync(profileDir, { recursive: true });
  const proc = Bun.spawn(
    [CHROME, "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profileDir}`, "--hide-scrollbars", "--no-first-run", "--autoplay-policy=no-user-gesture-required", "--allow-file-access-from-files", "--disable-features=Translate", "about:blank"],
    { stdout: "ignore", stderr: "ignore" },
  );
  for (let i = 0; i < 50; i++) {
    try {
      await fetch(`http://127.0.0.1:${port}/json/version`);
      break;
    } catch {
      await Bun.sleep(200);
    }
  }
  return { port, close: () => proc.kill() };
}

export class Tab {
  private ws!: WebSocket;
  private id = 0;
  private waiting = new Map<number, (v: any) => void>();

  static async open(port: number) {
    const t = new Tab();
    const target = (await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json()) as { webSocketDebuggerUrl: string };
    t.ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((r) => (t.ws.onopen = r));
    t.ws.onmessage = (e) => {
      const msg = JSON.parse(String(e.data));
      if (msg.id && t.waiting.has(msg.id)) {
        t.waiting.get(msg.id)!(msg);
        t.waiting.delete(msg.id);
      }
    };
    await t.send("Page.enable");
    await t.send("Runtime.enable");
    return t;
  }

  send(method: string, params: Record<string, unknown> = {}): Promise<any> {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((r) => this.waiting.set(id, r));
  }

  async size(width: number, height: number, scale: number, mobile = false) {
    await this.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: scale, mobile });
  }

  async go(url: string, settleMs = 2500) {
    await this.send("Page.navigate", { url });
    await Bun.sleep(settleMs);
  }

  async eval<T = unknown>(expression: string): Promise<T> {
    const r = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 300));
    return r.result?.result?.value as T;
  }

  async shot(path: string, format: "png" | "jpeg" = "png", quality = 92) {
    const r = await this.send("Page.captureScreenshot", { format, quality: format === "jpeg" ? quality : undefined, captureBeyondViewport: false });
    await Bun.write(path, Buffer.from(r.result.data, "base64"));
  }

  close() {
    this.ws.close();
  }
}
