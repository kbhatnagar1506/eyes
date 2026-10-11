// Real app footage for the demo video, from the local dev server on a private
// hub: the CCTV clip seeked frame by frame (AttackNet's live boxes drawn by
// the app itself) and the Atlanta map recorded while an incident spreads.
// Run: bun scripts/devpost/video/capture.ts <out dir> <chrome profile dir> <frame.jpg>

import { mkdirSync } from "node:fs";
import { launch, Tab } from "../cdp";

const [out, profile, frameJpg] = process.argv.slice(2);
const HUB = "ws://127.0.0.1:8799";
const APP = `http://localhost:5173/?hub=${encodeURIComponent(HUB)}`;
const FPS = 30;
mkdirSync(`${out}/ds`, { recursive: true });
mkdirSync(`${out}/map`, { recursive: true });

const chrome = await launch(profile);
try {
  const tab = await Tab.open(chrome.port);
  await tab.size(1600, 900, 1.2);
  await tab.go(APP, 6000);

  // 1. The CCTV clip: 8 s around the attack, seeked frame by frame.
  const from = 5.5;
  const n = 8 * FPS;
  for (let i = 0; i < n; i++) {
    await tab.eval(`(async () => {
      const v = document.querySelector("#clipVideo");
      v.pause();
      await new Promise((r) => { v.addEventListener("seeked", r, { once: true }); v.currentTime = ${from + i / FPS}; });
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      return true;
    })()`);
    await tab.shot(`${out}/ds/${String(i).padStart(4, "0")}.jpg`, "jpeg", 90);
  }
  console.log(`dataset: ${n} frames`);

  // 2. The map, in real time, while an incident is raised and spreads.
  await tab.eval(`document.querySelector("#tabMesh").click()`);
  await Bun.sleep(9000);
  const stamps: number[] = [];
  const t0 = performance.now();
  let raised = false;
  let polled = false;
  for (let i = 0; performance.now() - t0 < 12000; i++) {
    const t = (performance.now() - t0) / 1000;
    if (!raised && t > 1.2) {
      raised = true;
      Bun.spawn(["bun", "scripts/test/raise.ts", "assault", "-", frameJpg], {
        env: { ...process.env, EYES_HUB: `${HUB}/mesh`, EYES_CAMERA: "CCTV · Student Center East", EYES_DETAIL: "CCTV camera 03 · attack detected" },
        stdout: "ignore",
        stderr: "ignore",
      });
    }
    if (!polled && t > 6) {
      polled = true;
      const f = Bun.file(process.env.PEOPLE_IN!);
      await Bun.write(f, (await f.text()) + "+15550000002 [safe]\n+15550000003 [safe]\n+15550000005 [help]\n");
    }
    await tab.shot(`${out}/map/${String(i).padStart(4, "0")}.jpg`, "jpeg", 88);
    stamps.push(+((performance.now() - t0) / 1000).toFixed(3));
  }
  await Bun.write(`${out}/map/stamps.json`, JSON.stringify(stamps));
  console.log(`map: ${stamps.length} frames over ${stamps.at(-1)} s`);
} finally {
  chrome.close();
}
