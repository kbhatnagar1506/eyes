// Raw app screenshots for the Devpost gallery, from the local dev server on a
// private hub (so nothing reaches real phones). Expects: vite on :5173, a hub
// on :8799 with dry-run agents, and an incident already raised on it.
// Run: bun scripts/devpost/shoot-app.ts <out dir> <chrome profile dir>

import { launch, Tab } from "./cdp";

const [out, profile] = process.argv.slice(2);
const HUB = "ws://127.0.0.1:8799";
const APP = `http://localhost:5173/?hub=${encodeURIComponent(HUB)}`;
const W = 1640;
const H = 960;
const S = 1.5;
// ONLY=mesh,phones re-shoots just those.
const only = process.env.ONLY?.split(",");
const want = (k: string) => !only || only.includes(k);

// Freeze the Model page: everything revealed, no motion, final numbers.
const STILL = `(() => {
  const st = document.createElement("style");
  st.textContent = "*,*::before,*::after{animation:none!important;transition:none!important} .md-reveal{opacity:1!important;transform:none!important}";
  document.head.appendChild(st);
  document.querySelectorAll(".md-reveal").forEach((e) => e.classList.add("md-in"));
  const v = document.querySelector("#modelView"); if (v) v.style.scrollBehavior = "auto";
  return true;
})()`;
const scrollTo = (sel: string, offset = 0) => `(() => { const v = document.querySelector("#modelView"); const el = document.querySelector("${sel}"); v.scrollTop = el ? el.getBoundingClientRect().top - v.getBoundingClientRect().top + v.scrollTop - ${offset} : 0; return v.scrollTop; })()`;

const chrome = await launch(profile);
try {
  const tab = await Tab.open(chrome.port);
  await tab.size(W, H, S);
  await tab.go(APP, 6000);

  // Dataset tool: the showcase CCTV clip, paused on the attack.
  if (want("dataset")) await tab.eval(`(async () => {
    const v = document.querySelector("#clipVideo");
    for (let i = 0; i < 40 && !(v.readyState >= 2); i++) await new Promise((r) => setTimeout(r, 250));
    v.pause(); v.currentTime = 8.2;
    await new Promise((r) => setTimeout(r, 2500));
    return v.readyState;
  })()`);
  if (want("dataset")) {
    await tab.shot(`${out}/dataset.png`);
    console.log("dataset");
  }

  // Mesh map with the live incident.
  if (want("mesh")) {
    await tab.eval(`document.querySelector("#tabMesh").click()`);
    await Bun.sleep(9000);
    await tab.shot(`${out}/mesh.png`);
    console.log("mesh");
  }

  // Model page sections.
  if (want("model")) {
  await tab.eval(`document.querySelector("#tabModel").click()`);
  await Bun.sleep(5000);
  await tab.eval(STILL);
  await Bun.sleep(600);
  for (const [name, sel, off] of [
    ["model-hero", "", 0],
    ["model-arch", "#mdArch", 70],
    ["model-results", "#mdResultsTop", 70],
    ["model-training", "#mdTraining", 70],
    ["model-curves", "#mdResults", 70],
    ["model-infra", "#mdInfra", 70],
    ["model-flow", "#mdFlow", 70],
  ] as const) {
    // The infrastructure diagram is taller than the screen: a bigger viewport
    // at lower density gives the same 3:2-ish frame with all of it in view.
    if (name === "model-infra") await tab.size(Math.round(W / 0.74), Math.round(H / 0.74), S * 0.74);
    else await tab.size(W, H, S);
    await tab.eval(sel ? scrollTo(sel, off) : `document.querySelector("#modelView").scrollTop = 0`);
    await Bun.sleep(1200);
    await tab.shot(`${out}/${name}.png`);
    console.log(name);
  }
  }
  tab.close();

  // Phones: the officer's alert, then responding; a rider's alert.
  if (want("phones")) {
  const phone = await Tab.open(chrome.port);
  await phone.size(390, 844, 3, true);
  await phone.go(`http://localhost:5173/?role=police&hub=${encodeURIComponent(HUB)}`, 7000);
  await phone.shot(`${out}/phone-officer-alert.png`);
  console.log("phone-officer-alert");
  await phone.eval(`(() => { const b = [...document.querySelectorAll("button")].find((x) => /Respond/.test(x.textContent)); b?.click(); return !!b; })()`);
  await Bun.sleep(5000);
  await phone.eval(`window.scrollTo(0, 0)`);
  await phone.shot(`${out}/phone-officer-responding.png`);
  console.log("phone-officer-responding");
  await phone.go(`http://localhost:5173/?role=rider&hub=${encodeURIComponent(HUB)}`, 7000);
  await phone.shot(`${out}/phone-rider.png`);
  console.log("phone-rider");
  phone.close();
  }
} finally {
  chrome.close();
}
