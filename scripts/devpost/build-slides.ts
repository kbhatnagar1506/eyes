// Devpost gallery: 3:2 slides (1800×1200 CSS px, rendered at 1.5× = 2700×1800)
// framing real app screenshots, plus a title slide and two iMessage slides.
// Run: bun scripts/devpost/build-slides.ts <chrome profile dir>
// Reads devpost/raw/*.png, writes devpost/slides.html and devpost/gallery/*.jpg.

import { mkdirSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { logoSvg } from "../../src/logo";
import { launch, Tab } from "./cdp";

const ROOT = resolve(import.meta.dir, "../..");
const RAW = `${ROOT}/devpost/raw`;
const OUT = `${ROOT}/devpost/gallery`;
const FRAME_JPG = process.env.FRAME_JPG ?? "";

type Slide = { file: string; kicker: string; title: string; sub: string; body: string };

const shot = (name: string) => `<div class="shot"><img src="${RAW}/${name}.png" alt=""/></div>`;
const phones = (...names: string[]) => `<div class="phones">${names.map((n) => `<div class="phone"><img src="${RAW}/${n}.png" alt=""/></div>`).join("")}</div>`;

// iMessage-style conversation (our own product's messages, as an illustration).
const bubble = (from: "eyes" | "me", html: string, extra = "") => `<div class="b ${from} ${extra}">${html}</div>`;
const imsg = (title: string, rows: string) => `
  <div class="im">
    <div class="im-top"><div class="im-av">${logoSvg(40)}</div><div class="im-name">Eyes</div><div class="im-sub">iMessage · Photon Spectrum</div></div>
    <div class="im-body">${rows}</div>
    <div class="im-cap">${title}</div>
  </div>`;
const nl = (s: string) => s.replace(/\n/g, "<br/>");

const cardText = `🚨 <b>Assault reported near GSU Student Center East</b>
12:49 AM · spotted by an Eyes camera

⚠️ Be careful around GSU Student Center East. Avoid the area and don't step in; stay in busy, well-lit places.

Help is on the way:
🚓 GSU Police Department · about 2 min · 🟢 on the way
🚑 Grady Health System · about 2 min · paged
👥 5 people within 800 m warned · 2 safe · 1 needs help
📸 1 camera frame sent to police`;
const briefText = `👁 Sentinel: Assault detected at GSU Student Center East at 12:49 AM.
🛰 Network: Delivered over internet + P2P Wi-Fi + mesh radio to 3 devices.
🧭 Dispatch: Paging GSU Police (2 min) to secure the scene, Grady Health (2 min) for medical support.
🧍 Safety: If you are near GSU Student Center East, move away. Help is coming.`;

const slides: Slide[] = [
  { file: "01-eyes", kicker: "", title: "", sub: "", body: "" },
  {
    file: "02-detect",
    kicker: "01 · Detect",
    title: "AttackNet flags an attack on real CCTV",
    sub: "Body language for every person in the frame. A box turns red only above 30%. 83% F1 on this clip, which the model never trained on.",
    body: shot("dataset"),
  },
  {
    file: "03-spread",
    kicker: "02 · Spread",
    title: "The alert crosses Atlanta in seconds, even offline",
    sub: "130 city mesh nodes: police, hospitals, MARTA stations and live buses. A networking agent sends every message over the best link: internet, P2P Wi-Fi or mesh radio.",
    body: shot("mesh"),
  },
  {
    file: "04-respond",
    kicker: "03 · Respond",
    title: "Responders see the exact camera frame",
    sub: "Officer, bus driver and rider screens join by QR. The alert, the evidence, the distance and the map, then one tap to respond.",
    body: phones("phone-officer-alert", "phone-officer-responding", "phone-rider"),
  },
  {
    file: "05-coordinate",
    kicker: "04 · Coordinate",
    title: "AI agents coordinate the response over iMessage",
    sub: "Built on Photon Spectrum. One live status card, edited in place as responders move. The agents' brief threaded underneath. Tapbacks instead of noise.",
    body: `<div class="ims">${imsg(
      "Responders' thread",
      bubble("eyes", nl(cardText), "card") +
        `<div class="edited">Edited</div>` +
        bubble("eyes", nl(briefText), "thread") +
        (FRAME_JPG ? bubble("eyes", `<img class="ph" src="${FRAME_JPG}"/><div class="phc">📸 Evidence #1 · GSU Student Center East</div>`, "thread photo") : "") +
        bubble("me", "on my way", "tap") ,
    )}
    <div class="ims-notes">
      <div class="note"><b>Calm by default</b>One card per incident, edited in place: an officer confirming or people checking in update it without another buzz.</div>
      <div class="note"><b>Multi-agent, grounded</b>Sentinel, Network, Dispatch, Police, EMS, Safety and Care each see the others' lines and only state facts: nobody is "on the way" until a human says so.</div>
      <div class="note"><b>Humans in the loop</b>Reply "on my way" and the card turns green with a 👍. Ask anything and get an answer from the live incident.</div>
    </div></div>`,
  },
  {
    file: "06-protect",
    kicker: "05 · Protect",
    title: "Anyone can report. Everyone nearby is warned.",
    sub: "Text what's happening and where: Eyes raises the incident. People signed up within 800 m get a loud alert, a safe spot and an \"Are you safe?\" poll. Anyone hurt gets care until EMS arrives.",
    body: `<div class="ims two">${imsg(
      "Someone reports",
      bubble("me", "there's a fight at Woodruff Park") +
        bubble("eyes", nl("🚨 Got it. I've raised an alert for assault near Woodruff Park.\n\nPaging GSU Police Department (about 2 min) and Grady Health System (about 2 min) now, and warning people nearby.\n\nIf anyone is in danger right now, call 911.")) +
        `<div class="poll"><b>Are you safe?</b><span>I'm safe</span><span>I need help</span></div>` +
        bubble("me", "I'm safe", "heart"),
    )}${imsg(
      "Someone nearby",
      bubble("eyes", nl("⚠️ <b>EYES SAFETY ALERT</b>\nAssault reported at Woodruff Park, 420 m south-east of you.\nAvoid the area and don't step in; stay in busy, well-lit places.\nNearest safe spot: Atlanta Central Library, 240 m from you."), "loud") +
        `<div class="poll"><b>Are you safe?</b><span>I'm safe</span><span class="on">I need help ✓</span></div>` +
        bubble("eyes", nl("🆘 Help is coming to Five Points. An ambulance from Grady Health System has been requested (about 3 min).\n\nIf anyone isn't breathing or is badly hurt, call 911 now.")) +
        bubble("me", "my friend is bleeding from his head") +
        bubble("eyes", "I've passed this to the EMS crew. Apply firm, direct pressure with a clean cloth. Is he conscious?"),
    )}</div>`,
  },
  {
    file: "07-model",
    kicker: "06 · The model",
    title: "Detect. Dispatch. Act.",
    sub: "AttackNet, our main model: 1.9M weights reading 84 body-language signals per person, trained on a Google Cloud NVIDIA T4.",
    body: shot("model-hero"),
  },
  {
    file: "08-architecture",
    kicker: "07 · Architecture",
    title: "Pose, body language, then two transformers",
    sub: "YOLOv8x-pose finds every person's joints. A temporal transformer reads each person's last 2 seconds; a people transformer lets everyone in the frame attend to each other.",
    body: shot("model-arch"),
  },
  {
    file: "09-results",
    kicker: "08 · Results",
    title: "92% right on 400 clips it never saw",
    sub: "AUC 0.97 on held-out real-world clips. On our hand-labeled CCTV, F1 63% and AUC 0.72, up from v1. Every number is on footage the model never trained on.",
    body: shot("model-results"),
  },
  {
    file: "10-training",
    kicker: "09 · Training",
    title: "Trained on 1,400 real-world fight and normal clips",
    sub: "Loss falls while held-out accuracy climbs to 95%. We also tested adapting it to our CCTV; it scored worse on unseen clips, so we didn't ship it.",
    body: shot("model-training"),
  },
  {
    file: "11-evaluation",
    kicker: "10 · Evaluation",
    title: "ROC, precision-recall and confusion, all on unseen footage",
    sub: "Real-world clips held back from training, our own CCTV scored one clip at a time, and sparring clips kept out entirely.",
    body: shot("model-curves"),
  },
  {
    file: "12-infrastructure",
    kicker: "11 · Infrastructure",
    title: "From the camera to an iMessage on a phone",
    sub: "Vercel, a Bun mesh hub behind a Cloudflare tunnel, Gemini 2.5 Flash on Vertex AI, Photon Spectrum, Cloud SQL Postgres, Cloud Storage and an NVIDIA T4 GPU.",
    body: shot("model-infra"),
  },
  {
    file: "13-flow",
    kicker: "12 · Response flow",
    title: "From danger on camera to help on the way",
    sub: "Detect, spread, coordinate, notify, protect, close. Every step recorded in Postgres and the media bucket.",
    body: shot("model-flow"),
  },
];

const titleSlide = `
  <section class="slide title">
    <div class="glow"></div>
    <div class="t-in">
      <div class="t-logo">${logoSvg(330)}</div>
      <div>
        <div class="t-kick">Hacklanta 2026</div>
        <h1>Eyes</h1>
        <div class="t-tag">Cameras that watch. <span>Eyes that act.</span></div>
        <p>A real-time safety network for public places. Cameras detect danger, a mesh carries the alert even offline, and AI agents dispatch help and warn people nearby over iMessage.</p>
        <div class="chips"><span>👁 Detect · Dispatch · Act</span><span>🕸 Mesh that works offline</span><span>💬 AI agents on iMessage</span><span>⚡ Trained on an NVIDIA T4</span></div>
      </div>
    </div>
  </section>`;

const html = `<!doctype html><html><head><meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box} html,body{margin:0;background:#07090d}
.slide{position:relative;width:1800px;height:1200px;overflow:hidden;color:#e7e9ee;font-family:Inter,system-ui,sans-serif;
  background:radial-gradient(45% 55% at 12% 0%,rgba(45,212,191,.16),transparent 70%),radial-gradient(40% 50% at 95% 10%,rgba(244,63,94,.13),transparent 70%),radial-gradient(40% 40% at 60% 110%,rgba(96,165,250,.12),transparent 70%),
  linear-gradient(rgba(255,255,255,.022) 1px,transparent 1px) 0 0/48px 48px,linear-gradient(90deg,rgba(255,255,255,.022) 1px,transparent 1px) 0 0/48px 48px,#07090d}
.brand{position:absolute;top:44px;left:120px;display:flex;align-items:center;gap:12px;font:700 24px/1 "Space Grotesk";letter-spacing:-.02em}
.idx{position:absolute;top:52px;right:120px;font:600 15px/1 "JetBrains Mono";color:#5b6478}
.head{position:absolute;top:110px;left:120px;right:120px}
.kick{font:600 16px/1 "JetBrains Mono";letter-spacing:.1em;text-transform:uppercase;color:#2dd4bf}
h2{font:700 54px/1.05 "Space Grotesk";letter-spacing:-.03em;margin:14px 0 12px}
.sub{font-size:21px;line-height:1.45;color:#a3abbd;max-width:1420px;margin:0}
.body{position:absolute;left:120px;right:120px;top:330px;bottom:56px;display:flex;align-items:center;justify-content:center}
.shot{height:100%;aspect-ratio:1640/960;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.12);box-shadow:0 40px 90px -30px rgba(0,0,0,.9),0 0 0 6px rgba(255,255,255,.03),0 0 80px -20px rgba(45,212,191,.35)}
.shot img{width:100%;height:100%;object-fit:cover;object-position:top;display:block}
.phones{display:flex;gap:56px;height:100%}
.phone{height:100%;aspect-ratio:390/844;border-radius:44px;overflow:hidden;border:10px solid #1b1f27;box-shadow:0 40px 80px -30px rgba(0,0,0,.9),0 0 70px -20px rgba(244,63,94,.35)}
.phone img{width:100%;height:100%;object-fit:cover;object-position:top;display:block}
.ims{display:flex;gap:44px;height:100%;align-items:stretch}
.im{width:560px;height:100%;border-radius:40px;border:10px solid #1b1f27;background:#000;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 40px 80px -30px rgba(0,0,0,.9),0 0 70px -20px rgba(96,165,250,.35)}
.two .im{width:640px}
.im-top{display:flex;flex-direction:column;align-items:center;padding:16px 0 12px;background:#111317;border-bottom:1px solid #222}
.im-av{width:52px;height:52px;border-radius:50%;background:#1d2230;display:grid;place-items:center}
.im-name{font-weight:600;font-size:16px;margin-top:6px}.im-sub{font-size:11px;color:#8b93a7;margin-top:2px}
.im-body{flex:1;padding:16px 14px;display:flex;flex-direction:column;gap:8px;overflow:hidden;font-size:15.5px;line-height:1.38;font-family:-apple-system,"SF Pro Text",Inter,sans-serif}
.im-cap{text-align:center;font:600 12px/1 "JetBrains Mono";letter-spacing:.08em;text-transform:uppercase;color:#5b6478;padding:12px}
.b{max-width:86%;padding:9px 13px;border-radius:19px;position:relative}
.b.eyes{background:#26282d;align-self:flex-start;border-bottom-left-radius:6px}
.b.me{background:#0a84ff;align-self:flex-end;border-bottom-right-radius:6px;color:#fff}
.b.card{max-width:92%}
.b.thread{margin-left:22px;font-size:13.5px;line-height:1.32;color:#d4d7de;border-left:3px solid #3a3f4a}
.b.loud{box-shadow:0 0 0 2px rgba(244,63,94,.55)}
.b.tap::after{content:"👍";position:absolute;top:-16px;left:-14px;background:#26282d;border-radius:50%;width:30px;height:30px;display:grid;place-items:center;font-size:15px;border:2px solid #000}
.b.heart::after{content:"❤️";position:absolute;top:-16px;left:-14px;background:#26282d;border-radius:50%;width:30px;height:30px;display:grid;place-items:center;font-size:14px;border:2px solid #000}
.edited{align-self:flex-start;font-size:11px;color:#8b93a7;margin:-4px 0 0 8px}
.ph{width:100%;max-height:150px;object-fit:cover;border-radius:12px;display:block}.phc{font-size:13px;margin-top:6px;color:#c7cbd3}.b.photo{padding:6px}
.poll{align-self:flex-start;background:#1c1e22;border:1px solid #2c2f36;border-radius:16px;padding:10px 12px;display:flex;flex-direction:column;gap:6px;min-width:250px;font-size:14px}
.poll span{background:#2a2d33;border-radius:10px;padding:7px 10px}.poll span.on{background:#0a84ff;color:#fff}
.ims-notes{width:560px;display:flex;flex-direction:column;justify-content:center;gap:22px}
.note{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:22px 24px;font-size:18px;line-height:1.5;color:#a3abbd}
.note b{display:block;color:#e7e9ee;font:700 22px/1.2 "Space Grotesk";margin-bottom:8px}
.title .glow{position:absolute;inset:0;background:radial-gradient(35% 45% at 25% 50%,rgba(45,212,191,.22),transparent 70%)}
.t-in{position:absolute;inset:0;display:flex;align-items:center;gap:90px;padding:0 140px}
.t-logo{filter:drop-shadow(0 40px 60px rgba(0,0,0,.7)) drop-shadow(0 0 80px rgba(45,212,191,.35))}
.t-kick{font:600 18px/1 "JetBrains Mono";letter-spacing:.12em;text-transform:uppercase;color:#2dd4bf}
.title h1{font:700 170px/0.95 "Space Grotesk";letter-spacing:-.05em;margin:18px 0 10px}
.t-tag{font:600 40px/1.2 "Space Grotesk";letter-spacing:-.02em;color:#e7e9ee}
.t-tag span{background:linear-gradient(100deg,#5eead4,#60a5fa 50%,#f472b6);-webkit-background-clip:text;background-clip:text;color:transparent}
.title p{font-size:23px;line-height:1.5;color:#a3abbd;max-width:860px;margin:26px 0 30px}
.chips{display:flex;flex-wrap:wrap;gap:12px;max-width:900px}
.chips span{font-size:17px;padding:10px 16px;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1)}
</style></head><body>
${slides
  .map((s, i) =>
    i === 0
      ? titleSlide
      : `<section class="slide"><div class="brand">${logoSvg(34)}Eyes</div><div class="idx">${String(i + 1).padStart(2, "0")} / ${slides.length}</div>
    <div class="head"><div class="kick">${s.kicker}</div><h2>${s.title}</h2><p class="sub">${s.sub}</p></div><div class="body">${s.body}</div></section>`,
  )
  .join("\n")}
</body></html>`;

mkdirSync(OUT, { recursive: true });
const htmlPath = `${ROOT}/devpost/slides.html`;
await Bun.write(htmlPath, html);

const chrome = await launch(process.argv[2]);
try {
  const tab = await Tab.open(chrome.port);
  await tab.size(1800, 1200, 1.5);
  await tab.go(`file://${htmlPath}`, 4000);
  for (let i = 0; i < slides.length; i++) {
    await tab.eval(`window.scrollTo(0, ${i * 1200})`);
    await Bun.sleep(500);
    const path = `${OUT}/${slides[i].file}.jpg`;
    await tab.shot(path, "jpeg", 90);
    console.log(`${slides[i].file}.jpg ${(statSync(path).size / 1e6).toFixed(2)} MB`);
  }
} finally {
  chrome.close();
}
