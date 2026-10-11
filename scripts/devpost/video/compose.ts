// The 60 s product + architecture video. Every scene is HTML driven by one
// render(t) function (no CSS animations), so each frame is exact; headless
// Chrome renders 1800 frames at 1920×1080 and OpenCV encodes them to H.264.
// Run: bun scripts/devpost/video/compose.ts <footage dir> <chrome profile dir> <out frames dir>

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { logoSvg } from "../../../src/logo";
import { launch, Tab } from "../cdp";

const [footage, profile, framesDir] = process.argv.slice(2);
const ROOT = resolve(import.meta.dir, "../../..");
const RAW = `${ROOT}/devpost/raw`;
const FPS = 30;
const DURATION = 60;
const mapStamps: number[] = JSON.parse(await Bun.file(`${footage}/map/stamps.json`).text());

const browser = (id: string, inner: string) => `
  <div class="win" id="${id}">
    <div class="win-bar"><i></i><i></i><i></i><div class="win-url">🔒 eyes-hacklanta.vercel.app</div></div>
    <div class="win-view">${inner}</div>
  </div>`;
const chips = (id: string, items: string[]) => `<div class="chips" id="${id}">${items.map((c, i) => `<div class="chip" data-i="${i}">${c}</div>`).join("")}</div>`;
const head = (id: string, kick: string, title: string) => `<div class="head" id="${id}"><div class="kick">${kick}</div><h2>${title}</h2></div>`;

const im = (id: string, rows: string) => `
  <div class="im" id="${id}">
    <div class="im-top"><div class="im-av">${logoSvg(44)}</div><div class="im-name">Eyes</div><div class="im-sub">iMessage</div></div>
    <div class="im-body">${rows}</div>
  </div>`;

const html = `<!doctype html><html><head><meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box} html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#07090d}
#stage{position:relative;width:1920px;height:1080px;overflow:hidden;color:#e7e9ee;font-family:Inter,system-ui,sans-serif;
 background:linear-gradient(rgba(255,255,255,.022) 1px,transparent 1px) 0 0/48px 48px,linear-gradient(90deg,rgba(255,255,255,.022) 1px,transparent 1px) 0 0/48px 48px,#07090d}
#glowA,#glowB{position:absolute;width:1400px;height:1400px;border-radius:50%;filter:blur(40px);pointer-events:none}
#glowA{background:radial-gradient(circle,rgba(45,212,191,.22),transparent 60%)}
#glowB{background:radial-gradient(circle,rgba(244,63,94,.16),transparent 60%)}
.scene{position:absolute;inset:0;opacity:0}
.head{position:absolute;left:110px;top:84px}
.kick{font:600 20px/1 "JetBrains Mono";letter-spacing:.14em;text-transform:uppercase;color:#2dd4bf}
h2{font:700 60px/1.05 "Space Grotesk";letter-spacing:-.03em;margin:16px 0 0;max-width:1720px;white-space:nowrap}
.win{position:absolute;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.12);background:#0b0e13;
 box-shadow:0 50px 120px -40px rgba(0,0,0,.95),0 0 0 8px rgba(255,255,255,.025),0 0 120px -30px rgba(45,212,191,.4)}
.win-bar{height:44px;background:#12161d;display:flex;align-items:center;gap:9px;padding:0 18px;border-bottom:1px solid rgba(255,255,255,.07)}
.win-bar i{width:13px;height:13px;border-radius:50%;background:#3a3f4a}.win-bar i:nth-child(1){background:#ff5f57}.win-bar i:nth-child(2){background:#febc2e}.win-bar i:nth-child(3){background:#28c840}
.win-url{margin-left:20px;flex:1;max-width:560px;height:26px;border-radius:8px;background:#0b0e13;color:#8b93a7;font:500 13px/26px Inter;padding-left:14px}
.win-view{position:relative;overflow:hidden;width:100%;height:calc(100% - 44px)}
.win-view img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform-origin:0 0}
.chips{position:absolute;display:flex;flex-direction:column;gap:14px}
.chip{opacity:0;padding:16px 22px;border-radius:16px;background:rgba(14,18,25,.86);border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(10px);
 font:600 22px/1.25 "Space Grotesk";box-shadow:0 20px 40px -20px rgba(0,0,0,.9)}
.chip b{display:block;font:700 34px/1.05 "JetBrains Mono";letter-spacing:-.02em;margin-bottom:4px;color:#5eead4}
/* title */
#s1 .t-wrap{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
#logo1{filter:drop-shadow(0 40px 60px rgba(0,0,0,.7)) drop-shadow(0 0 90px rgba(45,212,191,.4))}
#word1{font:700 150px/1 "Space Grotesk";letter-spacing:-.05em;margin-top:10px}
#tag1{display:flex;gap:26px;margin-top:22px;font:700 46px/1 "Space Grotesk";letter-spacing:-.02em}
#tag1 span{opacity:0}
#tag1 span:nth-child(1){color:#5eead4}#tag1 span:nth-child(2){color:#60a5fa}#tag1 span:nth-child(3){color:#f472b6}
/* iMessage */
.im{position:absolute;border-radius:52px;border:12px solid #1b1f27;background:#000;overflow:hidden;box-shadow:0 50px 100px -40px rgba(0,0,0,.95),0 0 100px -30px rgba(96,165,250,.45)}
.im-top{display:flex;flex-direction:column;align-items:center;padding:20px 0 14px;background:#111317;border-bottom:1px solid #222}
.im-av{width:58px;height:58px;border-radius:50%;background:#1d2230;display:grid;place-items:center}
.im-name{font-weight:600;font-size:18px;margin-top:6px}.im-sub{font-size:12px;color:#8b93a7}
.im-body{padding:18px 16px;display:flex;flex-direction:column;gap:10px;font:16.5px/1.4 -apple-system,"SF Pro Text",Inter,sans-serif}
.b{max-width:88%;padding:11px 15px;border-radius:21px;position:relative;opacity:0;transform-origin:left bottom}
.b.e{background:#26282d;align-self:flex-start;border-bottom-left-radius:6px}
.b.m{background:#0a84ff;align-self:flex-end;border-bottom-right-radius:6px;color:#fff;transform-origin:right bottom}
.b.th{margin-left:26px;font-size:15px;color:#d4d7de;border-left:3px solid #3a3f4a}
.b.loud{box-shadow:0 0 0 2px rgba(244,63,94,.7)}
.typing{opacity:0;align-self:flex-start;margin-left:26px;background:#26282d;border-radius:18px;padding:12px 16px;display:flex;gap:6px}
.typing i{width:9px;height:9px;border-radius:50%;background:#8b93a7}
.tap{position:absolute;top:-18px;left:-16px;width:34px;height:34px;border-radius:50%;background:#26282d;border:2px solid #000;display:grid;place-items:center;font-size:17px;opacity:0}
.edited{opacity:0;font-size:12px;color:#8b93a7;margin:-4px 0 0 10px;align-self:flex-start}
.flip{border-radius:6px;padding:0 3px}
.poll{opacity:0;align-self:flex-start;background:#1c1e22;border:1px solid #2c2f36;border-radius:18px;padding:12px 14px;display:flex;flex-direction:column;gap:8px;min-width:280px;font-size:15.5px}
.poll span{background:#2a2d33;border-radius:11px;padding:9px 12px}
.phone{position:absolute;border-radius:52px;overflow:hidden;border:12px solid #1b1f27;box-shadow:0 50px 100px -40px rgba(0,0,0,.95),0 0 100px -30px rgba(244,63,94,.4)}
.phone img{width:100%;height:100%;object-fit:cover;object-position:top;display:block}
/* architecture */
.node{position:absolute;width:250px;padding:20px 20px 18px;border-radius:20px;background:rgba(14,18,25,.92);border:1px solid rgba(255,255,255,.12);opacity:0;box-shadow:0 24px 50px -24px rgba(0,0,0,.9)}
.node .ic{font-size:30px}.node b{display:block;font:700 22px/1.2 "Space Grotesk";margin:10px 0 6px}.node span{font-size:15px;line-height:1.4;color:#a3abbd}
.node.hot{border-color:rgba(45,212,191,.6);box-shadow:0 0 0 1px rgba(45,212,191,.25),0 0 60px -10px rgba(45,212,191,.55)}
#wires{position:absolute;inset:0}
.kpis{position:absolute;left:110px;right:110px;top:330px;display:grid;grid-template-columns:repeat(4,1fr);gap:28px}
.kpi{opacity:0;padding:34px 30px;border-radius:24px;background:rgba(14,18,25,.9);border:1px solid rgba(255,255,255,.1)}
.kpi b{display:block;font:700 76px/1 "JetBrains Mono";letter-spacing:-.04em;background:linear-gradient(90deg,#5eead4,#60a5fa);-webkit-background-clip:text;background-clip:text;color:transparent}
.kpi span{display:block;margin-top:14px;font:600 22px/1.3 "Space Grotesk";color:#cfd4dc}.kpi i{display:block;margin-top:6px;font-style:normal;font-size:16px;color:#8b93a7}
/* end */
#s7 .t-wrap{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
#word7{font:700 120px/1 "Space Grotesk";letter-spacing:-.05em;margin-top:6px}
#tag7{font:600 44px/1.2 "Space Grotesk";letter-spacing:-.02em;margin-top:14px}
#tag7 span{background:linear-gradient(100deg,#5eead4,#60a5fa 50%,#f472b6);-webkit-background-clip:text;background-clip:text;color:transparent}
#url7{margin-top:34px;display:flex;gap:16px}
#url7 div{font:600 22px/1 "JetBrains Mono";padding:14px 22px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14)}
#bar{position:absolute;left:0;bottom:0;height:4px;background:linear-gradient(90deg,#2dd4bf,#60a5fa,#f472b6)}
</style></head><body><div id="stage">
<div id="glowA"></div><div id="glowB"></div>

<section class="scene" id="s1"><div class="t-wrap"><div id="logo1">${logoSvg(260)}</div><div id="word1">Eyes</div>
  <div id="tag1"><span>Detect.</span><span>Dispatch.</span><span>Act.</span></div></div></section>

<section class="scene" id="s2">${head("h2", "01 · Detect", "AttackNet spots an attack as it happens")}
  ${browser("w2", `<img id="ds"/>`)}
  ${chips("c2", ["<b>1.9M</b>weights, our own model", "<b>84</b>body-language signals per person", "<b>92%</b>right on footage it never saw", "<b>Red</b>only above 30% confidence"])}</section>

<section class="scene" id="s3">${head("h3", "02 · Spread", "The alert crosses Atlanta in seconds")}
  ${browser("w3", `<img id="mapA"/><img id="mapB"/>`)}
  ${chips("c3", ["<b>130</b>city mesh nodes", "<b>Police · EMS · MARTA</b>everyone nearby, at once", "<b>Wi-Fi · radio · internet</b>the best link, every message", "<b>Offline</b>still works when the network is down"])}</section>

<section class="scene" id="s4">${head("h4", "03 · Coordinate", "AI agents run the response in iMessage")}
  ${im("im4", `
    <div class="b e" id="card">🚨 <b>Assault reported near GSU Student Center East</b><br/>12:49 AM · spotted by an Eyes camera<br/><br/>⚠️ Be careful around GSU Student Center East. Avoid the area; stay in busy, well-lit places.<br/><br/>Help is on the way:<br/>🚓 GSU Police · about 2 min · <span class="flip" id="pol">paged</span><br/>🚑 Grady Health · about 2 min · paged<br/>👥 5 people within 800 m warned</div>
    <div class="edited" id="edited">Edited</div>
    <div class="typing" id="typing"><i></i><i></i><i></i></div>
    <div class="b e th" id="brief">👁 Sentinel: Assault detected at 12:49 AM.<br/>🛰 Network: delivered over internet + Wi-Fi + mesh radio.<br/>🧭 Dispatch: paging GSU Police (2 min) and Grady Health (2 min).<br/>🧍 Safety: people nearby, move away. Help is coming.</div>
    <div class="b m" id="omw">on my way<div class="tap" id="tap">👍</div></div>`)}
  ${chips("c4", ["<b>Live card</b>edited in place, no extra buzz", "<b>A team of agents</b>Sentinel · Dispatch · Police · EMS · Safety · Care", "<b>Humans in the loop</b>“on my way” turns it green 👍"])}</section>

<section class="scene" id="s5">${head("h5", "04 · Protect", "Everyone nearby is warned. Anyone hurt gets help.")}
  ${im("im5", `
    <div class="b e loud" id="alert">⚠️ <b>EYES SAFETY ALERT</b><br/>Assault reported at Woodruff Park, 420 m south-east of you.<br/>Avoid the area; stay in busy, well-lit places.<br/>Nearest safe spot: Atlanta Central Library, 240 m.</div>
    <div class="poll" id="poll"><b>Are you safe?</b><span>I'm safe</span><span id="help">I need help</span></div>
    <div class="b e" id="care">🆘 Help is coming. An ambulance from Grady has been requested (about 3 min).<br/><br/>If anyone isn't breathing, call 911 now.</div>
    <div class="b m" id="bleed">my friend is bleeding</div>
    <div class="b e" id="aid">Press firmly on it with a clean cloth. I've told the EMS crew. Is he conscious?</div>`)}
  <div class="phone" id="ph5"><img src="${RAW}/phone-officer-alert.png"/></div>
  ${chips("c5", ["<b>800 m</b>opt-in alerts with a safe spot", "<b>1 tap</b>“Are you safe?” poll", "<b>Care</b>EMS paged, first aid until they arrive"])}</section>

<section class="scene" id="s6">${head("h6", "05 · Under the hood", "One system, from camera to iMessage")}
  <svg id="wires" viewBox="0 0 1920 1080"><defs><linearGradient id="wg" x1="0" x2="1"><stop offset="0" stop-color="#2dd4bf"/><stop offset="1" stop-color="#60a5fa"/></linearGradient></defs>
    <path id="wire" d="M 360 520 L 1560 520" stroke="url(#wg)" stroke-width="3" fill="none" stroke-dasharray="1200" stroke-dashoffset="1200"/>
    <path id="wire2" d="M 1000 600 L 1000 760" stroke="#3a4252" stroke-width="2" fill="none" stroke-dasharray="6 8" opacity="0"/>
    <circle id="pkt" r="10" fill="#fff" opacity="0" style="filter:drop-shadow(0 0 12px #5eead4)"/></svg>
  <div class="node" id="n0" style="left:110px;top:420px"><div class="ic">📷</div><b>Camera</b><span>MediaPipe pose in the browser</span></div>
  <div class="node" id="n1" style="left:410px;top:420px"><div class="ic">🧠</div><b>AttackNet</b><span>1.9M-weight transformer · NVIDIA T4</span></div>
  <div class="node" id="n2" style="left:710px;top:420px"><div class="ic">🕸</div><b>Eyes mesh</b><span>Wi-Fi · mesh radio · internet</span></div>
  <div class="node" id="n3" style="left:1010px;top:420px"><div class="ic">🤖</div><b>Dispatch AI</b><span>Agents on Gemini 2.5 Flash</span></div>
  <div class="node" id="n4" style="left:1310px;top:420px"><div class="ic">💬</div><b>Photon</b><span>Spectrum → iMessage</span></div>
  <div class="node" id="n5" style="left:1610px;top:420px;width:200px"><div class="ic">🚓</div><b>Responders</b><span>and people nearby</span></div>
  <div class="node" id="n6" style="left:760px;top:760px;width:480px"><div class="ic">☁️</div><b>Google Cloud · Vercel</b><span>Vertex AI · Cloud SQL · Cloud Storage · Compute Engine</span></div>
  <div class="kpis" id="kp">
    <div class="kpi"><b>92%</b><span>right on unseen clips</span><i>400 held-out real-world videos</i></div>
    <div class="kpi"><b>0.97</b><span>AUC</span><i>0.5 is a coin flip</i></div>
    <div class="kpi"><b>130</b><span>city mesh nodes</span><i>police · hospitals · MARTA</i></div>
    <div class="kpi"><b>Live</b><span>today</span><i>eyes-hacklanta.vercel.app</i></div>
  </div></section>

<section class="scene" id="s7"><div class="t-wrap"><div id="logo7">${logoSvg(200)}</div><div id="word7">Eyes</div>
  <div id="tag7">Cameras that watch. <span>Eyes that act.</span></div>
  <div id="url7"><div>eyes-hacklanta.vercel.app</div><div>github.com/kbhatnagar1506/eyes</div></div></div></section>
<div id="bar"></div>
</div>
<script>
const FOOT = ${JSON.stringify(footage)};
const STAMPS = ${JSON.stringify(mapStamps)};
const $ = (s) => document.querySelector(s);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = (x) => { x = clamp(x); return x < .5 ? 4*x*x*x : 1 - Math.pow(-2*x + 2, 3) / 2; };
const out = (x) => 1 - Math.pow(1 - clamp(x), 3);
const lerp = (a, b, k) => a + (b - a) * k;
// 0 → 1 → 0 envelope for a scene shown between a and b.
const env = (t, a, b, fi = .45, fo = .45) => Math.min(out((t - a) / fi), out((b - t) / fo));
const pop = (el, k, dy = 14, s0 = .94) => { el.style.opacity = k; el.style.transform = "translateY(" + (1 - k) * dy + "px) scale(" + lerp(s0, 1, k) + ")"; };
const show = (el, t, at, dur = .45, dy = 14) => pop(el, out((t - at) / dur), dy);
const pad = (n) => String(n).padStart(4, "0");
function loadImg(img, src) { if (img.dataset.src === src) return Promise.resolve(); img.dataset.src = src; img.src = src; return img.decode().catch(() => {}); }
function scene(id, t, a, b) { const k = env(t, a, b); const el = $(id); el.style.opacity = k; el.style.transform = "scale(" + lerp(1.025, 1, k) + ")"; return t >= a - .1 && t <= b + .1; }
function chipsIn(id, t, at, gap = .55) { document.querySelectorAll(id + " .chip").forEach((c, i) => { const k = out((t - at - i * gap) / .5); c.style.opacity = k; c.style.transform = "translateX(" + (1 - k) * 40 + "px)"; }); }

async function render(t) {
  const work = [];
  // Background glow drifts.
  $("#glowA").style.transform = "translate(" + (-400 + 260 * Math.sin(t / 7)) + "px," + (-500 + 120 * Math.cos(t / 9)) + "px)";
  $("#glowB").style.transform = "translate(" + (1100 + 200 * Math.cos(t / 8)) + "px," + (-300 + 160 * Math.sin(t / 6)) + "px)";
  $("#bar").style.width = (t / ${DURATION} * 100) + "%";

  // 1 · title (0–5)
  if (scene("#s1", t, 0, 5)) {
    const k = out(t / 1.1);
    $("#logo1").style.transform = "scale(" + lerp(.6, 1, k) + ") rotate(" + lerp(-14, 0, k) + "deg)";
    $("#logo1").style.opacity = k;
    const eye = $("#logo1 .logo-eye"); if (eye) eye.style.transform = "translate(" + (t > 2.2 && t < 3.4 ? -3.5 : t > 3.4 && t < 4.2 ? 3 : 0) + "px,0)";
    show($("#word1"), t, .5, .6, 24);
    document.querySelectorAll("#tag1 span").forEach((s, i) => show(s, t, 1.4 + i * .5, .45, 18));
  }
  // 2 · detect (5–15)
  if (scene("#s2", t, 5, 15)) {
    const u = t - 5;
    show($("#h2"), u, .1, .5, 20);
    const w = $("#w2"); w.style.left = "110px"; w.style.top = "250px"; w.style.width = "1260px"; w.style.height = "753px";
    show(w, u, .25, .6, 30);
    // Half-speed through the attack (AttackNet red at 98-99%), then hold on it.
    const fi = Math.min(93, Math.max(0, Math.floor((u - .25) * 15)));
    work.push(loadImg($("#ds"), FOOT + "/ds/" + pad(fi) + ".jpg"));
    // Push in on the red box (page coords of its centre), then ease back out.
    const z = ease((u - 2.0) / 1.6) - ease((u - 8.4) / 1.0);
    const s = lerp(1, 1.9, z), W = 1260, H = 709, px = .425, py = .49;
    $("#ds").style.transform = "translate(" + lerp(0, W / 2 - s * px * W, z) + "px," + lerp(0, H / 2 - s * py * H, z) + "px) scale(" + s + ")";
    const c = $("#c2"); c.style.left = "1420px"; c.style.top = "300px"; c.style.width = "400px";
    chipsIn("#c2", u, 1.6, .9);
  }
  // 3 · spread (15–25)
  if (scene("#s3", t, 15, 25)) {
    const u = t - 15;
    show($("#h3"), u, .1, .5, 20);
    const w = $("#w3"); w.style.left = "110px"; w.style.top = "250px"; w.style.width = "1260px"; w.style.height = "753px";
    show(w, u, .25, .6, 30);
    const mt = clamp(u * 1.12, 0, STAMPS[STAMPS.length - 1]);
    let j = 0; while (j < STAMPS.length - 2 && STAMPS[j + 1] <= mt) j++;
    const f = clamp((mt - STAMPS[j]) / Math.max(.001, STAMPS[j + 1] - STAMPS[j]));
    work.push(loadImg($("#mapA"), FOOT + "/map/" + pad(j) + ".jpg"), loadImg($("#mapB"), FOOT + "/map/" + pad(j + 1) + ".jpg"));
    $("#mapB").style.opacity = f;
    const kb = lerp(1, 1.07, ease(u / 10));
    $("#mapA").style.transform = $("#mapB").style.transform = "translate(" + lerp(0, -1260 * .03, ease(u / 10)) + "px,0) scale(" + kb + ")";
    const c = $("#c3"); c.style.left = "1420px"; c.style.top = "300px"; c.style.width = "400px";
    chipsIn("#c3", u, 1.4, 1.2);
  }
  // 4 · coordinate (25–37)
  if (scene("#s4", t, 25, 37)) {
    const u = t - 25;
    show($("#h4"), u, .1, .5, 20);
    const p = $("#im4"); p.style.left = "420px"; p.style.top = "232px"; p.style.width = "560px"; p.style.height = "820px";
    show(p, u, .3, .6, 40);
    pop($("#card"), out((u - .9) / .45));
    const typingK = clamp((u - 2.8) / .2) * (1 - clamp((u - 4.3) / .15));
    $("#typing").style.opacity = typingK;
    $("#typing").style.display = u > 4.45 ? "none" : "flex";
    document.querySelectorAll("#typing i").forEach((d, i) => { d.style.transform = "translateY(" + (-4 * Math.max(0, Math.sin((u * 7) - i * .9))) + "px)"; });
    pop($("#brief"), out((u - 4.4) / .45));
    pop($("#omw"), out((u - 6.9) / .4));
    const tk = out((u - 7.7) / .3); $("#tap").style.opacity = tk; $("#tap").style.transform = "scale(" + lerp(.2, 1, tk) + ")";
    const on = u > 8.1;
    const pol = $("#pol"); pol.textContent = on ? "🟢 on the way" : "paged";
    pol.style.background = on ? "rgba(45,212,191," + (.45 * (1 - clamp((u - 8.1) / 1.4))) + ")" : "transparent";
    $("#edited").style.opacity = out((u - 8.3) / .4);
    const c = $("#c4"); c.style.left = "1080px"; c.style.top = "360px"; c.style.width = "620px";
    document.querySelectorAll("#c4 .chip").forEach((el, i) => { const at = [1.2, 4.6, 8.2][i]; const k = out((u - at) / .5); el.style.opacity = k; el.style.transform = "translateX(" + (1 - k) * 40 + "px)"; });
  }
  // 5 · protect (37–46)
  if (scene("#s5", t, 37, 46)) {
    const u = t - 37;
    show($("#h5"), u, .1, .5, 20);
    const p = $("#im5"); p.style.left = "250px"; p.style.top = "232px"; p.style.width = "560px"; p.style.height = "820px";
    show(p, u, .3, .6, 40);
    pop($("#alert"), out((u - .8) / .45));
    $("#alert").style.boxShadow = "0 0 0 2px rgba(244,63,94,.7), 0 0 " + (14 + 14 * Math.sin(u * 6)) + "px rgba(244,63,94," + (u < 3 ? .55 : .2) + ")";
    pop($("#poll"), out((u - 1.8) / .45));
    const h = $("#help"); const sel = u > 3.4; h.style.background = sel ? "#0a84ff" : "#2a2d33"; h.style.color = sel ? "#fff" : ""; h.textContent = sel ? "I need help ✓" : "I need help";
    pop($("#care"), out((u - 4.1) / .45));
    pop($("#bleed"), out((u - 5.6) / .4));
    pop($("#aid"), out((u - 6.6) / .45));
    const ph = $("#ph5"); ph.style.left = "900px"; ph.style.top = "232px"; ph.style.width = "380px"; ph.style.height = "822px";
    show(ph, u, .7, .6, 40);
    const c = $("#c5"); c.style.left = "1360px"; c.style.top = "380px"; c.style.width = "440px";
    chipsIn("#c5", u, 1.2, 1.6);
  }
  // 6 · under the hood (46–56)
  if (scene("#s6", t, 46, 56)) {
    const u = t - 46;
    show($("#h6"), u, .1, .5, 20);
    const diagram = 1 - out((u - 5.6) / .5);
    for (let i = 0; i < 7; i++) {
      const n = $("#n" + i);
      const k = out((u - .5 - i * .35) / .45) * diagram;
      n.style.opacity = k; n.style.transform = "translateY(" + (1 - k) * 22 + "px)";
    }
    $("#wire").style.strokeDashoffset = 1200 * (1 - ease((u - .8) / 2.4));
    $("#wires").style.opacity = diagram;
    $("#wire2").style.opacity = out((u - 2.8) / .4);
    const pk = $("#pkt"); const pu = ((u - 3.2) % 1.6) / 1.6;
    pk.setAttribute("cx", 360 + 1200 * ease(pu)); pk.setAttribute("cy", 520); pk.setAttribute("opacity", u > 3.2 ? 1 : 0);
    const lit = u > 3.2 ? Math.min(5, Math.floor(ease(pu) * 6)) : -1;
    for (let i = 0; i < 6; i++) $("#n" + i).classList.toggle("hot", i === lit);
    document.querySelectorAll("#kp .kpi").forEach((el, i) => { const k = out((u - 6 - i * .3) / .5); el.style.opacity = k; el.style.transform = "translateY(" + (1 - k) * 30 + "px)"; });
  }
  // 7 · end (56–60)
  if (scene("#s7", t, 56, 60.6, .5, .01)) {
    const u = t - 56;
    const k = out(u / .8); $("#logo7").style.opacity = k; $("#logo7").style.transform = "scale(" + lerp(.75, 1, k) + ")";
    show($("#word7"), u, .3, .5, 20); show($("#tag7"), u, .8, .5, 16); show($("#url7"), u, 1.4, .5, 12);
  }
  await Promise.all(work);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return true;
}
window.render = render;
</script></body></html>`;

const htmlPath = `${footage}/compose.html`;
await Bun.write(htmlPath, html);
mkdirSync(framesDir, { recursive: true });
const only = process.env.FRAMES?.split(",").map(Number); // FRAMES=150,450 renders just those (previews)

const chrome = await launch(profile);
try {
  const tab = await Tab.open(chrome.port);
  await tab.size(1920, 1080, 1);
  await tab.go(`file://${htmlPath}`, 4000);
  const total = FPS * DURATION;
  const list = only ?? Array.from({ length: total }, (_, i) => i);
  const t0 = performance.now();
  for (const i of list) {
    await tab.eval(`render(${(i / FPS).toFixed(4)})`);
    await tab.shot(`${framesDir}/${String(i).padStart(5, "0")}.jpg`, "jpeg", 92);
    if (i % 150 === 0) console.log(`frame ${i}/${total} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
  }
} finally {
  chrome.close();
}
