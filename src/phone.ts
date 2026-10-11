// Phone app for a person on the mesh: police officer (the hero), bus driver,
// or rider. Open https://<laptop>:5174/?role=police (or driver / rider).
//
// States, driven by the incident on the mesh and what this person has done:
//   all clear  → radar, mesh status, recent incidents
//   alert      → a live incident this person hasn't acted on: the evidence
//                frame from the camera, where, how far, one big action
//   responding → after acting: navigation, the response timeline, and the
//                agents' thread (the same one that goes out over iMessage)

import { logoSvg } from "./logo";
import "./style.css";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { TRANSPORT_LABEL, type TransportKind } from "./mesh/node";
import { metersBetween } from "./mesh/radio";
import { problemLabel, PROBLEMS, ROLE_LABEL, type ProblemId, type Role } from "./mesh/scenario";
import { joinMesh } from "./mesh/session";
import { framesFor, safeJpeg } from "./mesh/frames";

type PhoneRole = Extract<Role, "driver" | "rider" | "police">;
type Doc = Record<string, unknown> & { id: string };

interface Action {
  label: string;
  value: string;
  primary?: boolean;
}

const PROFILE: Record<PhoneRole, { icon: string; title: string; actions: Action[]; doneLabel: string }> = {
  police: {
    icon: "🚓",
    title: "Officer on duty",
    actions: [
      { label: "🚓 Respond", value: "responding", primary: true },
      { label: "Request backup", value: "backup requested" },
    ],
    doneLabel: "Responding",
  },
  driver: {
    icon: "🧑‍✈️",
    title: "Bus operator",
    actions: [
      { label: "✓ Acknowledge", value: "acknowledged", primary: true },
      { label: "Hold the bus", value: "holding the bus" },
    ],
    doneLabel: "Acknowledged",
  },
  rider: {
    icon: "🧍",
    title: "Rider",
    actions: [
      { label: "🆘 I need help", value: "needs help", primary: true },
      { label: "I'm safe", value: "safe" },
    ],
    doneLabel: "Status sent",
  },
};

// Live = worth an alarm. Older ones still show, as history.
const LIVE_MS = 10 * 60_000;
// Raised more than this before it reached the phone: it was caught up on
// joining the mesh, so no siren for it.
const LOUD_IF_NEWER_MS = 60_000;
const URBAN_KMH = 25;
const DEVICE_STALE_MS = 12_000;

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const clock = (t: number) => new Date(t).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
const fmtDist = (m: number) => (m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`);

function ago(t: number) {
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${s % 60}s ago` : `${Math.floor(m / 60)}h ${m % 60}m ago`;
}

export async function startPhone(role: PhoneRole) {
  const profile = PROFILE[role];
  const app = document.querySelector<HTMLDivElement>("#app")!;
  app.innerHTML = `
    <div class="pa" id="pa" data-state="clear">
      <header class="pa-top">
        <div class="pa-who">
          <div class="pa-avatar">${profile.icon}</div>
          <div>
            <div class="pa-name">${profile.title}</div>
            <div class="pa-sub" id="paWhere">joining the mesh…</div>
          </div>
        </div>
        <button class="pa-mesh" id="paMeshBtn" type="button"><span class="pa-dot"></span><span id="paMeshText">MESH</span></button>
      </header>
      <div class="pa-sheet hidden" id="paSheet"></div>
      <main class="pa-body" id="paBody"></main>
      <div class="pa-mapwrap hidden" id="paMapWrap"><div class="pa-map" id="paMap"></div></div>
      <section class="pa-extra" id="paExtra"></section>
      <footer class="pa-actions" id="paActions"></footer>
    </div>
  `;
  const $ = <T extends HTMLElement>(sel: string) => app.querySelector<T>(sel)!;
  // Agent messages and frames arrive while someone is about to tap: only
  // touch a section when its content actually changed, so buttons aren't
  // swapped out from under a finger. Time-ago labels tick separately.
  const setHtml = (sel: string, html: string) => {
    const el = $(sel);
    if (el.dataset.html === html) return;
    el.dataset.html = html;
    el.innerHTML = html;
    tickAgo();
  };
  const tickAgo = () => app.querySelectorAll<HTMLElement>("[data-ago]").forEach((el) => (el.textContent = ago(Number(el.dataset.ago))));

  const s = await joinMesh(role);
  const node = s.node;
  $("#paWhere").textContent = s.position.place ?? "GSU Student Center East";

  // ---- mini map (kept alive across re-renders) ----------------------------
  let map: L.Map | null = null;
  let incMarker: L.Marker | null = null;
  let meMarker: L.Marker | null = null;
  let mapFor: string | null = null;
  function showMap(inc: Doc) {
    $("#paMapWrap").classList.remove("hidden");
    if (mapFor === inc.id) return; // already framed; don't jump on every update
    mapFor = inc.id;
    const at: L.LatLngExpression = [Number(inc.lat), Number(inc.lng)];
    const me: L.LatLngExpression = [node.info.lat, node.info.lng];
    $("#paMapWrap").classList.remove("hidden");
    if (!map) {
      map = L.map($("#paMap"), { zoomControl: false, attributionControl: false, dragging: false, scrollWheelZoom: false, doubleClickZoom: false, touchZoom: false, keyboard: false });
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map);
    }
    const icon = (cls: string, html = "") => L.divIcon({ className: "", html: `<div class="${cls}">${html}</div>`, iconSize: [22, 22], iconAnchor: [11, 11] });
    incMarker?.remove();
    meMarker?.remove();
    incMarker = L.marker(at, { icon: icon("pa-pin-inc") }).addTo(map);
    meMarker = L.marker(me, { icon: icon("pa-pin-me") }).addTo(map);
    const bounds = L.latLngBounds([at, me]);
    if (metersBetween(node.info, { lat: Number(inc.lat), lng: Number(inc.lng) }) < 150) map.setView(at, 17);
    else map.fitBounds(bounds.pad(0.35));
    setTimeout(() => map?.invalidateSize(), 50);
  }
  const hideMap = () => {
    $("#paMapWrap").classList.add("hidden");
    mapFor = null;
  };

  // ---- what reached this phone, and how ----------------------------------
  const arrivals = new Map<string, { at: number; transport: TransportKind; hops: number }>();
  const frameArrived = new Map<string, number>();
  let shown: string | null = null;
  node.onDeliver((e) => {
    const at = Date.now();
    // Runs after the render the same delivery triggered: record, then redraw.
    queueMicrotask(() => {
      if (!e.colls.includes("incidents") || !shown || arrivals.has(shown)) return;
      arrivals.set(shown, { at, transport: e.transport, hops: e.hops });
      render();
    });
  });

  // ---- state ----------------------------------------------------------------
  const incidents = () => node.store.list("incidents").filter((d) => d.at !== undefined) as Doc[];
  const isLive = (inc: Doc) => Date.now() - Number(inc.at) < LIVE_MS && inc.status !== "resolved";
  const mine = (inc: Doc) => inc[`ack:${role}`];

  function pickShown(): Doc | null {
    const live = incidents()
      .filter(isLive)
      .sort((a, b) => Number(b.at) - Number(a.at));
    return live[0] ?? null;
  }

  function stateOf(inc: Doc | null): "clear" | "alert" | "responding" {
    if (!inc) return "clear";
    return mine(inc) ? "responding" : "alert";
  }

  // ---- render ---------------------------------------------------------------
  function render() {
    const inc = pickShown();
    const state = stateOf(inc);
    $("#pa").dataset.state = state;
    if (!inc) {
      shown = null;
      renderClear();
      return;
    }
    if (inc.id !== shown) {
      const fresh = Date.now() - Number(inc.at) < LOUD_IF_NEWER_MS;
      shown = inc.id;
      if (state === "alert" && fresh) siren();
    }
    if (state === "alert") renderAlert(inc);
    else renderResponding(inc);
  }

  function renderClear() {
    hideMap();
    const recent = incidents()
      .sort((a, b) => Number(b.at) - Number(a.at))
      .slice(0, 4);
    setHtml("#paBody", `
      <div class="pa-clear">
        <div class="pa-radar"><div class="pa-sweep"></div><div class="pa-ring r1"></div><div class="pa-ring r2"></div><div class="pa-ring r3"></div><div class="pa-mascot">${logoSvg(56)}</div></div>
        <div class="pa-clear-title">All clear</div>
        <div class="pa-clear-sub">Watching the area around you through the Eyes mesh. You'll be alerted the moment a camera sees trouble nearby.</div>
      </div>`);
    setHtml("#paExtra", recent.length
      ? `<div class="pa-card"><div class="pa-label">Recent</div>${recent
          .map(
            (r) =>
              `<div class="pa-hist"><span>${PROBLEMS[r.kind as ProblemId]?.icon ?? "⚠"} ${esc(problemLabel(r.kind))}</span><span class="pa-dim">${esc(String(r.label ?? ""))}</span><span class="pa-dim"><span data-ago="${Number(r.at)}"></span> · ${esc(String(r.status ?? "detected"))}</span></div>`,
          )
          .join("")}</div>`
      : "");
    setHtml("#paActions", "");
  }

  // Newest punch big, earlier punches as a strip: every frame the camera
  // caught is here, in order.
  function frameHtml(inc: Doc, compact = false) {
    const frames = framesFor(node.store, inc.id)
      .map((f) => ({ src: safeJpeg(f.data), seq: Number(f.seq ?? 1), at: Number(f.at ?? inc.at) }))
      .filter((f): f is { src: string; seq: number; at: number } => !!f.src);
    if (frames.length) {
      const latest = frames[frames.length - 1];
      const earlier = frames.slice(0, -1).reverse();
      return `<figure class="pa-frame ${compact ? "pa-frame-sm" : ""}" data-seq="${latest.seq}">
          <img src="${latest.src}" alt="Camera frame, punch ${latest.seq}"/>
          <figcaption><span class="pa-live">● EVIDENCE #${latest.seq}</span> · ${clock(latest.at)} · ${frames.length} frame${frames.length === 1 ? "" : "s"} received</figcaption>
        </figure>
        ${earlier.length ? `<div class="pa-thumbs">${earlier.map((f) => `<figure><img src="${f.src}" alt="Punch ${f.seq}"/><span>#${f.seq}</span></figure>`).join("")}</div>` : ""}`;
    }
    const simulated = /simulated/i.test(String(inc.detail ?? ""));
    return simulated
      ? `<div class="pa-frame pa-frame-empty">Simulated incident: no camera frame</div>`
      : `<div class="pa-frame pa-frame-empty pa-shimmer">Receiving camera frame over the mesh…</div>`;
  }

  function factsChips(inc: Doc) {
    const meters = metersBetween(node.info, { lat: Number(inc.lat), lng: Number(inc.lng) });
    const eta = Math.max(1, Math.round(1 + meters / 1000 / (URBAN_KMH / 60)));
    const arrival = arrivals.get(inc.id);
    const chips = [
      `<span class="pa-chip"><b>${fmtDist(meters)}</b> away</span>`,
      role !== "rider" ? `<span class="pa-chip">ETA <b>${eta} min</b></span>` : "",
      // No latency figure: it would compare the camera's clock with this
      // phone's, and device clocks disagree by more than the mesh takes.
      arrival ? `<span class="pa-chip">via ${esc(TRANSPORT_LABEL[arrival.transport])} · ${arrival.hops} hop${arrival.hops === 1 ? "" : "s"}</span>` : "",
    ];
    return `<div class="pa-chips">${chips.join("")}</div>`;
  }

  function renderAlert(inc: Doc) {
    const p = PROBLEMS[inc.kind as ProblemId];
    setHtml("#paBody", `
      <div class="pa-alert-top"><span class="pa-siren"></span>PRIORITY ALERT · <span data-ago="${Number(inc.at)}"></span></div>
      ${frameHtml(inc)}
      <div class="pa-title">${p?.icon ?? "⚠"} ${esc(problemLabel(inc.kind))}</div>
      <div class="pa-place">${esc(String(inc.label ?? ""))}</div>
      <div class="pa-detail">${esc(String(inc.detail ?? ""))}</div>
      ${factsChips(inc)}`);
    showMap(inc);
    setHtml("#paExtra", "");
    setHtml("#paActions", profile.actions
      .map((a) => `<button type="button" class="${a.primary ? "pa-btn-primary" : "pa-btn"}" data-v="${esc(a.value)}">${a.label}</button>`)
      .join(""));
  }

  function renderResponding(inc: Doc) {
    const p = PROBLEMS[inc.kind as ProblemId];
    const myStatus = String(inc[`status:${role}`] ?? profile.doneLabel);
    setHtml("#paBody", `
      <div class="pa-resp-top">${profile.icon} ${esc(myStatus.charAt(0).toUpperCase() + myStatus.slice(1))} · <span data-ago="${Number(mine(inc))}"></span></div>
      <div class="pa-title pa-title-sm">${p?.icon ?? "⚠"} ${esc(problemLabel(inc.kind))} · ${esc(String(inc.label ?? ""))}</div>
      ${factsChips(inc)}
      ${frameHtml(inc, true)}`);
    showMap(inc);
    setHtml("#paExtra", `${timelineHtml(inc)}${threadHtml(inc)}`);
    const nav = `https://maps.apple.com/?daddr=${Number(inc.lat)},${Number(inc.lng)}&dirflg=d`;
    setHtml("#paActions", `
      ${role !== "rider" ? `<a class="pa-btn-primary" href="${nav}" target="_blank" rel="noopener">🧭 Navigate</a>` : ""}
      ${role === "police" ? (inc.injuredReportedAt || inc.injured ? `<button type="button" class="pa-btn" disabled>🚑 EMS paged</button>` : `<button type="button" class="pa-btn pa-btn-warn" data-act="injured">🚑 Person injured</button>`) : ""}
      <button type="button" class="pa-btn" data-act="resolve">${role === "police" ? "Mark resolved" : "Done"}</button>`);
  }

  function timelineHtml(inc: Doc) {
    const camera = node.store.get("presence", String(inc.camera ?? ""));
    const arrival = arrivals.get(inc.id);
    const frameAt = frameArrived.get(inc.id);
    const steps: { t: number; text: string }[] = [{ t: Number(inc.at), text: `Detected by ${esc(String(camera?.name ?? "CCTV"))}` }];
    if (arrival) steps.push({ t: arrival.at, text: `Reached you via ${esc(TRANSPORT_LABEL[arrival.transport])}, ${arrival.hops} hop${arrival.hops === 1 ? "" : "s"}` });
    for (const f of framesFor(node.store, inc.id)) {
      const t = frameArrived.get(f.id) ?? frameAt;
      if (t) steps.push({ t, text: `📸 Evidence frame #${Number(f.seq ?? 1)} received` });
    }
    for (const [k, v] of Object.entries(inc)) {
      if (!k.startsWith("ack:")) continue;
      const who = k.slice(4);
      const label = who === "dispatch-ai" ? "Dispatch AI" : who === "ems" ? "EMS" : who === role ? "You" : ROLE_LABEL[who as Role] ?? who;
      steps.push({ t: Number(v), text: `${esc(label)}: ${esc(String(inc[`status:${who}`] ?? "acknowledged"))}` });
    }
    if (inc.injuredReportedAt) steps.push({ t: Number(inc.injuredReportedAt), text: "🚑 Injured person reported from the scene" });
    if (inc.careAt) steps.push({ t: Number(inc.careAt), text: `🩹 Care: ${esc(String(inc.care ?? "EMS paged"))}` });
    if (inc.areaNotified) steps.push({ t: Number(inc.at) + 1, text: `⚠️ ${Number(inc.areaNotified)} people nearby alerted by iMessage · ${Number(inc.areaSafe ?? 0)} safe${Number(inc.areaHelp ?? 0) ? ` · ${Number(inc.areaHelp)} need help` : ""}` });
    if (inc.resolvedAt) steps.push({ t: Number(inc.resolvedAt), text: "Resolved" });
    steps.sort((a, b) => a.t - b.t);
    return `<div class="pa-card"><div class="pa-label">Response</div><ol class="pa-timeline">${steps
      .map((s) => `<li><span class="pa-t">${clock(s.t)}</span>${s.text}</li>`)
      .join("")}</ol></div>`;
  }

  function threadHtml(inc: Doc) {
    const lines = node.store
      .list("agentchat")
      .filter((d) => d.incident === inc.id)
      .sort((a, b) => Number(a.seq) - Number(b.seq));
    if (!lines.length) return "";
    return `<div class="pa-card"><div class="pa-label">Agents · iMessage thread</div><div class="pa-thread">${lines
      .map((l) => `<div class="pa-bubble"><div class="pa-bub-who">${esc(String(l.emoji ?? ""))} ${esc(String(l.agent ?? ""))}</div>${esc(String(l.text ?? ""))}</div>`)
      .join("")}</div></div>`;
  }

  // Another punch in the same fight: short buzz and a flash, not a full siren.
  function newPunch() {
    try {
      navigator.vibrate?.([250, 80, 250]);
    } catch {
      /* not supported */
    }
    const fig = app.querySelector<HTMLElement>(".pa-frame");
    fig?.classList.remove("pa-flash");
    void fig?.offsetWidth;
    fig?.classList.add("pa-flash");
  }

  function siren() {
    try {
      navigator.vibrate?.([500, 150, 500, 150, 800]);
    } catch {
      /* not supported */
    }
    try {
      const ctx = new AudioContext();
      const gain = ctx.createGain();
      gain.gain.value = 0.12;
      gain.connect(ctx.destination);
      [0, 0.45].forEach((offset) => {
        const osc = ctx.createOscillator();
        osc.type = "square";
        osc.frequency.setValueAtTime(880, ctx.currentTime + offset);
        osc.frequency.linearRampToValueAtTime(660, ctx.currentTime + offset + 0.35);
        osc.connect(gain);
        osc.start(ctx.currentTime + offset);
        osc.stop(ctx.currentTime + offset + 0.38);
      });
    } catch {
      /* audio needs a prior tap on some phones */
    }
  }

  // ---- mesh status pill + device sheet --------------------------------------
  function devices() {
    const now = Date.now();
    const byName = new Map<string, { name: string; role: string; count: number; route: TransportKind | null }>();
    for (const p of node.store.list("presence")) {
      if (p.virtual || p.id === node.id || now - Number(p.seen ?? 0) > DEVICE_STALE_MS) continue;
      const name = String(p.name);
      const entry = byName.get(name) ?? { name, role: String(p.role), count: 0, route: null };
      entry.count++;
      entry.route ??= node.route(p.id);
      byName.set(name, entry);
    }
    return [...byName.values()];
  }

  function renderMesh() {
    const devs = devices();
    const peers = node.neighbors();
    const up = (k: TransportKind) => peers.some((p) => node.linkStats(p).some((l) => l.kind === k && l.alive));
    const total = devs.reduce((n, d) => n + d.count, 0);
    // Only the networks actually carrying traffic right now.
    const live = ([["p2p", "P2P"], ["relay", "NET"], ["local", "LOCAL"], ["radio", "RADIO"]] as [TransportKind, string][]).filter(([k]) => up(k));
    $("#paMeshText").innerHTML = live.length
      ? `MESH · ${total} ${live.map(([, l]) => `<span class="on">${l}</span>`).join(" ")}`
      : `MESH · <span class="off">connecting…</span>`;
    $("#paMeshBtn").classList.toggle("pa-mesh-off", !live.length);
    $("#paSheet").innerHTML = `<div class="pa-label">On the mesh</div>${
      devs
        .map((d) => `<div class="pa-dev"><span>${esc(d.name)}${d.count > 1 ? ` <span class="pa-dim">×${d.count}</span>` : ""}</span><span class="pa-chip">${d.route ? esc(TRANSPORT_LABEL[d.route]) : "via mesh hops"}</span></div>`)
        .join("") || `<div class="pa-dim">No other devices yet</div>`
    }`;
  }
  $("#paMeshBtn").addEventListener("click", () => $("#paSheet").classList.toggle("hidden"));

  // One handler for the action bar, so re-rendered buttons always work.
  $("#paActions").addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>("[data-v], [data-act]");
    if (!b || !shown) return;
    if (b.dataset.v) node.write("incidents", shown, { [`ack:${role}`]: Date.now(), [`status:${role}`]: b.dataset.v });
    // Care takes it from here: EMS paged on iMessage, hospital pre-alerted.
    else if (b.dataset.act === "injured") node.write("incidents", shown, { injuredReportedAt: Date.now() });
    else if (b.dataset.act === "resolve")
      node.write("incidents", shown, role === "police" ? { status: "resolved", resolvedAt: Date.now(), "status:police": "resolved" } : { [`status:${role}`]: "done" });
  });

  // ---- wiring -----------------------------------------------------------------
  const frameCounts = new Map<string, number>();
  node.store.onChange((c, id) => {
    if (c === "frames" && !frameArrived.has(id)) frameArrived.set(id, Date.now());
    if (c === "frames")
      queueMicrotask(() => {
        if (!shown) return;
        const n = framesFor(node.store, shown).length;
        const before = frameCounts.get(shown) ?? n;
        frameCounts.set(shown, n);
        if (n > before) newPunch();
      });
    if (c === "incidents" || c === "frames" || c === "agentchat") queueMicrotask(render);
  });
  node.onLinksChanged(renderMesh);
  setInterval(renderMesh, 1500);
  setInterval(() => {
    tickAgo();
    if (shown && !pickShown()) render(); // went stale
  }, 1000);
  renderMesh();
  render();
}
