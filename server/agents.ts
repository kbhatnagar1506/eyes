// Eyes dispatch agents: a team of AI agents that coordinate the response to
// every incident on the mesh and talk it through over real iMessage (Photon).
//
// - Joins the mesh as its own device ("Dispatch AI") and watches incidents.
// - For each new incident, the agents take turns (each sees the conversation
//   so far): Sentinel (the camera that saw it), Network (how the alert got
//   out), Dispatch (the plan), the responder (police / EMS / fire), Safety
//   (guidance for people nearby), then Dispatch again to close.
// - Facts come from our data (nearest responders, distances, ETAs, which
//   networks are up, who acknowledged); the model only words them.
// - Calm by default on iMessage (Photon): one live status card per incident,
//   edited in place as things change (edits don't buzz the phone); the
//   agents' reasoning as one threaded reply under it; native tapbacks for
//   acknowledgements. Only urgent changes (someone hurt, resolved) notify.
//   Every agent line still goes to the mesh, so the map shows the full thread.
// - The public side (people.ts): opt-in area alerts with an "Are you safe?"
//   poll, care for anyone hurt (EMS paged, first aid by text, relayed to the
//   crew), and an all-clear when police resolve it.
//
// Run: npm run agents   (dry run unless EYES_SEND=1; see .env.example)

import { mkdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { attachment, poll, Spectrum, type ContentInput, type Space } from "spectrum-ts";
import { effect, imessage } from "spectrum-ts/providers/imessage";
import { MeshNode } from "../src/mesh/node";
import { framesFor, safeJpeg } from "../src/mesh/frames";
import { accessToken, PROJECT_ID } from "./gcp";
import { startRecorder } from "./recorder";
import { HubClient } from "../src/mesh/hub-client";
import { metersBetween } from "../src/mesh/radio";
import { GATEWAY_POS, PROBLEMS, problemLabel, type MartaData, type Place, type PlaceCategory, type ProblemId } from "../src/mesh/scenario";
import { createIntake } from "./intake";
import { ALERT_RADIUS_M, createPeople, GUIDANCE, msgText, OPT_HELP, OPT_SAFE, RESPONDING, type Msg } from "./people";

// ---- config ------------------------------------------------------------------

const HUB_URL = process.env.EYES_HUB ?? "ws://127.0.0.1:8787/mesh";
const SEND = process.env.EYES_SEND === "1";
// Who gets the agents' thread over iMessage (Photon Spectrum Cloud): phone
// numbers or Apple ID emails, comma-separated.
const RECIPIENTS = (process.env.EYES_IMESSAGE_TO ?? "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
// Every punch's camera frame goes to police as a photo. Defaults to the thread.
const POLICE_TO = (process.env.EYES_POLICE_TO ?? "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
// EMS pages (injured people) go here. Defaults to the police line.
const EMS_TO = (process.env.EYES_EMS_TO ?? "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
const FRAME_DIR = join(tmpdir(), "eyes-frames");
const LOCATION = "us-central1";
const MODEL = process.env.EYES_MODEL ?? "gemini-2.5-flash";
// Incidents older than this when we first see them already happened before
// we were online; don't page anyone about them.
const STALE_MS = 90_000;
const DEVICE_STALE_MS = 12_000;
const URBAN_KMH = 25; // emergency vehicle average through downtown traffic
const DISPATCH_MIN = 1; // time to get rolling

const log = (...a: unknown[]) => console.log(new Date().toLocaleTimeString([], { hour12: false }), ...a);

// ---- data --------------------------------------------------------------------

const places: Place[] = (() => {
  const raw = JSON.parse(readFileSync(new URL("../public/data/places.json", import.meta.url), "utf8")) as { places: [number, number, string, PlaceCategory][] };
  const kept: Place[] = [];
  for (const [lat, lng, name, cat] of raw.places) {
    const p = { lat, lng, name, cat };
    if (kept.some((k) => k.cat === cat && metersBetween(k, p) < 60)) continue;
    kept.push(p);
  }
  return kept;
})();

const stops: MartaData["stops"] = JSON.parse(readFileSync(new URL("../public/data/marta.json", import.meta.url), "utf8")).stops;

function nearestOf(cat: PlaceCategory, at: { lat: number; lng: number }) {
  let best: Place | null = null;
  let bestM = Infinity;
  for (const p of places) {
    if (p.cat !== cat) continue;
    const m = metersBetween(p, at);
    if (m < bestM) {
      best = p;
      bestM = m;
    }
  }
  return best ? { name: best.name, kind: cat, distanceM: Math.round(bestM), etaMin: Math.round(DISPATCH_MIN + bestM / 1000 / (URBAN_KMH / 60)) } : null;
}

// ---- mesh ----------------------------------------------------------------------

const node = new MeshNode({ name: "Dispatch AI", role: "gateway", lat: GATEWAY_POS.lat, lng: GATEWAY_POS.lng }, { id: `gateway-${process.pid}` });
new HubClient(node, HUB_URL);
const startedAt = Date.now();
const arrivals = new Map<string, { transport: string; hops: number }>();
let lastDelivery: { transport: string; hops: number } | null = null;
node.onDeliver((e) => {
  if (e.colls.includes("incidents")) lastDelivery = { transport: e.transport, hops: e.hops };
});

function networkFacts() {
  const outage: Record<string, unknown> = node.store.get("control", "outage") ?? {};
  const now = Date.now();
  const devices = node.store
    .list("presence")
    .filter((p) => !p.virtual && p.id !== node.id && now - Number(p.seen ?? 0) < DEVICE_STALE_MS)
    .map((p) => ({ name: String(p.name), role: String(p.role) }));
  return {
    internet: outage.relay ? "DOWN" : "up",
    p2pWifi: outage.p2p ? "DOWN" : "up",
    meshRadio: outage.radio ? "DOWN" : "up",
    devicesOnline: devices,
  };
}

type Inc = Record<string, unknown> & { id: string };

function factsFor(inc: Inc) {
  const at = { lat: Number(inc.lat), lng: Number(inc.lng) };
  const problem = PROBLEMS[inc.kind as ProblemId];
  const responders = (problem?.responders ?? ["police", "hospital"]).map((c) => nearestOf(c, at)).filter(Boolean);
  // Only real people who tapped a button on their phone. The agents in the
  // thread are AI and never count as a human acknowledging.
  const acks = Object.entries(inc)
    .filter(([k]) => k.startsWith("ack:") && k !== "ack:dispatch-ai")
    .map(([k]) => ({ person: k.slice(4), status: String(inc[`status:${k.slice(4)}`] ?? "acknowledged") }));
  return {
    problem: problemLabel(inc.kind),
    place: String(inc.label ?? "Atlanta"),
    time: new Date(Number(inc.at)).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" }),
    detectedBy: String(inc.detail ?? "CCTV"),
    alertArrivedHereVia: arrivals.get(inc.id) ?? null,
    responders,
    humansWhoAcknowledgedOnTheirPhone: acks,
    network: networkFacts(),
    // People who signed up for alerts near here (texted the Eyes number).
    peopleNearby:
      inc.areaNotified === undefined
        ? null
        : { alertedWithinM: Number(inc.areaRadiusM), alerted: Number(inc.areaNotified), saidSafe: Number(inc.areaSafe ?? 0), needHelp: Number(inc.areaHelp ?? 0) },
    injured: inc.injured ? { people: Number(inc.injured), care: String(inc.care ?? "") } : null,
  };
}

// ---- the model -------------------------------------------------------------------

async function generate(prompt: string): Promise<string> {
  const url = `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL}:generateContent`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${accessToken()}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      // Short operational messages, not reasoning: no thinking budget.
      generationConfig: { temperature: 0.5, maxOutputTokens: 160, thinkingConfig: { thinkingBudget: 0 } },
    }),
  });
  const data: any = await resp.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
  if (!resp.ok || !text) throw new Error(`model: ${resp.status} ${JSON.stringify(data).slice(0, 200)}`);
  return text.replace(/^["']|["']$/g, "").replace(/\s+/g, " ");
}

// ---- agents ----------------------------------------------------------------------

interface Agent {
  id: string;
  name: string;
  emoji: string;
  role: string;
  // Used when the model is unreachable, so the thread never stalls on stage.
  fallback: (f: ReturnType<typeof factsFor>) => string;
}

const RESPONDER_AGENT: Record<string, Agent> = {
  police: {
    id: "police",
    name: "Police",
    emoji: "🚓",
    role: "the police liaison agent for the nearest station. Say you are paging the on-duty officer at that station (name it from facts.responders) and their ETA once they accept. Never say an officer accepted unless facts.humansWhoAcknowledgedOnTheirPhone includes police.",
    fallback: (f) => `Paging the on-duty officer at ${f.responders[0]?.name}. ETA about ${f.responders[0]?.etaMin} min once they accept.`,
  },
  hospital: {
    id: "ems",
    name: "EMS",
    emoji: "🚑",
    role: "the EMS agent for the nearest hospital. Say you are requesting an ambulance from that hospital (name it from facts.responders) and its ETA. Never claim a crew confirmed unless a human in facts.humansWhoAcknowledgedOnTheirPhone did.",
    fallback: (f) => `Requesting an ambulance from ${f.responders[0]?.name}, ETA about ${f.responders[0]?.etaMin} min.`,
  },
  fire: {
    id: "fire",
    name: "Fire",
    emoji: "🚒",
    role: "the fire department agent for the nearest station. Say you are alerting that station (name it from facts.responders) and the engine ETA. Never claim a crew confirmed unless a human in facts.humansWhoAcknowledgedOnTheirPhone did.",
    fallback: (f) => `Alerting ${f.responders[0]?.name}, engine ETA about ${f.responders[0]?.etaMin} min.`,
  },
};

const INTAKE: Agent = {
  id: "intake",
  name: "Intake",
  emoji: "📝",
  role: "the intake agent who took a report from a person over iMessage. Say what they reported, where and when, in one factual sentence, quoting them briefly if useful.",
  fallback: (f) => `${f.problem} reported near ${f.place} at ${f.time} by someone on iMessage.`,
};
const SENTINEL: Agent = {
  id: "sentinel",
  name: "Sentinel",
  emoji: "👁",
  role: "the vision agent on the CCTV camera that detected the problem. Report what was detected, where and when, in one factual sentence.",
  fallback: (f) => `${f.problem} detected at ${f.place} at ${f.time}.`,
};
const DISPATCH: Agent = {
  id: "dispatch",
  name: "Dispatch",
  emoji: "🧭",
  role: "the dispatch coordinator. Give the plan: which responders you are paging (from facts.responders, nearest first, with ETAs) and what each should do on arrival. They are paged, not en route: only say someone is on the way if facts.humansWhoAcknowledgedOnTheirPhone includes them.",
  fallback: (f) => `Sending ${f.responders.map((r) => `${r?.name} (${r?.etaMin} min)`).join(" and ")}.`,
};
const SAFETY: Agent = {
  id: "safety",
  name: "Safety",
  emoji: "🧍",
  role: "the public-safety agent talking to people near the scene. Give one calm, concrete instruction fitting the problem (e.g. move away, stay in a lit area, keep the exit clear), and say help is coming.",
  fallback: (f) => `If you're near ${f.place}, move to a safe, lit area. Help is on the way.`,
};
const CARE: Agent = {
  id: "care",
  name: "Care",
  emoji: "🩹",
  role: "the care agent, an AI coordinating care by message until EMS arrives. Report in one or two sentences only what the update below says was done (EMS paged, hospital pre-alerted, who reported it). You are never physically with anyone; never say you are with the person, and don't claim you're texting anyone unless the update says so.",
  fallback: (f) => (f.injured ? `${f.injured.care}. ED pre-alerted; I'm with the person by text until the crew arrives.` : "Standing by for anyone hurt."),
};

type Line = { agent: string; emoji: string; text: string };

async function speak(agent: Agent, facts: ReturnType<typeof factsFor>, transcript: Line[], extra = ""): Promise<string> {
  const prompt = `You are ${agent.name}, ${agent.role}
You are part of Eyes, a real-time public-safety network in Atlanta. You are messaging the other agents in a shared iMessage thread that responders can read.

FACTS (the only source of truth; never invent names, numbers, units, or ETAs):
${JSON.stringify(facts, null, 1)}

THREAD SO FAR:
${transcript.map((l) => `${l.agent}: ${l.text}`).join("\n") || "(empty)"}
${extra}
Write your next message: one or two short sentences, under 200 characters, plain text, no emoji, no greeting, don't repeat what others already said.`;
  try {
    const text = await generate(prompt);
    return text.replace(new RegExp(`^${agent.name}\\s*:\\s*`, "i"), "");
  } catch (err) {
    log(`model unavailable for ${agent.name} (${String(err).slice(0, 120)}), using fallback`);
    return agent.fallback(facts);
  }
}

// ---- iMessage via Photon Spectrum Cloud --------------------------------------------
//
// One conversation per person with the Eyes number (Spectrum Pro lines are
// a shared pool, so DMs rather than a group). The thread recipients get the
// agents' conversation; police get every punch frame. Replies come back on
// the same Spectrum stream: questions get answered in the thread that asked,
// and a police reply like "on my way" counts as the officer responding.

type SpectrumApp = Awaited<ReturnType<typeof Spectrum>>;
let app: SpectrumApp | null = null;
let threadSpaces: Space[] = [];
let policeSpaces: Space[] = [];
let emsSpaces: Space[] = [];
// One conversation per person: the ones we open and the ones that text us.
const byHandle = new Map<string, Space>();
let openSpace: (handle: string) => Promise<Space | null> = async () => null;

const TYPING_MIN_MS = 500;

async function setupIMessage() {
  if (!SEND) {
    log("dry run: messages print here only. Set EYES_SEND=1 to send iMessages.");
    return;
  }
  if (!process.env.SPECTRUM_PROJECT_ID || !process.env.SPECTRUM_PROJECT_SECRET) {
    log("no Spectrum credentials (SPECTRUM_PROJECT_ID / SPECTRUM_PROJECT_SECRET in .env). Continuing as a dry run.");
    return;
  }
  if (!RECIPIENTS.length && !POLICE_TO.length) log("no responder recipients (EYES_IMESSAGE_TO / EYES_POLICE_TO): only the public side is on");
  try {
    app = await Spectrum({ providers: [imessage.config()], options: { logLevel: "warn" } });
  } catch (err) {
    log(`Spectrum unavailable: ${String(err).slice(0, 200)}. Continuing as a dry run.`);
    return;
  }
  const im = imessage(app);
  // Same number in several lists = same conversation.
  const open = async (handle: string) => {
    const existing = byHandle.get(handle);
    if (existing) return existing;
    const space = await im.space.create(await im.user(handle));
    byHandle.set(handle, space);
    return space;
  };
  openSpace = (handle) => open(handle).catch((err) => (log(`can't open a conversation: ${String(err).slice(0, 120)}`), null));
  threadSpaces = await Promise.all(RECIPIENTS.map(open));
  policeSpaces = POLICE_TO.length ? await Promise.all(POLICE_TO.map(open)) : threadSpaces;
  emsSpaces = EMS_TO.length ? await Promise.all(EMS_TO.map(open)) : policeSpaces;
  log(`Photon Spectrum: thread to ${threadSpaces.length}, police photos to ${policeSpaces.length}, EMS to ${emsSpaces.length} (iMessage)`);
  void listen(app);
}

async function listen(spectrum: SpectrumApp) {
  try {
    for await (const [space, message] of spectrum.messages) {
      if (message.sender?.kind === "agent") continue;
      const handle = message.sender?.id ?? space.id;
      if (!byHandle.has(handle)) byHandle.set(handle, space);
      const content = message.content;
      if (content.type === "poll_option") {
        void people.onPollVote(handle, content.option.title, content.selected).catch((err) => log(`poll vote failed: ${err}`));
        continue;
      }
      if (content.type !== "text") continue;
      void inbound(handle, content.text, space, message);
    }
  } catch (err) {
    log(`Spectrum stream ended: ${String(err).slice(0, 160)}`);
  }
}

// Responders' replies first ("on my way"), then the public side, then
// Dispatch answers whatever is left.
async function inbound(handle: string, raw: string, space?: Space, message?: Inbound) {
  const text = raw.trim();
  if (!text) return;
  const is = (list: Space[]) => (space ? list.some((s) => s.id === space.id) : false);
  // A native tapback instead of another text.
  const react = message ? (emoji: string) => message.react(emoji).then(() => undefined).catch(() => undefined) : undefined;
  try {
    if (RESPONDING.test(text)) {
      let handled = false;
      if (is(emsSpaces) && people.hasOpenCare()) handled = await people.onEmsResponding(text);
      if (is(policeSpaces) && latestIncident && !node.store.get("incidents", latestIncident)?.["ack:police"]) {
        // Same as the officer tapping Respond on the phone app.
        log(`police replied "${text}": marking responding`);
        node.write("incidents", latestIncident, { "ack:police": Date.now(), "status:police": "on the way" });
        handled = true;
      }
      if (handled) {
        await react?.("👍");
        return;
      }
    }
    if (await people.onText(handle, text, react)) return;
    // Reports, questions and everything else: the intake agent always answers.
    await intake.onText(handle, text, (q, id) => answer(q, space, message, id));
  } catch (err) {
    log(`reply failed: ${String(err).slice(0, 160)}`);
  }
}

// Typing bubbles while an agent is "thinking" (the model call).
async function whileTyping<T>(spaces: Space[], fn: () => Promise<T>): Promise<T> {
  if (!app || !spaces.length) return fn();
  await Promise.all(spaces.map((s) => s.startTyping().catch(() => {})));
  const started = Date.now();
  try {
    return await fn();
  } finally {
    const left = TYPING_MIN_MS - (Date.now() - started);
    if (left > 0) await new Promise((r) => setTimeout(r, left));
    await Promise.all(spaces.map((s) => s.stopTyping().catch(() => {})));
  }
}


// ---- the live status card ---------------------------------------------------------
//
// One message per incident per conversation, edited in place as responders
// move and people check in. iMessage allows 5 edits within 15 minutes; past
// that a fresh card goes out (and becomes the one that gets edited).

type Inbound = { react(emoji: string): Promise<unknown> };
type Sent = { edit(content: ContentInput): Promise<void>; reply(...content: ContentInput[]): Promise<unknown> };
interface Card { space: Space; msg: Sent | null; text: string; edits: number; firstAt: number }
const cards = new Map<string, Card[]>();
const EDIT_LIMIT = 5;
const EDIT_WINDOW_MS = 15 * 60_000;
const CARD_SETTLE_MS = 1200;
const ICON: Record<string, string> = { police: "🚓", hospital: "🚑", fire: "🚒" };
const ACK_KEY: Record<string, string> = { police: "police", hospital: "ems", fire: "fire" };

function cardText(inc: Inc) {
  const f = factsFor(inc);
  const resolved = inc.status === "resolved";
  const clock = (t: unknown) => new Date(Number(t)).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  if (resolved) {
    return [
      `✅ All clear near ${f.place}`,
      `The ${f.problem.toLowerCase()} reported at ${clock(inc.at)} was resolved at ${clock(inc.resolvedAt ?? Date.now())}.`,
      "",
      "The area is safe again. Thanks to everyone who responded.",
    ].join("\n");
  }
  const lines = [
    `🚨 ${f.problem} reported near ${f.place}`,
    `${clock(inc.at)} · ${inc.source === "imessage" ? `reported by ${Number(inc.reports ?? 1) > 1 ? `${Number(inc.reports)} people` : "someone"} nearby` : "spotted by an Eyes camera"}`,
    "",
    `⚠️ Be careful around ${f.place}. ${GUIDANCE[String(inc.kind)] ?? "Avoid the area for now and stay in busy, well-lit places."}`,
    "",
    "Help is on the way:",
  ];
  for (const r of f.responders) {
    if (!r) continue;
    const who = ACK_KEY[r.kind] ?? r.kind;
    const status = inc[`ack:${who}`] ? `🟢 ${String(inc[`status:${who}`] ?? "on the way").replace(/\s*\(.*\)$/, "")}` : "paged, waiting to confirm";
    lines.push(`${ICON[r.kind] ?? "•"} ${r.name} · about ${r.etaMin} min · ${status}`);
  }
  if (inc.areaNotified !== undefined) {
    const n = Number(inc.areaNotified);
    const help = Number(inc.areaHelp ?? 0);
    lines.push(n ? `👥 ${n} ${n === 1 ? "person" : "people"} within ${ALERT_RADIUS_M} m warned · ${Number(inc.areaSafe ?? 0)} safe${help ? ` · ${help} need help` : ""}` : `👥 Nobody signed up within ${ALERT_RADIUS_M} m yet`);
  }
  if (inc.injured) lines.push(`🩹 ${Number(inc.injured)} hurt · ${String(inc.care ?? "EMS paged")}`);
  const frames = framesFor(node.store, inc.id).length;
  if (frames) lines.push(`📸 ${frames} camera frame${frames === 1 ? "" : "s"} sent to police`);
  lines.push("", `Responding? Reply "on my way". Questions? Just ask.`);
  return lines.join("\n");
}

async function sendCard(incId: string, spaces: Space[]) {
  const inc = node.store.get("incidents", incId) as Inc;
  const text = cardText(inc);
  log(`card ${incId}:\n${text}`);
  const list: Card[] = [];
  for (const space of spaces) {
    let msg: Sent | null = null;
    if (app) {
      try {
        msg = ((await space.send(text)) as unknown as Sent | undefined) ?? null;
      } catch (err) {
        log(`send failed: ${String(err).slice(0, 160)}`);
      }
    }
    list.push({ space, msg, text, edits: 0, firstAt: Date.now() });
  }
  cards.set(incId, list);
}

const cardTimers = new Map<string, ReturnType<typeof setTimeout>>();
function refreshCard(incId: string) {
  if (!cards.has(incId) || cardTimers.has(incId)) return;
  cardTimers.set(
    incId,
    setTimeout(() => {
      cardTimers.delete(incId);
      void updateCard(incId).catch((err) => log(`card update failed: ${err}`));
    }, CARD_SETTLE_MS),
  );
}

async function updateCard(incId: string) {
  const inc = node.store.get("incidents", incId) as Inc | undefined;
  const list = cards.get(incId);
  if (!inc || !list) return;
  const text = cardText(inc);
  for (const card of list) {
    if (card.text === text) continue;
    card.text = text;
    const canEdit = card.msg && card.edits < EDIT_LIMIT && Date.now() - card.firstAt < EDIT_WINDOW_MS;
    log(`card ${incId} ${canEdit ? "edited" : "re-sent"}`);
    if (!app) continue;
    try {
      if (canEdit) {
        await card.msg!.edit(text);
        card.edits++;
      } else {
        card.msg = ((await card.space.send(text)) as unknown as Sent | undefined) ?? null;
        card.edits = 0;
        card.firstAt = Date.now();
      }
    } catch (err) {
      log(`card update failed: ${String(err).slice(0, 160)}`);
    }
  }
}

// A threaded reply under the incident's card (a plain send if there's no card).
async function replyIn(incId: string, content: ContentInput[], spaces: Space[] = threadSpaces) {
  for (const c of content) if (typeof c === "string") log(`↳ ${c.replace(/\n/g, " ⏎ ")}`);
  if (!app) return;
  const list = cards.get(incId) ?? [];
  for (const space of spaces) {
    const card = list.find((x) => x.space.id === space.id);
    try {
      if (card?.msg) await card.msg.reply(...content);
      else await space.send(...(content as [ContentInput]));
    } catch (err) {
      log(`reply failed: ${String(err).slice(0, 160)}`);
    }
  }
}

// ---- the conversation ------------------------------------------------------------

const handled = new Set<string>();
const transcripts = new Map<string, Line[]>();
let latestIncident: string | null = null;
let queue: Promise<void> = Promise.resolve();

function post(incId: string, line: Line) {
  const t = transcripts.get(incId)!;
  t.push(line);
  // Mirror into the mesh so the map shows the same thread.
  node.write("agentchat", `${incId}:${String(t.length).padStart(3, "0")}`, { incident: incId, agent: line.agent, emoji: line.emoji, text: line.text, seq: t.length, at: Date.now() });
}

// One agent's turn: typing bubble while it thinks, its line on the mesh.
// Returns the line; the caller decides whether it reaches iMessage.
async function think(incId: string, agent: Agent, extra = "", typingIn: Space[] = threadSpaces): Promise<Line> {
  const inc = node.store.get("incidents", incId) as Inc;
  if (!transcripts.has(incId)) transcripts.set(incId, []);
  const text = await whileTyping(typingIn, () => speak(agent, factsFor(inc), transcripts.get(incId)!, extra));
  const line = { agent: agent.name, emoji: agent.emoji, text };
  post(incId, line);
  return line;
}

const fmtLine = (l: Line) => `${l.emoji} ${l.agent}: ${l.text}`;

// Photos for an incident wait only for its alert header, not the whole
// agent round: the officer sees the alert, then the frame right away.
const headerSent = new Map<string, { promise: Promise<void>; resolve: () => void }>();
function headerGate(incId: string) {
  let gate = headerSent.get(incId);
  if (!gate) {
    let resolve!: () => void;
    const promise = new Promise<void>((r) => (resolve = r));
    gate = { promise, resolve };
    headerSent.set(incId, gate);
  }
  return gate;
}

async function coordinate(incId: string) {
  const inc = node.store.get("incidents", incId) as Inc;
  latestIncident = incId;
  transcripts.set(incId, []);
  const facts = factsFor(inc);
  log(`incident ${incId}: ${facts.problem} at ${facts.place}`);
  // The card goes out first, instantly: police too, right before their photo.
  const everyone = [...new Map([...threadSpaces, ...policeSpaces].map((sp) => [sp.id, sp])).values()];
  await sendCard(incId, everyone);
  headerGate(incId).resolve();
  // People signed up nearby hear it now, alongside the agents' round.
  void people.areaAlert(incId).catch((err) => log(`area alert failed: ${err}`));
  if (inc.kind === "medical") void people.startCare(incId, null, null, null, "the camera (medical emergency)").catch((err) => log(`care failed: ${err}`));

  // The agents take turns (each sees the others' lines); the phone shows
  // them typing, then gets their whole brief as one reply under the card.
  const first = PROBLEMS[inc.kind as ProblemId]?.responders[0] ?? "police";
  const brief: Line[] = [await think(incId, inc.source === "imessage" ? INTAKE : SENTINEL)];
  const net = facts.network;
  const up = [net.internet === "up" && "internet", net.p2pWifi === "up" && "P2P Wi-Fi", net.meshRadio === "up" && "mesh radio"].filter(Boolean).join(" + ") || "the mesh";
  const netLine = { agent: "Network", emoji: "🛰", text: `Delivered over ${up} to ${net.devicesOnline.length} device${net.devicesOnline.length === 1 ? "" : "s"}.` };
  post(incId, netLine);
  brief.push(netLine);
  brief.push(await think(incId, DISPATCH));
  node.write("incidents", incId, { status: "dispatched", "ack:dispatch-ai": Date.now(), "status:dispatch-ai": "units assigned" });
  brief.push(await think(incId, RESPONDER_AGENT[first] ?? RESPONDER_AGENT.police));
  brief.push(await think(incId, SAFETY));
  await replyIn(incId, [brief.map(fmtLine).join("\n\n")]);
}

// A question texted to the Eyes number gets Dispatch's answer in that same
// conversation (each person has their own thread with the number).
async function answer(question: string, from?: Space, message?: Inbound, about: string | null = latestIncident) {
  if (!about) return;
  const incId = about;
  log(`reply in thread: ${question}`);
  post(incId, { agent: "Thread", emoji: "💬", text: question });
  const to = from ? [from] : threadSpaces;
  queue = queue
    .then(async () => {
      const line = await think(incId, DISPATCH, `\nSomeone just texted: "${question}". Answer them directly from the facts.`, to);
      const text = fmtLine(line);
      log(text);
      if (!app) return;
      const threaded = message as unknown as { reply?: (c: ContentInput) => Promise<unknown> } | undefined;
      for (const space of to) {
        try {
          if (threaded?.reply && space === from) await threaded.reply(text);
          else await space.send(text);
        } catch (err) {
          log(`answer failed: ${String(err).slice(0, 160)}`);
        }
      }
    })
    .catch((err) => log(`answer failed: ${err}`));
}

// A real person acting on their phone (officer taps Respond, rider says they
// need help) gets a Dispatch update in the thread, once per person.
const seenAcks = new Map<string, Set<string>>();
function onHumanAck(id: string, inc: Record<string, unknown>) {
  if (!transcripts.has(id)) return; // not one we're coordinating
  const seen = seenAcks.get(id) ?? new Set<string>();
  seenAcks.set(id, seen);
  for (const k of Object.keys(inc)) {
    if (!k.startsWith("ack:") || k === "ack:dispatch-ai" || seen.has(k)) continue;
    seen.add(k);
    const who = k.slice(4);
    post(id, { agent: "Dispatch", emoji: "🧭", text: `${who}: ${String(inc[`status:${who}`] ?? "acknowledged")}` });
  }
  // The card shows it (an edit: no new buzz). Resolving is worth one buzz.
  refreshCard(id);
  if (inc.status === "resolved" && !resolvedNoted.has(id)) {
    resolvedNoted.add(id);
    void replyIn(id, [`✅ Resolved. Thanks, everyone.`]);
  }
}
const resolvedNoted = new Set<string>();

node.store.onChange((c, id) => {
  if (c !== "incidents") return;
  // All of a new incident's fields land in one batch; look once it has.
  queueMicrotask(() => {
    const inc = node.store.get("incidents", id);
    if (inc && handled.has(id)) onHumanAck(id, inc);
    if (!inc || inc.at === undefined || handled.has(id)) return;
    handled.add(id);
    if (lastDelivery) arrivals.set(id, lastDelivery);
    if (Number(inc.at) < startedAt - STALE_MS || Date.now() - Number(inc.at) > STALE_MS) return;
    if (inc["ack:dispatch-ai"]) return; // another Dispatch AI already took it
    headerGate(id);
    queue = queue.then(() => coordinate(id)).catch((err) => log(`coordination failed: ${err}`));
  });
});

// ---- evidence frames → police -------------------------------------------------

const sentFrames = new Set<string>();
let photoQueue: Promise<void> = Promise.resolve();

async function frameFile(frameId: string): Promise<string | null> {
  const src = safeJpeg(node.store.get("frames", frameId)?.data);
  if (!src) return null;
  mkdirSync(FRAME_DIR, { recursive: true });
  const path = join(FRAME_DIR, `${frameId.replace(/[^\w-]/g, "_")}.jpg`);
  await Bun.write(path, Buffer.from(src.slice(src.indexOf(",") + 1), "base64"));
  return path;
}

async function sendFrame(frameId: string) {
  const f = node.store.get("frames", frameId);
  const incId = String(f?.incident ?? frameId);
  const gate = headerSent.get(incId);
  if (!f || !gate) return; // not an incident we're coordinating (stale, or another Dispatch AI has it)
  await gate.promise;
  const path = await frameFile(frameId);
  if (!path) return;
  const inc = node.store.get("incidents", incId);
  const seq = Number(f.seq ?? 1);
  const text = `📸 Evidence #${seq} · ${new Date(Number(f.at ?? Date.now())).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" })} · ${inc?.label ?? ""}`;
  log(`${text} [photo ${path}]`);
  if (transcripts.has(incId)) post(incId, { agent: "Sentinel", emoji: "👁", text: `Evidence frame #${seq} sent to police.` });
  if (!app) return;
  await replyIn(incId, [attachment(path), text], policeSpaces);
  refreshCard(incId);
}

node.store.onChange((c, id) => {
  if (c !== "frames") return;
  queueMicrotask(() => {
    if (sentFrames.has(id) || !node.store.get("frames", id)?.data) return;
    sentFrames.add(id);
    photoQueue = photoQueue.then(() => sendFrame(id)).catch((err) => log(`frame dispatch failed: ${err}`));
  });
});

// ---- the public side -----------------------------------------------------------------

const LOUD = "com.apple.MobileSMS.expressivesend.loud" as const;
const toContent = (m: Msg): ContentInput[] =>
  typeof m === "string" ? [m] : "text" in m ? [effect(m.text, LOUD)] : "poll" in m ? [poll(m.poll, m.options)] : [attachment(m.photo), m.caption];

async function sendTo(spaces: Space[], label: string, msgs: Msg[]) {
  for (const m of msgs) log(`→ ${label}: ${msgText(m).replace(/\n/g, " ⏎ ")}`);
  for (const space of spaces) {
    for (const m of msgs) {
      try {
        await space.send(...(toContent(m) as [ContentInput]));
      } catch (err) {
        log(`send to ${label} failed: ${String(err).slice(0, 160)}`);
      }
    }
  }
}

const sendToHandle = async (handle: string, ...content: Msg[]) => {
  const space = app ? await openSpace(handle) : null;
  await sendTo(space ? [space] : [], `${handle.slice(0, 4)}…${handle.slice(-4)}`, content);
};
const typingFor = async <T,>(handle: string, fn: () => Promise<T>) => {
  const space = byHandle.get(handle);
  return space ? whileTyping([space], fn) : fn();
};

const people = createPeople({
  node,
  log,
  places,
  stops,
  send: sendToHandle,
  sendEms: (...content) => sendTo(app ? emsSpaces : [], "EMS", content),
  typing: typingFor,
  ask: generate,
  say: (incId, agent, extra) => {
    // Safety's counts live on the card; someone hurt is worth a reply.
    if (agent === "safety") {
      post(incId, { agent: SAFETY.name, emoji: SAFETY.emoji, text: "Area alert sent to people signed up nearby." });
      refreshCard(incId);
      return;
    }
    queue = queue
      .then(async () => {
        const line = await think(incId, CARE, extra);
        await replyIn(incId, [fmtLine(line)]);
        refreshCard(incId);
      })
      .catch((err) => log(`${agent} update failed: ${err}`));
  },
  nearestOf,
  latestIncident: () => latestIncident,
  latestFramePath: async (incId) => {
    const frames = framesFor(node.store, incId);
    return frames.length ? frameFile(frames[frames.length - 1].id) : null;
  },
  databaseUrl: process.env.EYES_DATABASE_URL,
});

const intake = createIntake({ node, log, places, stops, ask: generate, send: sendToHandle, typing: typingFor, people, nearestOf });

// Dry run: type "<number> <message>" here to play a person texting the Eyes
// number ("+15550001 I'm at Five Points", "+15550001 [help]" for the poll).
async function dryRunConsole() {
  log(`dry-run console: type "<number> <message>" to text Eyes as someone (poll answers: [safe] / [help])`);
  for await (const line of console) {
    const m = line.trim().match(/^(\S+)\s+(.+)$/);
    if (!m) continue;
    const [, handle, text] = m;
    if (/^\[(safe|help)\]$/i.test(text)) await people.onPollVote(handle, /safe/i.test(text) ? OPT_SAFE : OPT_HELP, true);
    else {
      const asResponder = RECIPIENTS.includes(handle) || POLICE_TO.includes(handle) || EMS_TO.includes(handle);
      if (asResponder && RESPONDING.test(text)) {
        if (people.hasOpenCare() && (EMS_TO.includes(handle) || (!EMS_TO.length && POLICE_TO.includes(handle)))) await people.onEmsResponding(text);
        continue;
      }
      await inbound(handle, text);
    }
  }
}

await setupIMessage();
await people.load();
if (!app) void dryRunConsole();
if (process.env.EYES_DATABASE_URL) {
  await startRecorder(node, process.env.EYES_DATABASE_URL, log).catch((err) => log(`recorder off: ${String(err).slice(0, 200)}`));
} else log("recorder off: set EYES_DATABASE_URL to keep incidents in Postgres");
log(`Dispatch AI on the mesh as ${node.id}, hub ${HUB_URL}`);

process.on("SIGINT", async () => {
  await app?.stop();
  process.exit(0);
});
