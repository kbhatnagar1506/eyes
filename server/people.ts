// The public side of Eyes over iMessage: everyone who isn't a responder.
//
// - Opt-in: anyone texts the Eyes number a place ("I'm at Five Points",
//   "join 55 Gilmer St") and gets alerts for that area. STOP removes them.
//   Nobody is texted who didn't ask to be.
// - Area alert: when an incident starts, everyone signed up within
//   ALERT_RADIUS_M gets a Loud-effect alert with how far away it is, which
//   way, a safe place to go, and a one-tap "Are you safe?" poll.
// - Care: someone answers "I need help", texts that a person is hurt, or the
//   officer on scene taps "Injured". EMS gets paged on iMessage with the
//   location (and the latest camera frame), the person gets first-aid
//   guidance and a live line to Dispatch, and everything they say is relayed
//   to EMS. When EMS replies "on my way", the person hears it.
// - All clear: when police mark the incident resolved, everyone alerted is told.
//
// Phone numbers never go on the mesh (every device that joins could read
// them). The map gets counts and approximate dots only.

import { createHash } from "node:crypto";
import { SQL } from "bun";
import type { MeshNode } from "../src/mesh/node";
import { matchLocal, searchOsm } from "../src/mesh/geocode";
import { metersBetween } from "../src/mesh/radio";
import { problemLabel, type MartaData, type Place } from "../src/mesh/scenario";

export const ALERT_RADIUS_M = 800;
const ALERT_LIVE_MS = 30 * 60_000; // replies to an alert count this long
const CARE_LIVE_MS = 45 * 60_000;
const SAFE_SPOT_MIN_M = 350; // a "safe spot" is at least this far from the scene
const POLL_TITLE = "Are you safe?";
export const OPT_SAFE = "I'm safe";
export const OPT_HELP = "I need help";

// What we send, before it becomes iMessage content (agents.ts does that), so
// dry runs and tests can read it.
export type Msg = string | { text: string; loud: true } | { poll: string; options: string[] } | { photo: string; caption: string };
export const msgText = (m: Msg) => (typeof m === "string" ? m : "text" in m ? `[loud] ${m.text}` : "poll" in m ? `[poll] ${m.poll} ${m.options.join(" / ")}` : `[photo] ${m.caption}`);
type At = { lat: number; lng: number };
// A native tapback on the person's own message, when the platform has one.
type React = (emoji: string) => Promise<void>;

export interface PeopleDeps {
  node: MeshNode;
  log: (...a: unknown[]) => void;
  places: Place[];
  stops: MartaData["stops"];
  // One message (or several) to one person. Logs only in a dry run.
  send: (handle: string, ...content: Msg[]) => Promise<void>;
  // Same, to the EMS line.
  sendEms: (...content: Msg[]) => Promise<void>;
  typing: <T>(handle: string, fn: () => Promise<T>) => Promise<T>;
  // The model; throws when unreachable (callers fall back to fixed text).
  ask: (prompt: string) => Promise<string>;
  // A line from an agent in the responders' thread (queued behind the others).
  say: (incId: string, agent: "care" | "safety", extra: string) => void;
  nearestOf: (cat: Place["cat"], at: At) => { name: string; distanceM: number; etaMin: number } | null;
  latestIncident: () => string | null;
  latestFramePath: (incId: string) => Promise<string | null>;
  databaseUrl?: string;
}

interface Subscriber {
  handle: string;
  place: string;
  lat: number;
  lng: number;
}

interface Alerted {
  handle: string;
  status: "notified" | "safe" | "help";
  distanceM: number;
  at: number;
}

interface CareCase {
  incId: string;
  handle: string | null; // null: reported by the officer, nobody to text
  place: string;
  at: At;
  openedAt: number;
  transcript: { who: "person" | "care"; text: string }[];
}

const mask = (h: string) => (h.length > 6 ? `${h.slice(0, 4)}…${h.slice(-4)}` : "…");
const fmtDist = (m: number) => (m < 1000 ? `${Math.round(m / 10) * 10} m` : `${(m / 1000).toFixed(1)} km`);
const anon = (h: string) => createHash("sha256").update(h).digest("hex").slice(0, 10);

function direction(from: At, to: At): string {
  const y = (to.lng - from.lng) * Math.cos((from.lat * Math.PI) / 180);
  const x = to.lat - from.lat;
  const deg = ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
  return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][Math.round(deg / 45) % 8];
}

// ---- what people text us ---------------------------------------------------------

const STOP = /^\s*(stop|unsubscribe|leave|quit|cancel)\s*[.!]*\s*$/i;
const JOIN = /^\s*(?:join|start|subscribe|alerts?|watch)\b[\s:,-]*(.*)$/i;
const IM_AT = /\b(?:i'?m|i am|im)\s+(?:at|near|by|around)\s+(.+)$/i;
const SAFE = /^\s*(?:i'?m |im |i am )?(?:safe|ok|okay|fine|good|all good|yes|yep|👍)\b/i;
const HELP = /\b(help|sos|hurt|injur\w*|bleed\w*|scared)\b/i;
const HURT = /\b(hurt|injur\w*|bleed\w*|blood|unconscious|passed out|not breathing|ambulance|knocked out|wounded|stabbed|broken)\b/i;
export const RESPONDING = /\b(on (my|the) way|omw|en ?route|responding|heading (there|over)|coming|i'?m there|arriv(ed|ing))\b/i;

// What to do near each kind of incident (also the fallback when the model is down).
export const GUIDANCE: Record<string, string> = {
  assault: "Avoid the area and don't step in; stay in busy, well-lit places.",
  robbery: "Avoid the area, keep your phone and bag out of sight, and stay with other people.",
  harassment: "Stay with other people in a busy, well-lit place and avoid the area.",
  medical: "Keep the area clear for EMS. If you're trained in first aid and close by, you can help.",
  fire: "Move away from the smoke and keep the street clear for fire crews.",
  package: "Keep well back from the area and don't touch any unattended bags.",
};

const FIRST_AID = `First aid you may give (standard lay first aid, nothing else):
- Anyone not breathing, unconscious, bleeding heavily, with a head injury and confusion, or chest pain: tell them to call 911 now; the 911 dispatcher will guide CPR.
- Bleeding: firm, direct pressure with a clean cloth; if it soaks through, add more on top, don't remove it.
- Possible head, neck or back injury: don't move them unless they're in danger.
- Unconscious but breathing (and no neck injury suspected): recovery position, on their side.
- Stay with them, keep them warm and calm, no food or drink.
Never diagnose, never suggest medicine, never say help has arrived unless the facts say so.`;

export function createPeople(d: PeopleDeps) {
  const subscribers = new Map<string, Subscriber>();
  const awaitingPlace = new Set<string>();
  const alerts = new Map<string, Map<string, Alerted>>(); // incident -> handle -> state
  const cases: CareCase[] = [];
  const allClearSent = new Set<string>();
  const areaAlerted = new Set<string>();
  const injuredSeen = new Set<string>();
  const sql = d.databaseUrl ? new SQL(d.databaseUrl) : null;

  // ---- subscribers (Postgres when there is one, so they survive restarts) ------

  async function load() {
    if (!sql) return;
    try {
      await sql.unsafe(`create table if not exists subscribers (
        handle    text primary key,
        place     text not null,
        lat       double precision not null,
        lng       double precision not null,
        joined_at timestamptz not null default now()
      )`);
      const rows: Subscriber[] = await sql`select handle, place, lat, lng from subscribers`;
      for (const r of rows) subscribers.set(r.handle, { ...r, lat: Number(r.lat), lng: Number(r.lng) });
      d.log(`people: ${subscribers.size} signed up for area alerts`);
    } catch (err) {
      d.log(`people: subscribers not stored (${String(err).slice(0, 160)}), keeping them in memory`);
    }
  }

  async function subscribe(handle: string, placeText: string) {
    awaitingPlace.delete(handle);
    const hit = matchLocal(placeText, d.places, d.stops) ?? (await searchOsm(placeText));
    if (!hit) {
      awaitingPlace.add(handle);
      await d.send(handle, `I couldn't find "${placeText}" in Atlanta. Try a landmark, a MARTA station, or a street address.`);
      return;
    }
    const sub = { handle, place: hit.name, lat: hit.lat, lng: hit.lng };
    subscribers.set(handle, sub);
    d.log(`people: ${mask(handle)} watching ${hit.name}`);
    if (sql) await sql`insert into subscribers (handle, place, lat, lng) values (${handle}, ${sub.place}, ${sub.lat}, ${sub.lng})
      on conflict (handle) do update set place = excluded.place, lat = excluded.lat, lng = excluded.lng`.catch(() => {});
    await d.send(
      handle,
      `✅ You're on Eyes alerts within ${fmtDist(ALERT_RADIUS_M)} of ${hit.name}. If something happens nearby I'll text you right away and check that you're OK.\n\nText a new place to move it, or STOP to leave.`,
    );
  }

  async function unsubscribe(handle: string) {
    subscribers.delete(handle);
    awaitingPlace.delete(handle);
    if (sql) await sql`delete from subscribers where handle = ${handle}`.catch(() => {});
    d.log(`people: ${mask(handle)} left`);
    await d.send(handle, "You're off Eyes alerts. Text JOIN and a place any time to come back.");
  }

  // ---- area alert -------------------------------------------------------------------

  function safeSpot(person: At, scene: At) {
    const fromScene = metersBetween(person, scene);
    const minFromScene = Math.max(SAFE_SPOT_MIN_M, fromScene);
    const candidates = [
      ...d.places.filter((p) => ["police", "hospital", "fire", "library", "mall", "university"].includes(p.cat)),
      // Station names without the bus bay ("Georgia State Station - Bay D").
      ...d.stops.filter((s) => s[3]).map(([lat, lng, name]) => ({ lat, lng, name: `${name.replace(/\s*-\s*Bay\s+\w+$/i, "")} (MARTA)` })),
    ];
    let best: { name: string; m: number } | null = null;
    for (const c of candidates) {
      if (metersBetween(c, scene) < minFromScene) continue;
      const m = metersBetween(c, person);
      if (!best || m < best.m) best = { name: c.name, m };
    }
    return best;
  }

  async function guidanceFor(inc: Record<string, unknown>) {
    const problem = problemLabel(inc.kind);
    try {
      return await d.ask(`You are the public-safety agent of Eyes, a safety network in Atlanta. A ${problem} was just detected at ${String(inc.label ?? "a nearby location")}. Write ONE calm, concrete instruction (under 140 characters, plain text, no emoji) for people up to ${ALERT_RADIUS_M} m away: what to do right now.
Keep it proportionate to a ${problem.toLowerCase()}: don't cause panic or give lockdown or active-shooter advice unless the problem is a weapon threat. Good for an assault: "Avoid the area and don't step in; stay in busy, well-lit places." Don't mention distances or police; that's said separately.`);
    } catch {
      return GUIDANCE[String(inc.kind)] ?? "Avoid the area, stay in busy, well-lit places, and don't approach.";
    }
  }

  function writeArea(incId: string) {
    const m = alerts.get(incId);
    if (!m) return;
    const all = [...m.values()];
    d.node.write("incidents", incId, {
      areaRadiusM: ALERT_RADIUS_M,
      areaNotified: all.length,
      areaSafe: all.filter((a) => a.status === "safe").length,
      areaHelp: all.filter((a) => a.status === "help").length,
    });
  }

  // Where people who reported (rather than signed up) were, for the map.
  const reporterAt = new Map<string, At>();

  function writeDot(incId: string, handle: string, status: Alerted["status"]) {
    const sub = subscribers.get(handle) ?? reporterAt.get(handle);
    if (!sub) return;
    // ~100 m precision on the shared map; the exact spot goes only to EMS.
    d.node.write("people", `${incId}:${anon(handle)}`, {
      incident: incId,
      lat: Math.round(sub.lat * 1000) / 1000,
      lng: Math.round(sub.lng * 1000) / 1000,
      status,
      at: Date.now(),
    });
  }

  async function areaAlert(incId: string) {
    const inc = d.node.store.get("incidents", incId);
    if (!inc || areaAlerted.has(incId)) return;
    areaAlerted.add(incId);
    const scene = { lat: Number(inc.lat), lng: Number(inc.lng) };
    // A reporter (tracked already) has their own confirmation: don't alert them twice.
    const state = alerts.get(incId) ?? new Map<string, Alerted>();
    alerts.set(incId, state);
    const near = [...subscribers.values()]
      .filter((s) => !state.has(s.handle))
      .map((s) => ({ s, m: metersBetween(s, scene) }))
      .filter((x) => x.m <= ALERT_RADIUS_M)
      .sort((a, b) => a.m - b.m);
    if (!near.length) {
      d.log(`people: nobody signed up within ${ALERT_RADIUS_M} m of ${inc.label}`);
      writeArea(incId);
      return;
    }
    const guidance = await guidanceFor(inc);
    const police = d.nearestOf("police", scene);
    const problem = problemLabel(inc.kind);
    for (const { s, m } of near) state.set(s.handle, { handle: s.handle, status: "notified", distanceM: Math.round(m), at: Date.now() });
    writeArea(incId);
    d.log(`people: area alert to ${near.length} within ${ALERT_RADIUS_M} m`);
    await Promise.all(
      near.map(async ({ s, m }) => {
        writeDot(incId, s.handle, "notified");
        const spot = safeSpot(s, scene);
        const where = m < 60 ? `right where you are (${inc.label})` : `at ${inc.label}, ${fmtDist(m)} ${direction(s, scene)} of you`;
        const text = [
          `⚠️ EYES SAFETY ALERT`,
          `${problem} reported ${where}.`,
          guidance,
          spot ? `Nearest safe spot: ${spot.name}, ${fmtDist(spot.m)} from you.` : "",
          police ? `Police from ${police.name} have been paged.` : "Police have been paged.",
        ]
          .filter(Boolean)
          .join("\n");
        await d.send(s.handle, { text, loud: true }, { poll: POLL_TITLE, options: [OPT_SAFE, OPT_HELP] });
      }),
    );
    d.say(incId, "safety", `\nYou just texted a safety alert to ${near.length} ${near.length === 1 ? "person" : "people"} signed up within ${ALERT_RADIUS_M} m, with a safe spot each and an "Are you safe?" poll. Say so in the thread.`);
  }

  function liveAlertFor(handle: string): { incId: string; a: Alerted } | null {
    let best: { incId: string; a: Alerted } | null = null;
    for (const [incId, m] of alerts) {
      const a = m.get(handle);
      if (a && Date.now() - a.at < ALERT_LIVE_MS && (!best || a.at > best.a.at)) best = { incId, a };
    }
    return best;
  }

  async function checkIn(incId: string, a: Alerted, status: "safe" | "help", react?: React) {
    if (a.status === status) return;
    a.status = status;
    writeArea(incId);
    writeDot(incId, a.handle, status);
    d.log(`people: ${mask(a.handle)} is ${status}`);
    if (status === "safe") {
      // A ❤️ tapback on their message says it without another text.
      if (react) await react("❤️");
      else await d.send(a.handle, "Glad you're safe. Stay clear of the area; I'll text you when it's all clear.");
      return;
    }
    const sub = subscribers.get(a.handle);
    await startCare(incId, a.handle, sub ? { lat: sub.lat, lng: sub.lng } : null, sub?.place ?? null, "a person nearby, answering the safety alert");
  }

  // ---- care ------------------------------------------------------------------------

  const openCase = (handle: string) => cases.find((c) => c.handle === handle && Date.now() - c.openedAt < CARE_LIVE_MS) ?? null;

  async function startCare(incId: string, handle: string | null, at: At | null, place: string | null, source: string) {
    const inc = d.node.store.get("incidents", incId);
    if (!inc) return;
    if (handle && openCase(handle)) return;
    const where = at ?? { lat: Number(inc.lat), lng: Number(inc.lng) };
    const placeName = place ?? String(inc.label ?? "the scene");
    const c: CareCase = { incId, handle, place: placeName, at: where, openedAt: Date.now(), transcript: [] };
    cases.push(c);
    const injured = cases.filter((x) => x.incId === incId).length;
    const ems = d.nearestOf("hospital", where);
    const emsLine = ems ? `EMS paged from ${ems.name}, about ${ems.etaMin} min` : "EMS paged";
    d.node.write("incidents", incId, { injured, care: emsLine, careAt: Date.now() });
    d.log(`care: ${source} at ${placeName} (${injured} for this incident)`);

    // EMS first: the minutes that matter are theirs.
    const time = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    const page = [
      `🚑 EMS REQUEST · ${time}`,
      `Person hurt at ${placeName} (${problemLabel(inc.kind)}${place && place !== inc.label ? ` reported nearby at ${inc.label}` : ""}).`,
      `Reported by: ${source}.`,
      ems ? `Nearest: ${ems.name}, ETA ~${ems.etaMin} min. ED pre-alert sent.` : "",
      `Map: https://maps.apple.com/?daddr=${where.lat.toFixed(5)},${where.lng.toFixed(5)}`,
      handle ? "Their messages from the scene will follow here." : "",
      `Reply "on my way" to confirm.`,
    ]
      .filter(Boolean)
      .join("\n");
    const frame = await d.latestFramePath(incId);
    await d.sendEms({ text: page, loud: true });
    if (frame) await d.sendEms({ photo: frame, caption: `📸 Latest camera frame · ${inc.label}` });

    if (handle) {
      const first = `🆘 Help is coming to ${placeName}. ${ems ? `An ambulance from ${ems.name} has been requested (about ${ems.etaMin} min).` : "EMS has been requested."}\n\nIf anyone isn't breathing or is badly hurt, call 911 now.\n\nAre you hurt, or is someone else? Tell me what's happening and I'll pass it to the crew.`;
      c.transcript.push({ who: "care", text: first });
      await d.send(handle, first);
    }
    d.say(incId, "care", `\nNew care request: ${source}, at ${placeName}. ${emsLine}; ED pre-alerted.${handle ? " You are now texting the person directly with first-aid guidance and relaying what they say to EMS." : " Nobody to text: the officer is with them."} Post this in the thread.`);
  }

  async function careReply(c: CareCase, text: string) {
    c.transcript.push({ who: "person", text });
    await d.sendEms(`💬 From the scene (${c.place}): "${text}"`);
    const inc: Record<string, unknown> = d.node.store.get("incidents", c.incId) ?? {};
    const facts = {
      place: c.place,
      problemNearby: problemLabel(inc.kind),
      ems: d.nearestOf("hospital", c.at),
      emsStatus: inc["ack:ems"] ? String(inc["status:ems"] ?? "en route") : "requested, waiting for the crew to confirm",
      policeStatus: inc["ack:police"] ? String(inc["status:police"] ?? "responding") : "paged",
    };
    const reply = await d.typing(c.handle!, async () => {
      try {
        return await d.ask(`You are Care, the Eyes agent texting a person at an emergency scene in Atlanta until EMS arrives. Be calm, warm and brief.
FACTS (only source of truth; never invent ETAs or say a crew confirmed unless emsStatus says so):
${JSON.stringify(facts)}
${FIRST_AID}
CONVERSATION:
${c.transcript.map((l) => `${l.who === "care" ? "Care" : "Person"}: ${l.text}`).join("\n")}
Write Care's next text: at most 3 short sentences, under 300 characters, plain text, no emoji. Give the single most important instruction for what they just said, then one short question if you need to know more. Mention that you passed it to the EMS crew only if it matters to them.`);
      } catch {
        return HURT.test(text)
          ? "Keep firm pressure on any bleeding and don't move them if their head or neck may be hurt. If they're not breathing, call 911 now. The EMS crew has your message."
          : "Thank you, I've passed that to the EMS crew. Stay with them and stay where it's safe. Tell me if anything changes.";
      }
    });
    c.transcript.push({ who: "care", text: reply });
    await d.send(c.handle!, reply);
  }

  // ---- inbound -------------------------------------------------------------------------

  // Returns true when the message was for us (otherwise Dispatch answers it).
  async function onText(handle: string, raw: string, react?: React): Promise<boolean> {
    const text = raw.trim();
    if (STOP.test(text)) {
      await unsubscribe(handle);
      return true;
    }
    const join = text.match(JOIN);
    const imAt = text.match(IM_AT);
    if (join || imAt || awaitingPlace.has(handle)) {
      const place = (join?.[1] ?? imAt?.[1] ?? text).replace(/[.!?]+$/, "").trim();
      if (!place) {
        awaitingPlace.add(handle);
        await d.send(handle, `Welcome to Eyes 👁 Where should I watch out for you? Text a place, like "Five Points", "Georgia State Station" or "55 Gilmer St".`);
      } else await subscribe(handle, place);
      return true;
    }
    const c = openCase(handle);
    if (c) {
      await careReply(c, text);
      return true;
    }
    const live = liveAlertFor(handle);
    if (live && live.a.status !== "help") {
      if (HELP.test(text)) {
        await checkIn(live.incId, live.a, "help");
        if (HURT.test(text)) await careReply(openCase(handle)!, text);
        return true;
      }
      if (SAFE.test(text)) {
        await checkIn(live.incId, live.a, "safe", react);
        return true;
      }
    }
    return false;
  }

  async function onPollVote(handle: string, option: string, selected: boolean) {
    if (!selected) return;
    const live = liveAlertFor(handle);
    if (!live) return;
    if (option === OPT_SAFE) await checkIn(live.incId, live.a, "safe");
    else if (option === OPT_HELP) await checkIn(live.incId, live.a, "help");
  }

  // EMS replied "on my way": tell the people waiting for them.
  async function onEmsResponding(text: string) {
    const live = cases.filter((c) => Date.now() - c.openedAt < CARE_LIVE_MS);
    const incIds = [...new Set(live.map((c) => c.incId))];
    for (const incId of incIds) {
      const inc = d.node.store.get("incidents", incId);
      if (!inc || inc["ack:ems"]) continue;
      d.node.write("incidents", incId, { "ack:ems": Date.now(), "status:ems": "en route (replied on iMessage)", care: "EMS en route" });
      d.log(`care: EMS replied "${text}"`);
      for (const c of live.filter((x) => x.incId === incId && x.handle)) {
        const ems = d.nearestOf("hospital", c.at);
        const msg = `🚑 The ambulance${ems ? ` from ${ems.name}` : ""} is on the way${ems ? `, about ${ems.etaMin} min` : ""}. Stay where you are if it's safe; they know where to find you.`;
        c.transcript.push({ who: "care", text: msg });
        await d.send(c.handle!, msg);
      }
      const at = [...new Set(live.filter((x) => x.incId === incId).map((x) => x.place))].join(" and ");
      d.say(incId, "care", `\nUpdate: the EMS crew just confirmed on iMessage they're en route to the injured person at ${at}. Post a one-line update naming that place.`);
    }
    return incIds.length > 0;
  }

  const hasOpenCare = () => cases.some((c) => Date.now() - c.openedAt < CARE_LIVE_MS);

  // Officer taps "Injured" on the phone app; police resolve the incident.
  d.node.store.onChange((coll, id) => {
    if (coll !== "incidents") return;
    queueMicrotask(() => {
      const inc = d.node.store.get("incidents", id);
      if (!inc) return;
      if (inc.injuredReportedAt && !injuredSeen.has(id)) {
        injuredSeen.add(id);
        if (Date.now() - Number(inc.injuredReportedAt) < CARE_LIVE_MS)
          void startCare(id, null, null, null, "the officer on scene").catch((err) => d.log(`care failed: ${err}`));
      }
      if (inc.status === "resolved" && !allClearSent.has(id) && (alerts.get(id)?.size || cases.some((c) => c.incId === id && c.handle))) {
        allClearSent.add(id);
        void allClear(id).catch((err) => d.log(`all clear failed: ${err}`));
      }
    });
  });

  async function allClear(incId: string) {
    const inc: Record<string, unknown> = d.node.store.get("incidents", incId) ?? {};
    const handles = new Set([...(alerts.get(incId)?.keys() ?? []), ...cases.filter((c) => c.incId === incId && c.handle).map((c) => c.handle!)]);
    const time = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    d.log(`people: all clear to ${handles.size}`);
    for (const h of handles) await d.send(h, `✅ All clear. Police resolved the ${problemLabel(inc.kind).toLowerCase()} at ${inc.label} at ${time}. Thanks for checking in. You're still on Eyes alerts.`);
  }

  // Someone who reported an incident: their "I'm safe" / "I need help"
  // counts for it, and they get the all clear.
  function track(incId: string, handle: string, at: At) {
    if (!subscribers.has(handle)) reporterAt.set(handle, at);
    const state = alerts.get(incId) ?? new Map<string, Alerted>();
    alerts.set(incId, state);
    if (!state.has(handle)) state.set(handle, { handle, status: "notified", distanceM: 0, at: Date.now() });
    writeArea(incId);
    writeDot(incId, handle, "notified");
  }

  return { load, areaAlert, onText, onPollVote, onEmsResponding, hasOpenCare, startCare, track, subscribers };
}

export type People = ReturnType<typeof createPeople>;
