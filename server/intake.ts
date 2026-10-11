// The front door of the Eyes line. Anything that isn't a responder's reply
// or an alert sign-up lands here. A model reads it (keywords if the model is
// down) and decides what the person needs:
//
// - A report ("there's a fight at Five Points", "someone collapsed outside
//   Peachtree Center", "send an ambulance"): it becomes a real incident on
//   the mesh at that place, so everything runs as if a camera had seen it:
//   the responders' card, dispatch, people nearby warned, care if someone's
//   hurt. The reporter gets a confirmation, a 911 reminder and an "Are you
//   safe?" poll. No place given: it asks where (or uses their saved place
//   when they say "here"). A second report near a live incident joins it.
// - A question about what's going on: a grounded answer about the live incident.
// - Anything else: a short, friendly reply that says what Eyes can do.
//
// Eyes can't place phone calls; whenever there's danger it says to call 911.

import type { MeshNode } from "../src/mesh/node";
import { matchLocal, searchOsm } from "../src/mesh/geocode";
import { metersBetween } from "../src/mesh/radio";
import { PROBLEMS, problemFromText, problemLabel, type MartaData, type Place, type ProblemId } from "../src/mesh/scenario";
import { OPT_HELP, OPT_SAFE, type Msg, type People } from "./people";

const PENDING_MS = 10 * 60_000; // how long we wait for "where?"
const LIVE_MS = 30 * 60_000; // an incident counts as ongoing this long
const SAME_PLACE_M = 400; // a report this close to a live incident joins it
const REPORT_GAP_MS = 2 * 60_000; // one new incident per person per 2 minutes
const HISTORY = 6; // recent lines kept per person, for natural follow-ups

type Kind = ProblemId;
interface Classified {
  intent: "report" | "question" | "chat";
  kind: Kind | null;
  place: string | null;
  hurt: boolean;
}
type At = { lat: number; lng: number; name: string };

export interface IntakeDeps {
  node: MeshNode;
  log: (...a: unknown[]) => void;
  places: Place[];
  stops: MartaData["stops"];
  ask: (prompt: string) => Promise<string>;
  send: (handle: string, ...msgs: Msg[]) => Promise<void>;
  typing: <T>(handle: string, fn: () => Promise<T>) => Promise<T>;
  people: People;
  nearestOf: (cat: Place["cat"], at: { lat: number; lng: number }) => { name: string; etaMin: number } | null;
}

// Grounded answer about a live incident, in the asker's own conversation.
type AnswerIncident = (question: string, incId: string) => Promise<void>;

const HURT = /\b(hurt|injur\w*|bleed\w*|blood|unconscious|passed out|collapsed?|not breathing|ambulance|seizure|overdos\w*|stabbed|shot)\b/i;
const HELPWORDS = /\b(help|police|cops|911|emergency|danger|attack\w*|fight\w*|threat\w*|following me|followed|gun|knife|weapon)\b/i;
const HERE = /\b(here|near me|where i am|my location|right now)\b/i;

export function createIntake(d: IntakeDeps) {
  const pending = new Map<string, { c: Classified; text: string; at: number }>();
  const lastReport = new Map<string, number>();
  const history = new Map<string, string[]>();

  const remember = (handle: string, line: string) => {
    const h = history.get(handle) ?? [];
    h.push(line);
    history.set(handle, h.slice(-HISTORY));
  };

  function liveIncidents() {
    const now = Date.now();
    return d.node.store
      .list("incidents")
      .filter((i) => i.status !== "resolved" && now - Number(i.at) < LIVE_MS && i.lat !== undefined)
      .sort((a, b) => Number(b.at) - Number(a.at));
  }

  // ---- understanding the message --------------------------------------------

  function keywords(text: string): Classified {
    const kind = problemFromText(text);
    const hurt = HURT.test(text);
    const place = text.match(/\b(?:at|near|by|outside|inside|in front of|on)\s+(.+?)(?:[.!?]|$)/i)?.[1] ?? null;
    const report = !!kind || hurt || HELPWORDS.test(text);
    return { intent: report ? "report" : text.trim().endsWith("?") ? "question" : "chat", kind: kind ?? (hurt ? "medical" : null), place, hurt };
  }

  async function classify(text: string, handle: string): Promise<Classified> {
    const kinds = Object.keys(PROBLEMS).join(" | ");
    try {
      const raw = await d.ask(`Classify a text message sent to Eyes, a public-safety service in Atlanta. Earlier messages from this person, oldest first: ${JSON.stringify(history.get(handle) ?? [])}
Message: ${JSON.stringify(text)}
Return ONLY a JSON object, no prose: {"intent": "report" | "question" | "chat", "kind": ${kinds} | null, "place": string | null, "hurt": boolean}
- "report": they are telling us about something happening now (a fight, someone hurt or collapsed, a threat, being followed, a fire, an unattended bag) or asking for police, an ambulance or help.
- "question": asking about an incident or alert.
- "chat": anything else (greetings, thanks, what is this, how does it work).
- "kind": the closest kind for a report, else null. "place": the location words exactly as they wrote them (landmark, station, street, cross streets), null if none. "hurt": someone is injured or needs medical help.`);
      const json = JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1));
      const kind = (Object.keys(PROBLEMS) as Kind[]).includes(json.kind) ? (json.kind as Kind) : null;
      const intent = ["report", "question", "chat"].includes(json.intent) ? json.intent : "chat";
      return { intent, kind, place: typeof json.place === "string" && json.place.trim() ? json.place.trim() : null, hurt: !!json.hurt };
    } catch {
      return keywords(text);
    }
  }

  async function locate(placeText: string): Promise<At | null> {
    const hit = matchLocal(placeText, d.places, d.stops) ?? (await searchOsm(placeText));
    return hit ? { lat: hit.lat, lng: hit.lng, name: hit.name } : null;
  }

  // ---- reports -----------------------------------------------------------------

  async function report(handle: string, text: string, c: Classified, placeText = c.place) {
    const saved = d.people.subscribers.get(handle);
    let where = placeText && !HERE.test(placeText) ? await locate(placeText) : null;
    if (!where && saved && (!placeText || HERE.test(placeText) || HERE.test(text))) where = { lat: saved.lat, lng: saved.lng, name: saved.place };
    if (!where) {
      pending.set(handle, { c, text, at: Date.now() });
      await d.send(handle, `📍 Where is this happening? Send a place, a cross street or a landmark, like "Five Points" or "Peachtree & 10th".\n\nIf anyone is in danger right now, call 911.`);
      return;
    }
    pending.delete(handle);
    const kind: Kind = c.kind ?? (c.hurt ? "medical" : "harassment");

    // Already on it? Join the live incident instead of starting a second one.
    const dup = liveIncidents().find((i) => metersBetween({ lat: Number(i.lat), lng: Number(i.lng) }, where!) <= SAME_PLACE_M);
    if (dup) {
      d.node.write("incidents", dup.id, { reports: Number(dup.reports ?? 1) + 1 });
      d.people.track(dup.id, handle, where);
      d.log(`intake: report joins ${dup.id} near ${dup.label}`);
      await d.send(handle, `Thank you. That matches an alert already active near ${dup.label}: responders are on it and I've added your report.\n\nIf anyone is in danger right now, call 911.`, { poll: "Are you safe?", options: [OPT_SAFE, OPT_HELP] });
      if (c.hurt) await d.people.startCare(dup.id, handle, where, where.name, "the person who reported it");
      return;
    }
    const last = lastReport.get(handle) ?? 0;
    if (Date.now() - last < REPORT_GAP_MS) {
      await d.send(handle, "I've just raised an alert from your last message. If this is something new, tell me where in a minute. If anyone is in danger right now, call 911.");
      return;
    }
    lastReport.set(handle, Date.now());

    const id = `rep-${Date.now().toString(36)}`;
    d.node.write("incidents", id, {
      kind,
      label: where.name,
      detail: `Reported over iMessage: "${text.slice(0, 160)}"`,
      source: "imessage",
      reports: 1,
      lat: where.lat,
      lng: where.lng,
      at: Date.now(),
      status: "reported",
    });
    d.people.track(id, handle, where);
    d.log(`intake: new ${kind} report ${id} at ${where.name}`);
    const paged = (PROBLEMS[kind]?.responders ?? ["police"])
      .map((cat) => d.nearestOf(cat, where!))
      .filter(Boolean)
      .map((r) => `${r!.name} (about ${r!.etaMin} min)`);
    await d.send(
      handle,
      `🚨 Got it. I've raised an alert for ${problemLabel(kind).toLowerCase()} near ${where.name}.\n\nPaging ${paged.join(" and ") || "the nearest responders"} now, and warning people nearby.\n\nIf anyone is in danger right now, call 911.`,
      { poll: "Are you safe?", options: [OPT_SAFE, OPT_HELP] },
    );
    if (c.hurt) await d.people.startCare(id, handle, where, where.name, "the person who reported it");
  }

  // ---- conversation ----------------------------------------------------------------

  async function chat(handle: string, text: string) {
    const live = liveIncidents()[0];
    const saved = d.people.subscribers.get(handle);
    const reply = await d.typing(handle, async () => {
      try {
        return await d.ask(`You are Eyes, the public-safety agent on this iMessage line in Atlanta. Be warm, calm and brief.
What Eyes does: cameras and people report danger; Eyes pages the nearest police and EMS, warns people nearby, and helps anyone hurt until EMS arrives.
What you can do in this chat: take a report (they just say what's happening and where), sign them up for alerts near a place (they text JOIN and a place, STOP to leave), answer questions about an active incident.
You cannot place phone calls. If anything sounds like danger, tell them to call 911.
Their saved alert place: ${saved ? saved.place : "none"}. Active incident: ${live ? `${problemLabel(live.kind)} near ${live.label}` : "none"}.
Recent conversation, oldest first: ${JSON.stringify(history.get(handle) ?? [])}
They just wrote: ${JSON.stringify(text)}
Reply in 1 to 3 short sentences, plain text, no markdown, at most one emoji. Never invent incidents, numbers or ETAs.`);
      } catch {
        return `👁 This is Eyes, the safety line for Atlanta. Tell me what's happening and where, and I'll get help moving. Text JOIN and a place to get alerts nearby. If anyone is in danger, call 911.`;
      }
    });
    remember(handle, `Eyes: ${reply}`);
    await d.send(handle, reply);
  }

  // Everything that reaches the front door ends here, so it always gets an answer.
  async function onText(handle: string, text: string, answerIncident: AnswerIncident) {
    remember(handle, `Person: ${text}`);
    const wait = pending.get(handle);
    if (wait && Date.now() - wait.at < PENDING_MS) {
      // Their answer to "where?".
      d.log(`intake: place for pending report: ${text}`);
      await report(handle, wait.text, wait.c, text);
      return;
    }
    const c = await classify(text, handle);
    d.log(`intake: ${c.intent}${c.kind ? ` (${c.kind})` : ""}${c.place ? ` at "${c.place}"` : ""}${c.hurt ? ", someone hurt" : ""}`);
    if (c.intent === "report") return report(handle, text, c);
    const live = liveIncidents()[0];
    if (c.intent === "question" && live) return answerIncident(text, live.id);
    return chat(handle, text);
  }

  return { onText };
}
