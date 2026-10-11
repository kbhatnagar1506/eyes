// Rehearses the public side end to end with fake people and no network:
// sign-ups, the area alert, poll answers, a care case relayed to EMS, EMS
// confirming, and the all-clear. Run: bun scripts/test/people.test.ts
import { readFileSync } from "node:fs";
import { MeshNode } from "../../src/mesh/node";
import { createPeople, msgText, OPT_HELP, OPT_SAFE } from "../../server/people";
import { metersBetween } from "../../src/mesh/radio";
import type { MartaData, Place } from "../../src/mesh/scenario";

const raw = JSON.parse(readFileSync("public/data/places.json", "utf8")) as { places: [number, number, string, Place["cat"]][] };
const places: Place[] = raw.places.map(([lat, lng, name, cat]) => ({ lat, lng, name, cat }));
const stops: MartaData["stops"] = JSON.parse(readFileSync("public/data/marta.json", "utf8")).stops;

const node = new MeshNode({ name: "test", role: "gateway", lat: 33.75, lng: -84.39 }, { id: "test", presence: false } as never);
const sent: { to: string; text: string }[] = [];
const said: string[] = [];
const people = createPeople({
  node,
  log: () => {},
  places,
  stops,
  send: async (h, ...c) => void c.forEach((x) => sent.push({ to: h, text: msgText(x) })),
  sendEms: async (...c) => void c.forEach((x) => sent.push({ to: "EMS", text: msgText(x) })),
  typing: (_h, fn) => fn(),
  ask: async () => {
    throw new Error("offline");
  },
  say: (_i, agent, extra) => void said.push(`${agent}: ${extra.trim().slice(0, 80)}`),
  nearestOf: (cat, at) => {
    const best = places.filter((p) => p.cat === cat).sort((a, b) => metersBetween(a, at) - metersBetween(b, at))[0];
    return best ? { name: best.name, distanceM: Math.round(metersBetween(best, at)), etaMin: 4 } : null;
  },
  latestIncident: () => "inc-1",
  latestFramePath: async () => null,
});

let failures = 0;
const expect = (ok: boolean, what: string) => {
  console.log(`${ok ? "✓" : "✗"} ${what}`);
  if (!ok) failures++;
};
const flush = () => new Promise((r) => setTimeout(r, 20));
const last = (to: string) => [...sent].reverse().find((s) => s.to === to)?.text ?? "";

const A = "+15550000001", B = "+15550000002", C = "+15550000003", FAR = "+15550000009";
await people.onText(A, "join");
expect(/Where should I watch/.test(last(A)), "JOIN with no place asks where");
await people.onText(A, "Georgia State Station");
expect(/You're on Eyes alerts within 800 m of Georgia State/.test(last(A)), "a place signs them up");
await people.onText(B, "I'm at Five Points");
expect(/Five Points/.test(last(B)), "\"I'm at Five Points\" signs up");
await people.onText(C, "join Woodruff Park");
await people.onText(FAR, "join Lenox Square");
expect(people.subscribers.size === 4, "four people signed up");
await people.onText(C, "join zzzz nowhere qqq");
expect(/couldn't find/.test(last(C)), "an unknown place is asked again");
await people.onText(C, "Woodruff Park");

// An assault at GSU Student Center East.
node.write("incidents", "inc-1", { kind: "assault", label: "GSU Student Center East", lat: 33.7527, lng: -84.3857, at: Date.now(), status: "detected" });
await people.areaAlert("inc-1");
const alerted = new Set(sent.filter((s) => /SAFETY ALERT/.test(s.text)).map((s) => s.to));
expect(alerted.has(A) && alerted.has(B) && alerted.has(C) && !alerted.has(FAR), `area alert reaches the 3 nearby, not Lenox (got ${[...alerted].join(", ")})`);
console.log("   ", sent.find((s) => s.to === B && /SAFETY ALERT/.test(s.text))?.text.replace(/\n/g, " ⏎ "));
expect(sent.some((s) => s.to === A && /poll/i.test(s.text) && s.text.includes(OPT_SAFE)), "each gets the Are you safe? poll");
expect(node.store.get("incidents", "inc-1")?.areaNotified === 3, "incident shows 3 alerted on the mesh");
expect(node.store.list("people").length === 3 && !JSON.stringify(node.store.list("people")).includes("555"), "map dots without phone numbers");

await people.onPollVote(A, OPT_SAFE, true);
expect(/Glad you're safe/.test(last(A)) && node.store.get("incidents", "inc-1")?.areaSafe === 1, "poll: I'm safe");
await people.onText(C, "ok");
expect(node.store.get("incidents", "inc-1")?.areaSafe === 2, "text \"ok\" counts as safe");

await people.onPollVote(B, OPT_HELP, true);
await flush();
expect(sent.some((s) => s.to === "EMS" && /EMS REQUEST/.test(s.text)), "I need help pages EMS");
expect(/Help is coming/.test(last(B)), "and tells them help is coming");
expect(Number(node.store.get("incidents", "inc-1")?.injured) === 1, "incident shows 1 injured");
await people.onText(B, "my friend is bleeding from his head");
expect(sent.some((s) => s.to === "EMS" && /From the scene.*bleeding/.test(s.text)), "what they say is relayed to EMS");
expect(/pressure|911/.test(last(B)), "and they get first-aid guidance");

await people.onEmsResponding("on my way");
expect(/ambulance.*on the way/i.test(last(B)), "EMS on my way reaches the person");
expect(!!node.store.get("incidents", "inc-1")?.["ack:ems"], "EMS shows as en route on the mesh");

node.write("incidents", "inc-1", { status: "resolved", resolvedAt: Date.now() });
await flush();
expect([A, B, C].every((h) => /All clear/.test(last(h))) && !/All clear/.test(last(FAR)), "all clear to everyone alerted");

await people.onText(A, "STOP");
expect(!people.subscribers.has(A) && /off Eyes alerts/.test(last(A)), "STOP removes them");

// Officer taps "Injured" on a new incident: EMS paged with nobody to text.
const before = sent.filter((s) => s.to === "EMS").length;
node.write("incidents", "inc-2", { kind: "assault", label: "Peachtree Center", lat: 33.759, lng: -84.387, at: Date.now() });
node.write("incidents", "inc-2", { injuredReportedAt: Date.now() });
await flush();
await flush();
expect(sent.filter((s) => s.to === "EMS").length > before, "officer-reported injury pages EMS");
console.log(`\n${said.length} thread updates queued: ${said.map((s) => s.split(":")[0]).join(", ")}`);
console.log(failures ? `${failures} FAILED` : "all passed");
process.exit(failures ? 1 : 0);
