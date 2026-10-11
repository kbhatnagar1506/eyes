// The Eyes line's front door, offline: reports with and without a place,
// "here" resolving to a saved place, a duplicate joining a live incident,
// someone hurt starting care, questions and chat. Runs once with the model
// answering in JSON and once with the model down (keyword fallback).
// Run: bun scripts/test/intake.test.ts
import { readFileSync } from "node:fs";
import { MeshNode } from "../../src/mesh/node";
import { createIntake } from "../../server/intake";
import { createPeople, msgText } from "../../server/people";
import { metersBetween } from "../../src/mesh/radio";
import type { MartaData, Place } from "../../src/mesh/scenario";

const raw = JSON.parse(readFileSync("public/data/places.json", "utf8")) as { places: [number, number, string, Place["cat"]][] };
const places: Place[] = raw.places.map(([lat, lng, name, cat]) => ({ lat, lng, name, cat }));
const stops: MartaData["stops"] = JSON.parse(readFileSync("public/data/marta.json", "utf8")).stops;
const nearestOf = (cat: Place["cat"], at: { lat: number; lng: number }) => {
  const best = places.filter((p) => p.cat === cat).sort((a, b) => metersBetween(a, at) - metersBetween(b, at))[0];
  return best ? { name: best.name, distanceM: Math.round(metersBetween(best, at)), etaMin: 3 } : null;
};

let failures = 0;
const expect = (ok: boolean, what: string) => {
  console.log(`${ok ? "✓" : "✗"} ${what}`);
  if (!ok) failures++;
};

async function run(modelUp: boolean) {
  console.log(`\n— model ${modelUp ? "up" : "down"} —`);
  const node = new MeshNode({ name: "t", role: "gateway", lat: 33.75, lng: -84.39 }, { id: `t-${modelUp}`, presence: false } as never);
  const sent: { to: string; text: string }[] = [];
  const send = async (h: string, ...m: Parameters<typeof msgText>[0][]) => void m.forEach((x) => sent.push({ to: h, text: msgText(x) }));
  // A stand-in model: classifies by keywords into the JSON the prompt asks for.
  const ask = async (prompt: string) => {
    if (!modelUp) throw new Error("offline");
    if (!prompt.startsWith("Classify")) return "Eyes here. Tell me what's happening and where.";
    const msg = JSON.parse(prompt.match(/Message: (".*")/)![1]) as string;
    const report = /fight|collapsed|ambulance|following|help/i.test(msg);
    const place = msg.match(/(?:at|near|outside) (.+?)(?:[.!?]|$)/i)?.[1] ?? null;
    return JSON.stringify({ intent: report ? "report" : msg.endsWith("?") ? "question" : "chat", kind: /collapsed|ambulance/i.test(msg) ? "medical" : /fight/i.test(msg) ? "assault" : report ? "harassment" : null, place, hurt: /collapsed|ambulance|bleeding/i.test(msg) });
  };
  const people = createPeople({ node, log: () => {}, places, stops, send, sendEms: async (...m) => void m.forEach((x) => sent.push({ to: "EMS", text: msgText(x) })), typing: (_h, fn) => fn(), ask, say: () => {}, nearestOf, latestIncident: () => null, latestFramePath: async () => null });
  const intake = createIntake({ node, log: () => {}, places, stops, ask, send, typing: (_h, fn) => fn(), people, nearestOf });
  const answered: string[] = [];
  const answer = async (q: string, id: string) => void answered.push(`${id}:${q}`);
  const last = (h: string) => [...sent].reverse().find((s) => s.to === h)?.text ?? "";
  const incidents = () => node.store.list("incidents");

  const A = "+15550001001", B = "+15550001002", C = "+15550001003", D = "+15550001004";
  await intake.onText(A, "there's a fight at Five Points", answer);
  expect(incidents().length === 1 && incidents()[0].kind === "assault" && /Five Points/.test(String(incidents()[0].label)), "a report with a place raises an incident there");
  expect(/raised an alert/.test(sent.find((s) => s.to === A)?.text ?? "") && /911/.test(sent.find((s) => s.to === A)?.text ?? ""), "reporter gets a confirmation with a 911 reminder");
  expect(sent.some((s) => s.to === A && /\[poll\] Are you safe/.test(s.text)), "and an Are you safe? poll");

  await intake.onText(B, "someone is following me, help", answer);
  expect(/Where is this happening/.test(last(B)), "no place: it asks where");
  await intake.onText(B, "Peachtree Center", answer);
  expect(incidents().some((i) => /Peachtree Center/.test(String(i.label))), "the answer to where raises the incident");

  await intake.onText(C, "big fight at Five Points Station!", answer);
  expect(incidents().filter((i) => /Five Points/.test(String(i.label))).length === 1 && Number(incidents().find((i) => /Five Points/.test(String(i.label)))!.reports) === 2, "a second report nearby joins the live incident");
  expect(/already active/.test(sent.find((s) => s.to === C)?.text ?? ""), "and tells them responders are already on it");

  await people.onText(D, "join Georgia State Station");
  await intake.onText(D, "someone collapsed here, need an ambulance", answer);
  const med = incidents().find((i) => i.kind === "medical");
  expect(!!med && /Georgia State/.test(String(med.label)), '"here" uses their saved place');
  expect(sent.some((s) => s.to === "EMS" && /EMS REQUEST/.test(s.text)), "someone hurt pages EMS");
  expect(/Help is coming/.test(sent.filter((s) => s.to === D).map((s) => s.text).join(" ")), "and starts care with the reporter");

  await intake.onText("+15550001009", "what's happening near five points?", answer);
  expect(answered.length === 1, "a question about a live incident gets a grounded answer");
  await intake.onText("+15550001008", "hey what is this", answer);
  expect(/Eyes/.test(last("+15550001008")), "chat gets a friendly reply");
}

await run(true);
await run(false);
console.log(failures ? `\n${failures} FAILED` : "\nall passed");
process.exit(failures ? 1 : 0);
