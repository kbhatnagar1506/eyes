// Joins a hub as a test camera and raises one incident (for rehearsals).
// Run: EYES_HUB=ws://127.0.0.1:8799/mesh bun scripts/test/raise.ts [kind] [injured|-] [frame.jpg]
import { MeshNode } from "../../src/mesh/node";
import { HubClient } from "../../src/mesh/hub-client";
const node = new MeshNode({ name: process.env.EYES_CAMERA ?? "Rehearsal camera", role: "camera", lat: 33.7527, lng: -84.3857 }, { id: `cam-test-${process.pid}` });
new HubClient(node, process.env.EYES_HUB ?? "ws://127.0.0.1:8787/mesh");
await Bun.sleep(1500);
const id = `inc-${Date.now().toString(36)}`;
node.write("incidents", id, { kind: process.argv[2] ?? "assault", label: "GSU Student Center East", detail: process.env.EYES_DETAIL ?? "Rehearsal: attack detected", lat: 33.7527, lng: -84.3857, at: Date.now(), status: "detected", camera: node.id });
console.log(`raised ${id}`);
// An evidence frame, as the camera sends for each detection.
const framePath = process.argv[4];
if (framePath) {
  await Bun.sleep(800);
  const data = `data:image/jpeg;base64,${Buffer.from(await Bun.file(framePath).arrayBuffer()).toString("base64")}`;
  node.write("frames", `${id}#001`, { incident: id, seq: 1, data, caption: process.env.EYES_DETAIL ?? "Rehearsal: attack detected", at: Date.now() });
  node.write("incidents", id, { punches: 1 });
  console.log(`frame sent (${data.length} chars)`);
}
if (process.argv[3] === "full") {
  // Plays out a whole incident: officer responds, finds someone hurt, resolves.
  await Bun.sleep(25000);
  node.write("incidents", id, { "ack:police": Date.now(), "status:police": "on the way" });
  console.log("officer responding");
  await Bun.sleep(20000);
  node.write("incidents", id, { injuredReportedAt: Date.now() });
  console.log("officer reported an injured person");
  await Bun.sleep(30000);
  node.write("incidents", id, { status: "resolved", resolvedAt: Date.now(), "status:police": "resolved" });
  console.log("resolved");
} else if (process.argv[3] === "injured") {
  await Bun.sleep(20000);
  node.write("incidents", id, { "ack:police": Date.now(), "status:police": "responding", injuredReportedAt: Date.now() });
  console.log("officer reported an injured person");
}
await Bun.sleep(4000);
process.exit(0);
