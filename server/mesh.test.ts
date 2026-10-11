import { afterAll, describe, expect, test } from "bun:test";
import { MeshNode, type RouteDecision, type Wire } from "../src/mesh/node";
import { radioLink } from "../src/mesh/radio";
import { HubClient } from "../src/mesh/hub-client";

const nodes: MeshNode[] = [];
function node(name: string, lat = 33.75, lng = -84.39) {
  const n = new MeshNode({ name, role: "test", lat, lng }, { id: name, heartbeatMs: 0, pingMs: 200 });
  nodes.push(n);
  return n;
}
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
afterAll(() => nodes.forEach((n) => n.stop()));

describe("gossip", () => {
  test("a write reaches a node with no direct link, hop by hop", async () => {
    // A - B - C - D in a line: A and D share no link.
    const [a, b, c, d] = ["A", "B", "C", "D"].map((x) => node(x));
    radioLink(a, b);
    radioLink(b, c);
    radioLink(c, d);
    let hopsAtD = -1;
    d.store.onChange((coll) => {
      if (coll === "incidents") hopsAtD = 1;
    });
    const arrivals: number[] = [];
    d.onPacket(() => {});
    // Count hops by watching what C sends to D.
    c.onPacket((e) => {
      if (e.to === "D" && e.origin === "A") arrivals.push(e.hops);
    });
    a.write("incidents", "i1", { kind: "assault", status: "detected" });
    await wait(600);
    expect(d.store.get("incidents", "i1")?.kind).toBe("assault");
    expect(hopsAtD).toBe(1);
    expect(Math.max(...arrivals)).toBe(2); // A->B hop 0, B->C hop 1, C->D hop 2
  });

  test("concurrent edits to different fields both survive, on every node", async () => {
    const [a, b, c] = ["E", "F", "G"].map((x) => node(x));
    radioLink(a, b);
    radioLink(b, c);
    a.write("incidents", "i2", { status: "detected" });
    await wait(300);
    // Two devices act at the same moment on different fields.
    a.write("incidents", "i2", { "ack:driver": 1 });
    c.write("incidents", "i2", { "ack:police": 1 });
    await wait(500);
    for (const n of [a, b, c]) {
      const doc = n.store.get("incidents", "i2")!;
      expect(doc["ack:driver"]).toBe(1);
      expect(doc["ack:police"]).toBe(1);
    }
  });

  test("same field written twice: everyone converges on the later write", async () => {
    const [a, b] = ["H", "I"].map((x) => node(x));
    radioLink(a, b);
    a.write("incidents", "i3", { status: "detected" });
    await wait(5);
    b.write("incidents", "i3", { status: "responding" });
    await wait(400);
    expect(a.store.get("incidents", "i3")!.status).toBe("responding");
    expect(b.store.get("incidents", "i3")!.status).toBe("responding");
  });
});

describe("networking agent", () => {
  // Two transports between the same pair, with chosen latencies.
  function dual(a: MeshNode, b: MeshNode, fastMs: number, slowMs: number) {
    const link = (from: MeshNode, to: MeshNode, kind: "radio" | "relay", ms: number) =>
      from.addLink(to.id, kind, (w: Wire) => setTimeout(() => to.receive(from.id, kind, JSON.parse(JSON.stringify(w))), ms));
    link(a, b, "radio", fastMs);
    link(b, a, "radio", fastMs);
    link(a, b, "relay", slowMs);
    link(b, a, "relay", slowMs);
  }

  test("picks the faster transport, fails over when it dies, and explains why", async () => {
    const a = node("J");
    const b = node("K");
    const decisions: RouteDecision[] = [];
    a.onDecision((d) => decisions.push(d));
    dual(a, b, 5, 80);
    await wait(700);
    expect(a.route("K")).toBe("radio");

    // The outage: the medium is gone for both ends.
    a.setDisabled("radio", true);
    b.setDisabled("radio", true);
    expect(a.route("K")).toBe("relay");
    expect(b.route("J")).toBe("relay");
    const last = decisions[decisions.length - 1];
    expect(last.to).toBe("relay");
    expect(last.reason).toContain("down (outage)");

    // Data still gets through on the surviving transport.
    b.write("incidents", "i4", { status: "detected" });
    await wait(300);
    expect(a.store.get("incidents", "i4")?.status).toBe("detected");

    a.setDisabled("radio", false);
    b.setDisabled("radio", false);
    await wait(700);
    expect(a.route("K")).toBe("radio");
  });

  test("a link that silently stops delivering is detected from pings, and nothing written meanwhile is lost", async () => {
    const a = node("L");
    const b = node("M");
    let radioBroken = false;
    const link = (from: MeshNode, to: MeshNode, kind: "radio" | "relay", ms: number) =>
      from.addLink(to.id, kind, (w: Wire) => {
        if (kind === "radio" && radioBroken) return; // bytes vanish, no error
        setTimeout(() => to.receive(from.id, kind, JSON.parse(JSON.stringify(w))), ms);
      });
    link(a, b, "radio", 5);
    link(b, a, "radio", 5);
    link(a, b, "relay", 80);
    link(b, a, "relay", 80);
    await wait(700);
    expect(b.route("L")).toBe("radio");

    radioBroken = true;
    b.write("incidents", "i6", { status: "detected" }); // goes into the dead link
    await wait(2500); // ~3 missed pings at 200 ms, then failover
    expect(b.route("L")).toBe("relay");
    expect(a.store.get("incidents", "i6")?.status).toBe("detected"); // resent on failover
  });
});

describe("hub", () => {
  test("two clients find each other and sync through the hub relay", async () => {
    const port = 18787;
    const proc = Bun.spawn(["bun", "server/hub.ts"], { env: { ...process.env, EYES_HUB_PORT: String(port) }, stdout: "ignore", stderr: "inherit" });
    try {
      await wait(500);
      const a = node("hubA");
      const b = node("hubB");
      const ha = new HubClient(a, `ws://127.0.0.1:${port}/mesh`);
      const hb = new HubClient(b, `ws://127.0.0.1:${port}/mesh`);
      await wait(800);
      // Both on this machine, so the hub marks them co-located.
      expect(a.route("hubB")).toBe("local");
      a.write("incidents", "i5", { status: "detected" });
      await wait(400);
      expect(b.store.get("incidents", "i5")?.status).toBe("detected");
      ha.stop();
      hb.stop();
    } finally {
      proc.kill();
    }
  });
});

describe("big data", () => {
  test("a late joiner catches up on state far bigger than one message, in pieces", async () => {
    const a = node("N");
    const b = node("O");
    const frame = "x".repeat(40_000); // about one evidence frame
    for (let i = 0; i < 5; i++) a.write("frames", `f${i}`, { data: frame });
    const sizes: number[] = [];
    a.addLink("O", "radio", (w: Wire) => {
      sizes.push(JSON.stringify(w).length);
      setTimeout(() => b.receive("N", "radio", JSON.parse(JSON.stringify(w))), 5);
    });
    b.addLink("N", "radio", (w: Wire) => setTimeout(() => a.receive("O", "radio", JSON.parse(JSON.stringify(w))), 5));
    await wait(300);
    for (let i = 0; i < 5; i++) expect(b.store.get("frames", `f${i}`)?.data).toBe(frame);
    expect(Math.max(...sizes)).toBeLessThan(64_000);
  });
});
