// One device on the Eyes mesh. Runs the same in the browser (camera, phones,
// simulated city nodes) and in Bun (the agent gateway).
//
// - Links: a neighbor can be reachable over several transports at once
//   (direct P2P Wi-Fi, the cloud relay, mesh radio, same-device). Transports
//   plug in with addLink/removeLink/receive; this class doesn't care how bytes
//   move.
// - Networking agent: pings every link, keeps latency/loss per transport, and
//   sends each message over the best one. Every switch is logged with a reason.
// - Sync: writes become field-level ops (see store.ts) and gossip outward. A
//   node forwards only the ops that were new to it, so each op crosses each
//   link about once and data reaches devices with no direct link, hop by hop.

import { HLC } from "./hlc";
import { Store, type Op } from "./store";

export type TransportKind = "local" | "p2p" | "radio" | "relay";

export const TRANSPORT_LABEL: Record<TransportKind, string> = {
  local: "Same device",
  p2p: "P2P Wi-Fi (WebRTC)",
  radio: "Mesh radio (sim)",
  relay: "Cloud relay (internet)",
};

// Fixed cost added to measured latency: at equal speed, prefer links that
// don't depend on infrastructure. The relay needs the internet and a server;
// a direct link keeps working when both are gone.
const TRANSPORT_COST_MS: Record<TransportKind, number> = { local: 0, p2p: 5, radio: 10, relay: 40 };

const PING_MS = 1000;
// A ping unanswered this long counts as lost.
const pingTimeoutMs = (pingMs: number) => Math.max(1000, 2 * pingMs);
// No pong for this long while pings are outstanding = the link is down
// (about three missed pings).
const deadAfterMs = (pingMs: number) => Math.max(1500, 3.5 * pingMs);
const LOSS_ALPHA = 0.3;
const RTT_ALPHA = 0.3;
const SEEN_LIMIT = 5000;
const DECISION_HISTORY = 20;

export type Wire =
  | { k: "ops"; mid: string; origin: string; hops: number; ops: Op[] }
  | { k: "ping"; n: number }
  | { k: "pong"; n: number };

export interface LinkStats {
  kind: TransportKind;
  rttMs: number | null;
  loss: number; // 0..1, smoothed
  alive: boolean;
}

export interface PacketEvent {
  from: string;
  to: string;
  transport: TransportKind;
  hops: number;
  ops: number;
  origin: string;
  colls: string[]; // which collections the ops touch (incident traffic vs presence chatter)
}

// Fresh data arriving at this node: which neighbor and transport it came over,
// and how many hops it took from the device that wrote it.
export interface DeliveryEvent {
  from: string;
  transport: TransportKind;
  hops: number;
  origin: string;
  colls: string[];
}

export interface RouteDecision {
  at: number;
  node: string;
  peer: string;
  from: TransportKind | null;
  to: TransportKind | null;
  reason: string;
}

interface Link {
  kind: TransportKind;
  send: (w: Wire) => void;
  rttMs: number | null;
  loss: number;
  addedAt: number;
  lastPong: number;
  pending: Map<number, number>; // ping n -> sent at
  exclude?: ReadonlySet<string>; // collections this neighbor doesn't carry
}

export interface NodeInfo {
  name: string;
  role: string;
  lat: number;
  lng: number;
  virtual?: boolean;
}

let midCounter = 0;

const collsOf = (ops: Op[]) => [...new Set(ops.map((o) => o.c))];

const MAX_MESSAGE_BYTES = 48_000;
const opSize = (op: Op) => JSON.stringify(op).length;
const approxSize = (ops: Op[]) => ops.reduce((s, o) => s + opSize(o), 0);

// Groups ops into messages under the size cap. A single op over the cap
// (shouldn't happen: frames are encoded to fit) still goes, alone.
function chunkOps(ops: Op[]): Op[][] {
  const chunks: Op[][] = [];
  let cur: Op[] = [];
  let size = 0;
  for (const op of ops) {
    const s = opSize(op);
    if (cur.length && size + s > MAX_MESSAGE_BYTES) {
      chunks.push(cur);
      cur = [];
      size = 0;
    }
    cur.push(op);
    size += s;
  }
  if (cur.length) chunks.push(cur);
  return chunks;
}

export class MeshNode {
  readonly id: string;
  readonly hlc: HLC;
  readonly store = new Store();
  readonly info: NodeInfo;
  // Transports switched off on purpose (stage demo: "kill the internet").
  readonly disabled = new Set<TransportKind>();
  // Newest first, so a view opened later can show what already happened.
  readonly recentDecisions: RouteDecision[] = [];

  private links = new Map<string, Map<TransportKind, Link>>();
  private chosen = new Map<string, TransportKind | null>();
  private seen: string[] = [];
  private seenSet = new Set<string>();
  private pingN = 0;
  private readonly pingMs: number;
  private timers: ReturnType<typeof setInterval>[] = [];
  private packetListeners = new Set<(e: PacketEvent) => void>();
  private decisionListeners = new Set<(d: RouteDecision) => void>();
  private linkListeners = new Set<() => void>();
  private deliveryListeners = new Set<(e: DeliveryEvent) => void>();

  // heartbeatMs 0 = publish presence once. presence false = never publish it
  // (simulated city nodes: the map already knows where they are, and hundreds
  // of presence docs would drown the real devices' traffic).
  constructor(info: NodeInfo, opts: { id?: string; heartbeatMs?: number; pingMs?: number; presence?: boolean } = {}) {
    this.id = opts.id ?? `${info.role}-${Math.random().toString(36).slice(2, 8)}`;
    this.info = info;
    this.hlc = new HLC(this.id);
    this.pingMs = opts.pingMs ?? PING_MS;
    this.timers.push(setInterval(() => this.tick(), this.pingMs));
    if (opts.presence === false) return;
    const hb = opts.heartbeatMs ?? 3000;
    if (hb > 0) this.timers.push(setInterval(() => this.publishPresence(), hb));
    this.publishPresence();
  }

  stop() {
    this.timers.forEach(clearInterval);
    this.timers = [];
  }

  // ---- data ---------------------------------------------------------------

  write(c: string, id: string, fields: Record<string, unknown>) {
    const ops: Op[] = Object.entries(fields).map(([f, v]) => ({ c, id, f, v, t: this.hlc.now() }));
    for (const op of ops) this.store.apply(op);
    this.gossip({ k: "ops", mid: this.newMid(), origin: this.id, hops: 0, ops }, null);
  }

  private publishPresence() {
    const links: Record<string, TransportKind> = {};
    for (const [peer] of this.links) {
      const k = this.chosen.get(peer);
      if (k) links[peer] = k;
    }
    this.write("presence", this.id, { ...this.info, links, seen: Date.now() });
  }

  // ---- transports plug in here ------------------------------------------

  // `exclude`: collections never sent over this link (simulated city nodes
  // only need alerts, not 40 KB evidence frames on every punch).
  addLink(peer: string, kind: TransportKind, send: (w: Wire) => void, opts: { exclude?: ReadonlySet<string> } = {}) {
    let byKind = this.links.get(peer);
    if (!byKind) this.links.set(peer, (byKind = new Map()));
    if (byKind.has(kind)) return;
    const now = Date.now();
    byKind.set(kind, { kind, send, rttMs: null, loss: 0, addedAt: now, lastPong: now, pending: new Map(), exclude: opts.exclude });
    // First contact over this transport: hand over everything we know. Cheap
    // (the state is small) and it's how a device that was offline catches up.
    this.sendVia(peer, kind, { k: "ops", mid: this.newMid(), origin: this.id, hops: 0, ops: this.store.snapshot() });
    this.ping(byKind.get(kind)!);
    this.evaluate(peer);
    this.linkListeners.forEach((l) => l());
  }

  removeLink(peer: string, kind: TransportKind) {
    const byKind = this.links.get(peer);
    if (!byKind?.delete(kind)) return;
    if (byKind.size === 0) this.links.delete(peer);
    this.evaluate(peer);
    this.linkListeners.forEach((l) => l());
  }

  receive(peer: string, kind: TransportKind, w: Wire) {
    const link = this.links.get(peer)?.get(kind);
    if (link && this.disabled.has(kind)) return; // outage: the bytes never arrive
    if (w.k === "ping") {
      if (link) link.send({ k: "pong", n: w.n });
      return;
    }
    if (w.k === "pong") {
      if (!link) return;
      const sentAt = link.pending.get(w.n);
      if (sentAt === undefined) return;
      link.pending.delete(w.n);
      const rtt = Date.now() - sentAt;
      link.rttMs = link.rttMs === null ? rtt : link.rttMs * (1 - RTT_ALPHA) + rtt * RTT_ALPHA;
      link.loss *= 1 - LOSS_ALPHA;
      link.lastPong = Date.now();
      this.evaluate(peer);
      return;
    }
    if (this.seenSet.has(w.mid)) return;
    this.remember(w.mid);
    const fresh: Op[] = [];
    for (const op of w.ops) {
      this.hlc.observe(op.t);
      if (this.store.apply(op)) fresh.push(op);
    }
    if (!fresh.length) return;
    const e: DeliveryEvent = { from: peer, transport: kind, hops: w.hops + 1, origin: w.origin, colls: collsOf(fresh) };
    this.deliveryListeners.forEach((l) => l(e));
    this.gossip({ ...w, hops: w.hops + 1, ops: fresh }, peer);
  }

  // ---- networking agent ---------------------------------------------------

  private score(link: Link): number {
    if (this.disabled.has(link.kind) || !this.alive(link)) return Infinity;
    return (link.rttMs ?? 50) * (1 + 4 * link.loss) + TRANSPORT_COST_MS[link.kind];
  }

  private alive(link: Link): boolean {
    return link.pending.size === 0 || Date.now() - link.lastPong < deadAfterMs(this.pingMs);
  }

  private best(peer: string): Link | null {
    let best: Link | null = null;
    let bestScore = Infinity;
    for (const link of this.links.get(peer)?.values() ?? []) {
      const s = this.score(link);
      if (s < bestScore) {
        best = link;
        bestScore = s;
      }
    }
    return best;
  }

  private evaluate(peer: string) {
    const prev = this.chosen.get(peer) ?? null;
    const next = this.best(peer)?.kind ?? null;
    if (prev === next) return;
    this.chosen.set(peer, next);
    if (next === null) this.chosen.delete(peer);
    // Failover: whatever went into the dead link may never have arrived, so
    // hand the neighbor the full state over the new route.
    if (prev !== null && next !== null) {
      this.sendVia(peer, next, { k: "ops", mid: this.newMid(), origin: this.id, hops: 0, ops: this.store.snapshot() });
    }
    const d: RouteDecision = { at: Date.now(), node: this.id, peer, from: prev, to: next, reason: this.explain(peer, prev, next) };
    this.recentDecisions.unshift(d);
    this.recentDecisions.length = Math.min(this.recentDecisions.length, DECISION_HISTORY);
    this.decisionListeners.forEach((l) => l(d));
    this.linkListeners.forEach((l) => l());
  }

  private explain(peer: string, prev: TransportKind | null, next: TransportKind | null): string {
    const links = [...(this.links.get(peer)?.values() ?? [])];
    const fmt = (l: Link) =>
      this.disabled.has(l.kind)
        ? `${TRANSPORT_LABEL[l.kind]} down (outage)`
        : !this.alive(l)
          ? `${TRANSPORT_LABEL[l.kind]} not answering`
          : `${TRANSPORT_LABEL[l.kind]} ${l.rttMs === null ? "(new link, not measured yet)" : `${Math.round(l.rttMs)} ms`}${l.loss > 0.05 ? `, ${Math.round(l.loss * 100)}% loss` : ""}`;
    if (next === null) return links.length ? `no working direct link (${links.map(fmt).join("; ")}) — reachable only through other devices` : "link closed";
    const chosen = links.find((l) => l.kind === next)!;
    const others = links.filter((l) => l.kind !== next);
    if (prev === null) return `${fmt(chosen)}${others.length ? ` — best of ${links.length} (${others.map(fmt).join("; ")})` : ""}`;
    return `${fmt(chosen)} beats ${others.map(fmt).join("; ") || TRANSPORT_LABEL[prev]}`;
  }

  linkStats(peer: string): LinkStats[] {
    return [...(this.links.get(peer)?.values() ?? [])].map((l) => ({
      kind: l.kind,
      rttMs: l.rttMs,
      loss: l.loss,
      alive: !this.disabled.has(l.kind) && this.alive(l),
    }));
  }

  route(peer: string): TransportKind | null {
    return this.chosen.get(peer) ?? null;
  }

  neighbors(): string[] {
    return [...this.links.keys()];
  }

  setDisabled(kind: TransportKind, off: boolean) {
    if (off) this.disabled.add(kind);
    else this.disabled.delete(kind);
    for (const peer of this.links.keys()) this.evaluate(peer);
  }

  private tick() {
    const now = Date.now();
    for (const [peer, byKind] of this.links) {
      for (const link of byKind.values()) {
        for (const [n, sentAt] of link.pending) {
          if (now - sentAt > pingTimeoutMs(this.pingMs)) {
            link.pending.delete(n);
            link.loss = link.loss * (1 - LOSS_ALPHA) + LOSS_ALPHA;
          }
        }
        this.ping(link);
      }
      this.evaluate(peer);
    }
  }

  private ping(link: Link) {
    if (this.disabled.has(link.kind)) return;
    const n = ++this.pingN;
    link.pending.set(n, Date.now());
    try {
      link.send({ k: "ping", n });
    } catch {
      /* a dead transport shows up as loss */
    }
  }

  // ---- gossip -------------------------------------------------------------

  private gossip(msg: Extract<Wire, { k: "ops" }>, except: string | null) {
    this.remember(msg.mid);
    for (const peer of this.links.keys()) {
      if (peer === except || peer === msg.origin) continue;
      const link = this.best(peer);
      if (link) this.sendVia(peer, link.kind, msg);
    }
  }

  private sendVia(peer: string, kind: TransportKind, w: Wire) {
    const link = this.links.get(peer)?.get(kind);
    if (!link || this.disabled.has(kind)) return;
    if (w.k === "ops" && link.exclude) {
      const ops = w.ops.filter((o) => !link.exclude!.has(o.c));
      if (!ops.length) return;
      w = { ...w, ops };
    }
    // Data channels refuse big messages (Safari's limit is 64 KB), and a full
    // snapshot with evidence frames can be much bigger: split it.
    if (w.k === "ops" && w.ops.length > 1 && approxSize(w.ops) > MAX_MESSAGE_BYTES) {
      const base = w;
      chunkOps(base.ops).forEach((ops, i) => this.sendVia(peer, kind, { ...base, mid: `${base.mid}.${i}`, ops }));
      return;
    }
    try {
      link.send(w);
    } catch {
      // The transport refused outright (socket closed): don't wait for pings
      // to notice. Mark it dead and reroute now.
      link.loss = 1;
      link.lastPong = 0;
      link.pending.set(-1, 0);
      queueMicrotask(() => this.evaluate(peer));
      return;
    }
    if (w.k === "ops" && w.ops.length) {
      const e: PacketEvent = { from: this.id, to: peer, transport: kind, hops: w.hops, ops: w.ops.length, origin: w.origin, colls: collsOf(w.ops) };
      this.packetListeners.forEach((l) => l(e));
    }
  }

  private newMid() {
    return `${this.id}:${Date.now().toString(36)}:${(midCounter++).toString(36)}`;
  }

  private remember(mid: string) {
    if (this.seenSet.has(mid)) return;
    this.seenSet.add(mid);
    this.seen.push(mid);
    if (this.seen.length > SEEN_LIMIT) this.seenSet.delete(this.seen.shift()!);
  }

  // ---- events ---------------------------------------------------------------

  onPacket(cb: (e: PacketEvent) => void) {
    this.packetListeners.add(cb);
    return () => this.packetListeners.delete(cb);
  }

  onDeliver(cb: (e: DeliveryEvent) => void) {
    this.deliveryListeners.add(cb);
    return () => this.deliveryListeners.delete(cb);
  }

  onDecision(cb: (d: RouteDecision) => void) {
    this.decisionListeners.add(cb);
    return () => this.decisionListeners.delete(cb);
  }

  onLinksChanged(cb: () => void) {
    this.linkListeners.add(cb);
    return () => this.linkListeners.delete(cb);
  }
}
