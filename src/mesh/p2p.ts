// Direct device-to-device links over WebRTC data channels. The hub is only
// used to exchange the connection setup; once a channel is open, data flows
// straight between the two devices and survives the hub going away.
//
// No STUN/TURN servers: links form over the local network only, which is the
// point (it has to work with no internet). Where the network blocks
// device-to-device traffic (common on venue Wi-Fi), the channel never opens
// and the networking agent keeps using the relay.

import type { MeshNode, Wire } from "./node";
import type { HubClient, HubPeer } from "./hub-client";

const CONNECT_TIMEOUT_MS = 8000;
const RETRY_MS = 20_000;

type Signal = { kind: "offer" | "answer"; sdp: string } | { kind: "ice"; candidate: RTCIceCandidateInit };

interface Conn {
  pc: RTCPeerConnection;
  dc: RTCDataChannel | null;
  startedAt: number;
}

export class P2PManager {
  private conns = new Map<string, Conn>();
  private lastAttempt = new Map<string, number>();
  private readonly node: MeshNode;
  private readonly hub: HubClient;
  private readonly allow: (peer: HubPeer) => boolean;

  // `allow` decides whether a pair should even try a direct link (used to
  // simulate radio range on the stage map).
  constructor(node: MeshNode, hub: HubClient, allow: (peer: HubPeer) => boolean = () => true) {
    this.node = node;
    this.hub = hub;
    this.allow = allow;
    hub.onPeers((peers) => this.sync(peers));
    hub.onSignal((from, data) => void this.onSignal(from, data as Signal));
    setInterval(() => this.sync(this.hub.peers), 5000);
  }

  private sync(peers: HubPeer[]) {
    const now = Date.now();
    for (const p of peers) {
      if (p.colocated) continue; // same machine already has a better path
      const c = this.conns.get(p.id);
      if (c && c.dc?.readyState !== "open" && now - c.startedAt > CONNECT_TIMEOUT_MS) this.close(p.id);
      if (!this.allow(p)) {
        if (this.conns.has(p.id)) this.close(p.id);
        continue;
      }
      // One side starts the connection so the two don't both offer at once.
      if (this.node.id < p.id && !this.conns.has(p.id) && now - (this.lastAttempt.get(p.id) ?? 0) > RETRY_MS) {
        this.lastAttempt.set(p.id, now);
        void this.start(p.id);
      }
    }
  }

  private newConn(peer: string): Conn {
    const pc = new RTCPeerConnection({ iceServers: [] });
    const conn: Conn = { pc, dc: null, startedAt: Date.now() };
    this.conns.set(peer, conn);
    pc.onicecandidate = (e) => {
      if (e.candidate) this.trySignal(peer, { kind: "ice", candidate: e.candidate.toJSON() });
    };
    pc.ondatachannel = (e) => this.attach(peer, conn, e.channel);
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "failed" || pc.connectionState === "closed") this.close(peer);
    };
    return conn;
  }

  private async start(peer: string) {
    const conn = this.newConn(peer);
    this.attach(peer, conn, conn.pc.createDataChannel("eyes", { ordered: true }));
    const offer = await conn.pc.createOffer();
    await conn.pc.setLocalDescription(offer);
    this.trySignal(peer, { kind: "offer", sdp: offer.sdp! });
  }

  private async onSignal(from: string, s: Signal) {
    try {
      if (s.kind === "offer") {
        this.close(from);
        const conn = this.newConn(from);
        await conn.pc.setRemoteDescription({ type: "offer", sdp: s.sdp });
        const answer = await conn.pc.createAnswer();
        await conn.pc.setLocalDescription(answer);
        this.trySignal(from, { kind: "answer", sdp: answer.sdp! });
      } else if (s.kind === "answer") {
        await this.conns.get(from)?.pc.setRemoteDescription({ type: "answer", sdp: s.sdp });
      } else if (s.kind === "ice") {
        await this.conns.get(from)?.pc.addIceCandidate(s.candidate);
      }
    } catch {
      this.close(from);
    }
  }

  private attach(peer: string, conn: Conn, dc: RTCDataChannel) {
    conn.dc = dc;
    dc.onopen = () => this.node.addLink(peer, "p2p", (w: Wire) => dc.send(JSON.stringify(w)));
    dc.onmessage = (e) => {
      try {
        this.node.receive(peer, "p2p", JSON.parse(String(e.data)) as Wire);
      } catch {
        /* malformed frame */
      }
    };
    dc.onclose = () => this.close(peer);
  }

  private close(peer: string) {
    const c = this.conns.get(peer);
    if (!c) return;
    this.conns.delete(peer);
    this.node.removeLink(peer, "p2p");
    try {
      c.dc?.close();
      c.pc.close();
    } catch {
      /* already closed */
    }
  }

  private trySignal(peer: string, s: Signal) {
    try {
      this.hub.signal(peer, s);
    } catch {
      /* hub down: this attempt dies, a later sync retries */
    }
  }

  openLinks(): string[] {
    return [...this.conns].filter(([, c]) => c.dc?.readyState === "open").map(([id]) => id);
  }
}
