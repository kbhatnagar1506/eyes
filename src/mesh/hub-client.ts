// Connects a MeshNode to the hub (server/hub.ts): announces it, learns who
// else is on the mesh, and gives every peer a relay link through the hub.
// Uses the global WebSocket, so the same code runs in the browser and in Bun.

import type { MeshNode, Wire } from "./node";

export interface HubPeer {
  id: string;
  name: string;
  role: string;
  colocated: boolean;
}

const RECONNECT_MS = 2000;

export class HubClient {
  peers: HubPeer[] = [];
  connected = false;
  private ws: WebSocket | null = null;
  private stopped = false;
  private signalListeners = new Set<(from: string, data: unknown) => void>();
  private peerListeners = new Set<(peers: HubPeer[]) => void>();
  private readonly node: MeshNode;
  private readonly url: string;

  constructor(node: MeshNode, url: string) {
    this.node = node;
    this.url = url;
    this.connect();
  }

  private connect() {
    if (this.stopped) return;
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.connected = true;
      ws.send(JSON.stringify({ t: "hello", id: this.node.id, name: this.node.info.name, role: this.node.info.role }));
    };
    ws.onmessage = (ev) => {
      let msg: any;
      try {
        msg = JSON.parse(String(ev.data));
      } catch {
        return;
      }
      if (msg.t === "peers") this.setPeers(msg.peers);
      else if (msg.t === "relay") {
        const peer = this.peers.find((p) => p.id === msg.from);
        this.node.receive(msg.from, peer?.colocated ? "local" : "relay", msg.w as Wire);
      } else if (msg.t === "signal") this.signalListeners.forEach((l) => l(msg.from, msg.data));
    };
    ws.onclose = () => {
      this.connected = false;
      // Hub gone: relay links go with it. Direct P2P links don't.
      this.setPeers([]);
      setTimeout(() => this.connect(), RECONNECT_MS);
    };
    ws.onerror = () => ws.close();
  }

  private setPeers(peers: HubPeer[]) {
    const next = new Map(peers.map((p) => [p.id, p]));
    for (const old of this.peers) {
      const now = next.get(old.id);
      if (!now || now.colocated !== old.colocated) this.node.removeLink(old.id, old.colocated ? "local" : "relay");
    }
    for (const p of peers) {
      const kind = p.colocated ? "local" : "relay";
      this.node.addLink(p.id, kind, (w) => this.send({ t: "relay", to: p.id, w }));
    }
    this.peers = peers;
    this.peerListeners.forEach((l) => l(peers));
  }

  send(msg: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
    else throw new Error("hub not connected");
  }

  signal(to: string, data: unknown) {
    this.send({ t: "signal", to, data });
  }

  onSignal(cb: (from: string, data: unknown) => void) {
    this.signalListeners.add(cb);
    return () => this.signalListeners.delete(cb);
  }

  onPeers(cb: (peers: HubPeer[]) => void) {
    this.peerListeners.add(cb);
    return () => this.peerListeners.delete(cb);
  }

  stop() {
    this.stopped = true;
    this.ws?.close();
  }
}
