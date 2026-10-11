// Eyes mesh hub: presence, WebRTC signaling, and relay. It holds no app data.
// Devices find each other here, use it to set up direct P2P links, and relay
// through it only when a direct link can't form. Kill this process and every
// direct P2P link keeps syncing.
//
// Also serves MARTA's live bus positions (GTFS-realtime) as JSON, since the
// feed has no CORS headers for the browser to read it directly.

import GtfsRealtimeBindings from "gtfs-realtime-bindings";
import type { ServerWebSocket } from "bun";

const PORT = Number(process.env.EYES_HUB_PORT ?? 8787);
const VEHICLES_URL = "https://gtfs-rt.itsmarta.com/TMGTFSRealTimeWebService/vehicle/vehiclepositions.pb";
const BUS_CACHE_MS = 15_000;

type ClientData = { addr: string; id?: string; name?: string; role?: string };
type Sock = ServerWebSocket<ClientData>;

const clients = new Map<string, Sock>();

const isLoopback = (addr: string) => addr === "127.0.0.1" || addr === "::1" || addr === "::ffff:127.0.0.1";

const bare = (a: string) => a.replace(/^::ffff:/, "");
// A private address shared by two devices is one machine. A shared public
// address isn't: through the tunnel, every phone on the same hotspot arrives
// from the same public IP.
const isPrivate = (a: string) => /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|fe80:|fc|fd)/i.test(bare(a));
const sameHost = (a: string, b: string) => (isLoopback(a) && isLoopback(b)) || (!!a && isPrivate(a) && bare(a) === bare(b));

// Devices reach the hub through the Vite dev server's proxy, which arrives
// from loopback. Trust its X-Forwarded-For only then: that's the real device.
function clientAddr(req: Request, socketAddr: string): string {
  const fwd = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return isLoopback(socketAddr) && fwd ? fwd : socketAddr;
}

// Each device gets its own peer list: peers on the same machine are marked
// so the link between them is "same device", not internet.
function broadcastPeers() {
  for (const [id, ws] of clients) {
    const peers = [...clients.values()]
      .filter((o) => o.data.id !== id)
      .map((o) => ({
        id: o.data.id,
        name: o.data.name,
        role: o.data.role,
        colocated: sameHost(o.data.addr, ws.data.addr),
      }));
    ws.send(JSON.stringify({ t: "peers", peers }));
  }
}

let busCache: { at: number; body: string } | null = null;

async function buses(): Promise<Response> {
  const headers = { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" };
  if (busCache && Date.now() - busCache.at < BUS_CACHE_MS) return new Response(busCache.body, { headers });
  try {
    const resp = await fetch(VEHICLES_URL);
    if (!resp.ok) throw new Error(`MARTA feed ${resp.status}`);
    const feed = GtfsRealtimeBindings.transit_realtime.FeedMessage.decode(new Uint8Array(await resp.arrayBuffer()));
    const vehicles = feed.entity
      .filter((e) => e.vehicle?.position)
      .map((e) => ({
        id: e.vehicle!.vehicle?.label || e.vehicle!.vehicle?.id || e.id,
        route: e.vehicle!.trip?.routeId ?? null,
        lat: e.vehicle!.position!.latitude,
        lng: e.vehicle!.position!.longitude,
        bearing: e.vehicle!.position!.bearing ?? null,
        ts: Number(e.vehicle!.timestamp ?? 0) * 1000,
      }));
    busCache = { at: Date.now(), body: JSON.stringify({ source: "MARTA GTFS-realtime", fetchedAt: Date.now(), vehicles }) };
    return new Response(busCache.body, { headers });
  } catch (err) {
    if (busCache) return new Response(busCache.body, { headers }); // stale beats nothing
    return new Response(JSON.stringify({ error: String(err), vehicles: [] }), { status: 502, headers });
  }
}

Bun.serve<ClientData>({
  port: PORT,
  hostname: "0.0.0.0", // phones on the same Wi-Fi connect straight to this laptop
  fetch(req, server) {
    const url = new URL(req.url);
    if (url.pathname === "/mesh") {
      if (server.upgrade(req, { data: { addr: clientAddr(req, server.requestIP(req)?.address ?? "") } })) return;
      return new Response("websocket upgrade failed", { status: 400 });
    }
    if (url.pathname === "/buses") return buses();
    return new Response("eyes mesh hub", { headers: { "Access-Control-Allow-Origin": "*" } });
  },
  websocket: {
    message(ws: Sock, raw) {
      let msg: any;
      try {
        msg = JSON.parse(String(raw));
      } catch {
        return;
      }
      if (msg.t === "hello") {
        if (typeof msg.id !== "string" || !msg.id) return;
        clients.get(msg.id)?.close(); // same device reconnecting
        ws.data.id = msg.id;
        ws.data.name = msg.name;
        ws.data.role = msg.role;
        clients.set(msg.id, ws);
        broadcastPeers();
        return;
      }
      const from = ws.data.id;
      if (!from) return;
      const target = clients.get(msg.to);
      if (!target) return;
      if (msg.t === "signal") target.send(JSON.stringify({ t: "signal", from, data: msg.data }));
      else if (msg.t === "relay") target.send(JSON.stringify({ t: "relay", from, w: msg.w }));
    },
    close(ws: Sock) {
      if (ws.data.id && clients.get(ws.data.id) === ws) {
        clients.delete(ws.data.id);
        broadcastPeers();
      }
    },
  },
});

console.log(`eyes hub on :${PORT} (ws /mesh, GET /buses)`);
