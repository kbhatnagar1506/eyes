// Stage view of the Eyes mesh over Atlanta.
//
// What's real: MARTA routes/stops (GTFS), live bus positions (GTFS-realtime),
// public places and responders (OpenStreetMap), every device in the room and
// the links between them, and the protocol every node runs (src/mesh/).
// What's simulated: the city nodes' radios. Those buses, precincts and
// stations don't have our software, so they run node.ts in this page over
// in-memory radio links with realistic per-hop delay. Infrastructure nodes
// (police, fire, hospitals, MARTA stations) also get a simulated internet
// uplink, the way real buildings have wired internet; buses and other places
// are radio only.
//
// The pin problem simulator stages a problem at any typed place: a CCTV node
// appears at the pin, detects it, and the alert spreads from there.

import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MeshNode, TRANSPORT_LABEL, type PacketEvent, type RouteDecision, type TransportKind, type Wire } from "./mesh/node";
import { buildRadioMesh, metersBetween, radioLink } from "./mesh/radio";
import {
  loadLiveBuses,
  loadMarta,
  loadPlaces,
  dataProblems,
  nearest,
  problemFromText,
  problemLabel,
  PROBLEMS,
  scene,
  type ProblemId,
  type LiveBus,
  type Place,
  type PlaceCategory,
  type Role,
} from "./mesh/scenario";
import { busesUrl, hubUrl, OUTAGE_DOC, setOutage, SWITCHABLE, type Session } from "./mesh/session";
import QRCode from "qrcode";
import { geocode, type GeoResult } from "./mesh/geocode";
import { framesFor, safeJpeg } from "./mesh/frames";

const CITY_RADIUS_M = 4500;
const RADIO_RANGE_M = 1100;
const CAMERA_RADIO_NEIGHBORS = 3;
const BUS_REFRESH_MS = 15_000;
const DEVICE_STALE_MS = 12_000;
const PACKET_MS = 420;
// Markers are see-through so routes and streets read underneath them.
const NODE_FILL_IDLE = 0.2;
const PERSON_COLOR: Record<string, string> = { notified: "#f59e0b", safe: "#22c55e", help: "#ef4444" };
const PERSON_GLYPH: Record<string, string> = { notified: "?", safe: "✓", help: "!" };
const PERSON_LABEL: Record<string, string> = { notified: "Alerted by iMessage, no answer yet", safe: "Said they're safe", help: "Needs help: EMS paged" };
const NODE_FILL_REACHED = 0.5;
// A pin farther than this from the current city mesh gets a new mesh built
// around it (the mesh only covers CITY_RADIUS_M around its center).
const REBUILD_FROM_M = 2500;
const BACKHAUL_KINDS = new Set<CityKind>(["police", "fire", "hospital", "station"]);
// Simulated city nodes only need the alert. Evidence frames (40 KB per punch)
// stay on real devices, or the whole city would re-copy every image.
const CITY_EXCLUDE: ReadonlySet<string> = new Set(["frames"]);
// Simulated wired-internet round trip for infrastructure uplinks.
const backhaulDelayMs = () => 35 + Math.random() * 45;

type CityKind = PlaceCategory | "station" | "bus" | "cctv";

const KIND_COLOR: Record<CityKind, string> = {
  cctv: "#f8fafc",
  police: "#3b82f6",
  fire: "#f97316",
  hospital: "#f43f5e",
  station: "#e2e8f0",
  bus: "#2dd4bf",
  school: "#facc15",
  university: "#a78bfa",
  library: "#22d3ee",
  stadium: "#eab308",
  mall: "#ec4899",
  venue: "#94a3b8",
  park: "#22c55e",
};

const KIND_LABEL: Record<CityKind, string> = {
  cctv: "CCTV camera",
  police: "Police",
  fire: "Fire station",
  hospital: "Hospital",
  station: "MARTA station",
  bus: "MARTA bus",
  school: "School",
  university: "University",
  library: "Library",
  stadium: "Stadium",
  mall: "Mall",
  venue: "Venue",
  park: "Park",
};

// How many of each kind join the simulated city mesh (nearest to the scene first).
const CITY_CAPS: Partial<Record<CityKind, number>> = {
  police: 99,
  fire: 99,
  hospital: 99,
  station: 99,
  bus: 60,
  university: 12,
  library: 12,
  school: 20,
  stadium: 6,
  mall: 6,
  venue: 12,
  park: 10,
};

const ROLE_ICON: Record<Role, string> = { camera: "📹", driver: "🧑‍✈️", rider: "🧍", police: "🚓", gateway: "🤖" };

const TRANSPORT_STYLE: Record<TransportKind, L.PolylineOptions> = {
  p2p: { color: "#22d3ee", weight: 3, opacity: 0.9 },
  relay: { color: "#a78bfa", weight: 2, opacity: 0.8, dashArray: "6 8" },
  local: { color: "#64748b", weight: 2, opacity: 0.7, dashArray: "2 6" },
  radio: { color: "#38bdf8", weight: 1.5, opacity: 0.6 },
};

interface CityNode {
  node: MeshNode;
  kind: CityKind;
  name: string;
  marker: L.CircleMarker;
  reachedAt: number | null;
  hops: number | null;
}

interface Packet {
  from: L.LatLng;
  to: L.LatLng;
  start: number;
  incident: boolean;
}

interface Ripple {
  at: L.LatLng;
  start: number;
  color: string;
  maxR: number;
  ms: number;
}

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

export class MeshView {
  private map!: L.Map;
  private canvas!: HTMLCanvasElement;
  private session: Session;
  private root: HTMLElement;
  private city: CityNode[] = [];
  private cityById = new Map<string, CityNode>();
  private packets: Packet[] = [];
  private ripples: Ripple[] = [];
  private deviceMarkers = new Map<string, L.Marker>();
  private deviceLinks = new Map<string, L.Polyline>();
  private liveBusLayer = L.layerGroup();
  private liveBusMarkers = new Map<string, L.CircleMarker>();
  private decisions: RouteDecision[] = [];
  private currentIncident: string | null = null;
  private stats = { places: 0, stops: 0, routes: 0, liveBuses: 0, radioLinks: 0, uplinks: 0 };
  private cityLayer = L.layerGroup();
  // People who signed up for alerts by iMessage (approximate spots, no numbers).
  private peopleLayer = L.layerGroup();
  private meshCenter: { lat: number; lng: number } = scene();
  private cityData: { places: Place[]; stops: [number, number, string, number][]; buses: LiveBus[] } = { places: [], stops: [], buses: [] };
  private pinCam: CityNode | null = null;
  private cityGeneration = 0;
  private readonly openedAt = Date.now();
  private started = false;
  private raf = 0;

  constructor(root: HTMLElement, session: Session) {
    this.root = root;
    this.session = session;
  }

  // Called when the tab becomes visible: Leaflet needs a sized container.
  async show() {
    if (!this.started) {
      this.started = true;
      await this.init();
    }
    this.map.invalidateSize();
    this.loop();
  }

  hide() {
    cancelAnimationFrame(this.raf);
  }

  private async init() {
    this.root.innerHTML = `
      <div class="mv-map" id="mvMap"></div>
      <canvas class="mv-overlay" id="mvOverlay"></canvas>
      <div class="mv-brand">
        <div class="mv-title"><span class="eye">👁</span> EYES <span class="mv-sub">Atlanta safety mesh</span></div>
        <div class="mv-counters" id="mvCounters"></div>
      </div>
      <div class="mv-panel mv-left">
        <div class="mv-panel-title">🛰 Networking agent</div>
        <div class="mv-note">Measures every link and sends each message over the best one.</div>
        <div id="mvLinks"></div>
        <div class="mv-panel-title mv-gap">Decisions</div>
        <div id="mvDecisions" class="mv-decisions"></div>
      </div>
      <div class="mv-panel mv-right" id="mvIncident">
        <div class="mv-panel-title">Incident</div>
        <div class="mv-note">No active incident. The mesh is watching.</div>
      </div>
      <div class="mv-controls">
        <div class="mv-chat-log" id="mvChatLog"></div>
        <form class="mv-chat-form" id="mvChatForm" autocomplete="off">
          <input id="mvChatInput" placeholder="Type a place: Five Points Station, Grady, 55 Gilmer St SE…" />
          <button type="submit">📍 Pin</button>
        </form>
        <div class="mv-row">
          <div class="mv-switches" id="mvSwitches"></div>
          <button id="mvSimulate" class="secondary">Simulate incident</button>
          <button id="mvJoin" class="secondary">📱 Join</button>
        </div>
      </div>
      <div class="mv-legend" id="mvLegend"></div>
      <div class="mv-join hidden" id="mvJoinSheet"></div>
    `;
    this.canvas = this.root.querySelector<HTMLCanvasElement>("#mvOverlay")!;

    this.map = L.map(this.root.querySelector<HTMLElement>("#mvMap")!, {
      zoomControl: false,
      preferCanvas: true,
      attributionControl: true,
    }).setView([scene().lat, scene().lng], 13);
    // OpenStreetMap's own tiles (no API key), darkened with a CSS filter on
    // the tile pane (.mv-map .leaflet-tile-pane in style.css).
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap contributors · MARTA GTFS",
    }).addTo(this.map);
    L.control.zoom({ position: "bottomright" }).addTo(this.map);

    const [marta, places] = await Promise.all([loadMarta(), loadPlaces()]);
    if (dataProblems.length) this.showDataProblem();
    this.drawTransit(marta.routes, marta.stops);
    this.drawPlaces(places);
    const buses = await loadLiveBuses(busesUrl());
    this.liveBusLayer.addTo(this.map);
    this.drawLiveBuses(buses);
    setInterval(async () => this.drawLiveBuses(await loadLiveBuses(busesUrl())), BUS_REFRESH_MS);

    this.cityData = { places, stops: marta.stops, buses };
    this.cityLayer.addTo(this.map);
    this.peopleLayer.addTo(this.map);
    this.buildCityMesh(scene());
    this.wireSession();
    this.renderLegend();
    this.renderSwitches();
    this.root.querySelector<HTMLButtonElement>("#mvSimulate")!.addEventListener("click", () => {
      raiseIncident(this.session.node, "Simulated incident (rehearsal)");
    });
    this.wireChat(places);
    this.root.querySelector<HTMLButtonElement>("#mvJoin")!.addEventListener("click", () => void this.toggleJoin());
    setInterval(() => this.refreshDevices(), 1000);
    this.refreshDevices();
  }

  private showDataProblem() {
    const el = document.createElement("div");
    el.className = "mv-data-problem";
    el.innerHTML = `<b>Map data didn't load.</b> ${dataProblems.map(esc).join(" · ")}<br><span class="mv-tip">The mesh, devices and alerts still work. If this is the hosted site, its signed media links may have expired: run npm run media:sign and redeploy.</span>`;
    this.root.append(el);
  }

  // ---- static layers --------------------------------------------------------

  private drawTransit(routes: { type: string; color: string; path: [number, number][] }[], stops: [number, number, string, number][]) {
    for (const r of routes) {
      if (r.type === "rail") {
        L.polyline(r.path, { color: r.color, weight: 10, opacity: 0.12, interactive: false }).addTo(this.map);
        L.polyline(r.path, { color: r.color, weight: 3.5, opacity: 0.95, interactive: false }).addTo(this.map);
      } else {
        L.polyline(r.path, { color: "#2dd4bf", weight: 1, opacity: 0.16, interactive: false }).addTo(this.map);
      }
    }
    for (const [lat, lng] of stops) {
      L.circleMarker([lat, lng], { radius: 1.2, stroke: false, fillColor: "#5eead4", fillOpacity: 0.22, interactive: false }).addTo(this.map);
    }
    this.stats.routes = routes.length;
    this.stats.stops = stops.length;
  }

  private drawPlaces(places: Place[]) {
    for (const p of places) {
      const big = p.cat === "police" || p.cat === "hospital" || p.cat === "fire";
      L.circleMarker([p.lat, p.lng], {
        radius: big ? 3.2 : 2.2,
        stroke: false,
        fillColor: KIND_COLOR[p.cat],
        fillOpacity: big ? 0.5 : 0.3,
      })
        .bindTooltip(`${esc(p.name)}<br><span class="mv-tip">${KIND_LABEL[p.cat]}</span>`)
        .addTo(this.map);
    }
    this.stats.places = places.length;
  }

  private drawLiveBuses(buses: LiveBus[]) {
    const seen = new Set<string>();
    for (const b of buses) {
      seen.add(b.id);
      const existing = this.liveBusMarkers.get(b.id);
      if (existing) existing.setLatLng([b.lat, b.lng]);
      else {
        const m = L.circleMarker([b.lat, b.lng], { radius: 3, color: "#0f172a", weight: 1, opacity: 0.5, fillColor: "#f8fafc", fillOpacity: 0.55 }).bindTooltip(
          `MARTA bus ${esc(b.id)}${b.route ? ` · Route ${esc(b.route)}` : ""}<br><span class="mv-tip">live position</span>`,
        );
        m.addTo(this.liveBusLayer);
        this.liveBusMarkers.set(b.id, m);
      }
    }
    for (const [id, m] of this.liveBusMarkers) {
      if (!seen.has(id)) {
        this.liveBusLayer.removeLayer(m);
        this.liveBusMarkers.delete(id);
      }
    }
    this.stats.liveBuses = buses.length;
    this.renderCounters();
  }

  // ---- simulated city mesh ---------------------------------------------------

  private buildCityMesh(center: { lat: number; lng: number }) {
    this.meshCenter = { lat: center.lat, lng: center.lng };
    const gen = ++this.cityGeneration;
    const { places, stops, buses } = this.cityData;
    const candidates: { kind: CityKind; name: string; lat: number; lng: number }[] = [];
    for (const p of places) candidates.push({ kind: p.cat, name: p.name, lat: p.lat, lng: p.lng });
    const stationNames = new Set<string>();
    for (const [lat, lng, name, isStation] of stops) {
      const key = name.replace(/\s+(Station|Stn).*$/i, "");
      if (!isStation || stationNames.has(key)) continue;
      stationNames.add(key);
      candidates.push({ kind: "station", name: `${key} Station`, lat, lng });
    }
    for (const b of buses) candidates.push({ kind: "bus", name: `Bus ${b.id}${b.route ? ` · Route ${b.route}` : ""}`, lat: b.lat, lng: b.lng });

    const picked: typeof candidates = [];
    const byKind = new Map<CityKind, typeof candidates>();
    for (const c of candidates) {
      if (metersBetween(c, center) > CITY_RADIUS_M) continue;
      byKind.set(c.kind, [...(byKind.get(c.kind) ?? []), c]);
    }
    for (const [kind, list] of byKind) {
      list.sort((a, b) => metersBetween(a, center) - metersBetween(b, center));
      picked.push(...list.slice(0, CITY_CAPS[kind] ?? 0));
    }

    picked.forEach((c, i) => this.addCityNode({ ...c, id: `city${gen}-${c.kind}-${i}` }));

    const pairs = buildRadioMesh(
      this.city.map((c) => c.node),
      RADIO_RANGE_M,
    );
    for (const [a, b] of pairs) this.drawRadioLink(a, b, 0.22);

    // The camera's radio reaches the nearest city nodes, if the mesh is around it.
    const cam = this.session.node;
    const near = this.city
      .filter((c) => metersBetween(c.node.info, cam.info) <= RADIO_RANGE_M)
      .sort((a, b) => metersBetween(a.node.info, cam.info) - metersBetween(b.node.info, cam.info))
      .slice(0, CAMERA_RADIO_NEIGHBORS);
    for (const c of near) {
      radioLink(cam, c.node, { exclude: CITY_EXCLUDE });
      this.drawRadioLink(cam, c.node, 0.5);
    }

    // Infrastructure has wired internet: an uplink to the cloud relay this
    // console sits on, so an alert anywhere in the city also reaches the
    // devices in the room. "Internet down" cuts these.
    let uplinks = 0;
    for (const c of this.city) {
      if (!BACKHAUL_KINDS.has(c.kind)) continue;
      this.uplink(c);
      uplinks++;
    }
    this.stats.radioLinks = pairs.length + near.length;
    this.stats.uplinks = uplinks;
    this.applyOutageToCity();
    this.renderCounters();
  }

  private addCityNode(c: { id: string; kind: CityKind; name: string; lat: number; lng: number }): CityNode {
    const node = new MeshNode({ name: c.name, role: c.kind, lat: c.lat, lng: c.lng, virtual: true }, { id: c.id, presence: false, pingMs: 10_000 });
    const marker = L.circleMarker([c.lat, c.lng], {
      radius: c.kind === "cctv" ? 7 : 5,
      color: KIND_COLOR[c.kind],
      weight: 2,
      opacity: 0.6,
      fillColor: "#0b1220",
      fillOpacity: NODE_FILL_IDLE,
    }).bindTooltip(
      `${esc(c.name)}<br><span class="mv-tip">${KIND_LABEL[c.kind]} · mesh node (simulated radio${BACKHAUL_KINDS.has(c.kind) ? " + internet uplink" : ""})</span>`,
    );
    marker.addTo(this.cityLayer);
    const cn: CityNode = { node, kind: c.kind, name: c.name, marker, reachedAt: null, hops: null };
    this.city.push(cn);
    this.cityById.set(node.id, cn);
    // A torn-down mesh still has packets in flight; ignore them.
    const current = () => this.cityById.get(node.id) === cn;
    node.onPacket((e) => current() && this.onPacket(e));
    // Arrives just after the store change that marked this node reached, so
    // the panel redraws to pick up the hop count.
    node.onDeliver((e) => {
      if (!current() || cn.hops !== null || !e.colls.includes("incidents")) return;
      cn.hops = e.hops;
      this.scheduleIncidentRender();
    });
    // The store reports each field as it lands; wait for the whole update
    // (a new incident's location and time arrive in the same batch).
    node.store.onChange((coll, id) => {
      if (coll !== "incidents") return;
      queueMicrotask(() => {
        if (!current()) return;
        if (id !== this.currentIncident) this.startIncident(id);
        else if (cn.reachedAt === null) this.markReached(cn);
      });
    });
    return cn;
  }

  private drawRadioLink(a: MeshNode, b: MeshNode, opacity: number) {
    L.polyline(
      [
        [a.info.lat, a.info.lng],
        [b.info.lat, b.info.lng],
      ],
      { color: "#38bdf8", weight: opacity > 0.3 ? 1.5 : 1, opacity, interactive: false },
    ).addTo(this.cityLayer);
  }

  private uplink(c: CityNode) {
    const me = this.session.node;
    const clone = (w: Wire) => JSON.parse(JSON.stringify(w)) as Wire;
    me.addLink(c.node.id, "relay", (w) => setTimeout(() => c.node.receive(me.id, "relay", clone(w)), backhaulDelayMs()), { exclude: CITY_EXCLUDE });
    c.node.addLink(me.id, "relay", (w) => setTimeout(() => me.receive(c.node.id, "relay", clone(w)), backhaulDelayMs()));
  }

  // City nodes live in this page, so the network-wide outage switches have to
  // reach their radios and uplinks too.
  private applyOutageToCity() {
    const doc = this.session.node.store.get(OUTAGE_DOC.c, OUTAGE_DOC.id);
    for (const cn of this.city) {
      cn.node.setDisabled("radio", !!doc?.radio);
      cn.node.setDisabled("relay", !!doc?.relay);
    }
  }

  private teardownCity() {
    const me = this.session.node;
    for (const cn of this.city) {
      cn.node.stop();
      me.removeLink(cn.node.id, "radio");
      me.removeLink(cn.node.id, "relay");
    }
    this.city = [];
    this.cityById.clear();
    this.pinCam = null;
    this.cityLayer.clearLayers();
  }

  private markReached(cn: CityNode) {
    const inc = this.currentIncident ? this.incidentDoc(this.currentIncident) : null;
    if (inc?.at === undefined) return;
    cn.reachedAt = Date.now() - Number(inc.at);
    // Hollow ring → filled with its own category color: the wave is visible
    // and police still look like police. A white flash marks the moment.
    const base = this.nodeStyle(cn);
    cn.marker.setStyle({ ...base, fillColor: KIND_COLOR[cn.kind], fillOpacity: NODE_FILL_REACHED, color: "#ffffff" });
    setTimeout(() => {
      if (cn.reachedAt !== null) cn.marker.setStyle({ ...this.nodeStyle(cn), fillColor: KIND_COLOR[cn.kind], fillOpacity: NODE_FILL_REACHED });
    }, 700);
    this.ripples.push({ at: L.latLng(cn.node.info.lat, cn.node.info.lng), start: performance.now(), color: KIND_COLOR[cn.kind], maxR: 22, ms: 900 });
    this.scheduleIncidentRender();
  }

  // The nearest node of each responder kind this problem needs (plus the
  // nearest MARTA station) is drawn bigger with a white ring.
  private responderIds = new Set<string>();
  private pickResponders(inc: Record<string, unknown>) {
    const where = { lat: Number(inc.lat ?? scene().lat), lng: Number(inc.lng ?? scene().lng) };
    const kinds: CityKind[] = [...(PROBLEMS[inc.kind as ProblemId]?.responders ?? ["police", "fire", "hospital"]), "station"];
    this.responderIds.clear();
    for (const kind of kinds) {
      const c = this.city.filter((x) => x.kind === kind).sort((a, b) => metersBetween(a.node.info, where) - metersBetween(b.node.info, where))[0];
      if (c) this.responderIds.add(c.node.id);
    }
  }

  private nodeStyle(cn: CityNode): L.CircleMarkerOptions {
    const responder = this.responderIds.has(cn.node.id);
    return {
      radius: responder ? 9 : cn.kind === "cctv" ? 7 : 5,
      color: responder ? "#ffffff" : KIND_COLOR[cn.kind],
      weight: responder ? 3 : 2,
      opacity: responder ? 0.95 : 0.6,
    };
  }

  // Up to ~150 nodes get the alert within a second; draw the panel once per frame.
  private renderQueued = false;
  private scheduleIncidentRender() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.renderIncident();
    });
  }

  // ---- real devices ------------------------------------------------------------

  private wireSession() {
    const node = this.session.node;
    node.onPacket((e) => this.onPacket(e));
    // Only links to real devices: dozens of city uplinks would bury them.
    const isDevice = (peer: string) => !peer.startsWith("city");
    this.decisions = node.recentDecisions.filter((d) => isDevice(d.peer)).slice(0, 10);
    this.renderDecisions();
    node.onDecision((d) => {
      if (!isDevice(d.peer)) return;
      this.decisions.unshift(d);
      this.decisions = this.decisions.slice(0, 10);
      this.renderDecisions();
    });
    node.onLinksChanged(() => this.refreshDevices());
    node.store.onChange((c, id) => {
      if (c === "incidents") {
        // After the whole batch of fields has landed (see addCityNode).
        queueMicrotask(() => {
          if (id !== this.currentIncident) this.startIncident(id);
          else this.scheduleIncidentRender();
        });
      }
      if (c === "frames" || c === "people") this.scheduleIncidentRender();
      if (c === OUTAGE_DOC.c && id === OUTAGE_DOC.id) {
        this.applyOutageToCity();
        this.renderSwitches();
      }
    });
    const existing = node.store.list("incidents").sort((a, b) => Number(b.at) - Number(a.at))[0];
    if (existing) this.startIncident(existing.id);
  }

  // The console can watch an incident it hasn't received itself (internet and
  // radio to this device both down): read it from whichever node has it.
  private incidentDoc(id: string) {
    return this.session.node.store.get("incidents", id) ?? this.city.map((c) => c.node.store.get("incidents", id)).find(Boolean);
  }

  private startIncident(id: string) {
    const inc = this.incidentDoc(id);
    if (!inc || inc.at === undefined) return;
    // An older incident replaying (a device rejoining, an ack on an old one)
    // doesn't take the screen from a newer one.
    const cur = this.currentIncident ? this.incidentDoc(this.currentIncident) : null;
    if (cur && Number(cur.at) > Number(inc.at)) return;
    this.currentIncident = id;
    this.pickResponders(inc);
    for (const cn of this.city) {
      cn.reachedAt = null;
      cn.hops = null;
      cn.marker.setStyle({ ...this.nodeStyle(cn), fillColor: "#0b1220", fillOpacity: NODE_FILL_IDLE });
      if (this.responderIds.has(cn.node.id)) cn.marker.bringToFront();
      if (cn.node.store.get("incidents", id)) this.markReached(cn);
    }
    const at = L.latLng(Number(inc.lat ?? scene().lat), Number(inc.lng ?? scene().lng));
    this.ripples.push({ at, start: performance.now(), color: "#ef4444", maxR: 160, ms: 2200 });
    this.ripples.push({ at, start: performance.now() + 500, color: "#ef4444", maxR: 160, ms: 2200 });
    this.map.flyTo(at, 14, { duration: 1.2 });
    this.renderIncident();
  }

  private devices() {
    const now = Date.now();
    return this.session.node.store
      .list("presence")
      .filter((p) => !p.virtual && now - Number(p.seen ?? 0) < DEVICE_STALE_MS)
      .map((p) => ({ id: p.id, name: String(p.name), role: String(p.role) as Role, lat: Number(p.lat), lng: Number(p.lng), links: (p.links ?? {}) as Record<string, TransportKind> }));
  }

  private refreshDevices() {
    const devs = this.devices();
    const live = new Set(devs.map((d) => d.id));
    for (const d of devs) {
      const html = `<div class="dev-pin dev-${d.role}${d.id === this.session.node.id ? " dev-self" : ""}"><span>${ROLE_ICON[d.role] ?? "📱"}</span><div class="dev-label">${esc(d.name)}</div></div>`;
      const m = this.deviceMarkers.get(d.id);
      if (m) {
        m.setLatLng([d.lat, d.lng]);
        m.setIcon(L.divIcon({ className: "", html, iconSize: [36, 36], iconAnchor: [18, 18] }));
      } else {
        this.deviceMarkers.set(d.id, L.marker([d.lat, d.lng], { icon: L.divIcon({ className: "", html, iconSize: [36, 36], iconAnchor: [18, 18] }), zIndexOffset: 1000 }).addTo(this.map));
      }
    }
    for (const [id, m] of this.deviceMarkers) {
      if (!live.has(id)) {
        m.remove();
        this.deviceMarkers.delete(id);
      }
    }
    // Device-to-device links as each device reports them (its own presence doc).
    const pos = new Map(devs.map((d) => [d.id, d]));
    const want = new Set<string>();
    for (const d of devs) {
      for (const [peer, kind] of Object.entries(d.links)) {
        const other = pos.get(peer);
        if (!other || d.id > peer) continue;
        const key = `${d.id}|${peer}`;
        want.add(key);
        const style = TRANSPORT_STYLE[kind] ?? TRANSPORT_STYLE.relay;
        const line = this.deviceLinks.get(key);
        const latlngs: L.LatLngExpression[] = [
          [d.lat, d.lng],
          [other.lat, other.lng],
        ];
        if (line) line.setLatLngs(latlngs).setStyle(style);
        else this.deviceLinks.set(key, L.polyline(latlngs, { ...style, interactive: false }).addTo(this.map));
      }
    }
    for (const [key, line] of this.deviceLinks) {
      if (!want.has(key)) {
        line.remove();
        this.deviceLinks.delete(key);
      }
    }
    this.renderLinks(devs);
    this.renderCounters();
  }

  private positionOf(id: string): L.LatLng | null {
    const cn = this.cityById.get(id);
    if (cn) return L.latLng(cn.node.info.lat, cn.node.info.lng);
    if (id === this.session.node.id) return L.latLng(this.session.node.info.lat, this.session.node.info.lng);
    const p = this.session.node.store.get("presence", id);
    return p ? L.latLng(Number(p.lat), Number(p.lng)) : null;
  }

  private onPacket(e: PacketEvent) {
    const from = this.positionOf(e.from);
    const to = this.positionOf(e.to);
    if (!from || !to || from.equals(to)) return;
    if (this.packets.length > 600) return; // keep the frame rate during the wave
    this.packets.push({ from, to, start: performance.now(), incident: e.colls.includes("incidents") });
  }

  // ---- join: QR codes that put a phone on this mesh in one scan --------------------

  private async toggleJoin() {
    const sheet = this.root.querySelector<HTMLDivElement>("#mvJoinSheet")!;
    if (!sheet.classList.contains("hidden")) {
      sheet.classList.add("hidden");
      return;
    }
    const local = /^(localhost|127\.|\[::1\])/.test(location.hostname);
    const hub = hubUrl();
    const link = (role: string) => `${location.origin}/?role=${role}${hub ? `&hub=${encodeURIComponent(hub)}` : ""}`;
    const roles = [
      ["police", "🚓 Police"],
      ["driver", "🧑‍✈️ Driver"],
      ["rider", "🧍 Rider"],
    ];
    const codes = await Promise.all(roles.map(([r]) => QRCode.toDataURL(link(r), { margin: 1, width: 220, color: { dark: "#0b0e13", light: "#ffffff" } })));
    sheet.innerHTML = `
      <div class="mv-panel-title">Join the mesh</div>
      <div class="mv-note">${local ? "This console is on localhost, which phones can't open. Use the hosted site (npm run tunnel prints the link) for scannable codes." : "Scan with a phone camera: it opens the app on this mesh."}</div>
      <div class="mv-qrs">${roles
        .map(([, label], i) => `<figure><img src="${codes[i]}" alt="${label} join code"/><figcaption>${label}</figcaption></figure>`)
        .join("")}</div>`;
    sheet.classList.remove("hidden");
  }

  // ---- chat: type a place, get a pin -------------------------------------------

  private wireChat(places: Place[]) {
    const form = this.root.querySelector<HTMLFormElement>("#mvChatForm")!;
    const input = this.root.querySelector<HTMLInputElement>("#mvChatInput")!;
    const log = this.root.querySelector<HTMLDivElement>("#mvChatLog")!;
    // Typing in the chat must not switch tabs (main.ts listens for 1/2/3).
    input.addEventListener("keydown", (e) => e.stopPropagation());

    const say = (who: "you" | "eyes", html: string) => {
      const line = document.createElement("div");
      line.className = `mv-msg mv-msg-${who}`;
      line.innerHTML = html;
      log.append(line);
      while (log.children.length > 8) log.firstElementChild!.remove();
      log.scrollTop = log.scrollHeight;
    };

    const run = (hit: GeoResult, problem: ProblemId) => {
      const p = PROBLEMS[problem];
      say("eyes", `${p.icon} Simulating <b>${p.label.toLowerCase()}</b> at ${esc(hit.name)}: CCTV there detects it, the mesh carries it.`);
      void this.simulateAt(hit, problem);
    };

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const q = input.value.trim();
      if (!q) return;
      input.value = "";
      say("you", esc(q));
      // "fire at Five Points" / "robbery near Grady": pin and simulate at once.
      const cmd = q.match(/^(.*?)\s+(?:at|near|in|@)\s+(.+)$/i);
      const problem = cmd ? problemFromText(cmd[1]) : null;
      const placeText = problem && cmd ? cmd[2] : q;
      const hit = await geocode(placeText);
      if (!hit) {
        say("eyes", `Couldn't find “${esc(placeText)}” in Atlanta. Try a station, building, or street address.`);
        return;
      }
      this.dropPin(hit);
      const near = (cat: PlaceCategory) => {
        const p = nearest(places, cat, hit);
        return p ? `${esc(p.name)} <span class="mv-tip">${fmtKm(metersBetween(p, hit))}</span>` : "—";
      };
      const stop = await nearestStop(hit);
      say(
        "eyes",
        `📍 <b>${esc(hit.name)}</b> <span class="mv-tip">${hit.source === "osm" ? "OpenStreetMap search" : KIND_LABEL[hit.kind as CityKind] ?? "MARTA stop"}</span>` +
          `<div class="mv-msg-near">🚓 ${near("police")}<br>🏥 ${near("hospital")}<br>🚒 ${near("fire")}${stop ? `<br>🚉 ${esc(stop.name)} <span class="mv-tip">${fmtKm(stop.m)}</span>` : ""}</div>` +
          (problem
            ? ""
            : `<div class="mv-probs">${(Object.entries(PROBLEMS) as [ProblemId, (typeof PROBLEMS)[ProblemId]][])
                .map(([id, p]) => `<button type="button" class="secondary" data-p="${id}">${p.icon} ${p.label}</button>`)
                .join("")}</div>`),
      );
      const reply = log.lastElementChild as HTMLElement;
      reply.querySelectorAll<HTMLButtonElement>("button[data-p]").forEach((b) => b.addEventListener("click", () => run(hit, b.dataset.p as ProblemId)));
      if (problem) run(hit, problem);
    });
  }

  // Pin problem simulator: a CCTV node appears at the pin (radio-linked to its
  // nearest city nodes) and raises the problem there. If the pin is beyond the
  // current city mesh, the mesh is rebuilt around it first.
  private async simulateAt(hit: GeoResult, problem: ProblemId) {
    if (metersBetween(hit, this.meshCenter) > REBUILD_FROM_M) {
      this.teardownCity();
      this.buildCityMesh(hit);
    }
    if (this.pinCam) {
      // Previous pin's camera leaves the mesh.
      this.pinCam.node.stop();
      for (const peer of this.pinCam.node.neighbors()) this.cityById.get(peer)?.node.removeLink(this.pinCam.node.id, "radio");
      this.pinCam.marker.remove();
      this.city = this.city.filter((c) => c !== this.pinCam);
      this.cityById.delete(this.pinCam.node.id);
    }
    const cam = this.addCityNode({ id: `city${this.cityGeneration}-cctv-${Date.now().toString(36)}`, kind: "cctv", name: `CCTV · ${hit.name}`, lat: hit.lat, lng: hit.lng });
    this.pinCam = cam;
    const near = this.city
      .filter((c) => c !== cam)
      .sort((a, b) => metersBetween(a.node.info, hit) - metersBetween(b.node.info, hit))
      .slice(0, CAMERA_RADIO_NEIGHBORS);
    for (const c of near) {
      radioLink(cam.node, c.node);
      this.drawRadioLink(cam.node, c.node, 0.5);
    }
    this.applyOutageToCity();
    // Give the new links a moment to come up before the alert goes out.
    await new Promise((r) => setTimeout(r, 400));
    const p = PROBLEMS[problem];
    cam.node.write("incidents", `inc-${Date.now().toString(36)}`, {
      kind: problem,
      status: "detected",
      detail: `${p.label} detected by CCTV (simulated)`,
      label: hit.name,
      lat: hit.lat,
      lng: hit.lng,
      camera: cam.node.id,
      at: Date.now(),
    });
  }

  private pin: L.Marker | null = null;

  private dropPin(hit: GeoResult) {
    this.pin?.remove();
    this.pin = L.marker([hit.lat, hit.lng], {
      icon: L.divIcon({ className: "", html: `<div class="mv-pin">📍</div>`, iconSize: [36, 36], iconAnchor: [18, 34] }),
      zIndexOffset: 2000,
    })
      .bindTooltip(esc(hit.name), { direction: "top", offset: [0, -30], permanent: true, className: "mv-pin-label" })
      .addTo(this.map);
    this.ripples.push({ at: L.latLng(hit.lat, hit.lng), start: performance.now(), color: "#f8fafc", maxR: 60, ms: 1200 });
    this.map.flyTo([hit.lat, hit.lng], Math.max(this.map.getZoom(), 15), { duration: 1 });
  }

  // ---- HUD ---------------------------------------------------------------------

  private renderCounters() {
    const el = this.root.querySelector("#mvCounters");
    if (!el) return;
    const devs = this.devices().length;
    el.innerHTML = [
      `<b>${devs}</b> live devices`,
      `<b>${this.city.length}</b> city mesh nodes`,
      `<b>${this.stats.radioLinks}</b> radio links`,
      `<b>${this.stats.liveBuses}</b> MARTA buses live`,
      `<b>${this.stats.places.toLocaleString()}</b> public places`,
      `<b>${this.stats.stops.toLocaleString()}</b> stops · ${this.stats.routes} routes`,
    ]
      .map((s) => `<span>${s}</span>`)
      .join("");
  }

  private renderLinks(devs: ReturnType<MeshView["devices"]>) {
    const el = this.root.querySelector("#mvLinks");
    if (!el) return;
    const node = this.session.node;
    const rows = devs
      .filter((d) => d.id !== node.id)
      .map((d) => {
        const chosen = node.route(d.id);
        const stats = node.linkStats(d.id);
        const cells = stats
          .sort((a, b) => a.kind.localeCompare(b.kind))
          .map(
            (s) =>
              `<span class="tx ${s.kind === chosen ? "tx-chosen" : ""} ${s.alive ? "" : "tx-down"}">${TRANSPORT_LABEL[s.kind]} ${
                s.alive ? (s.rttMs === null ? "…" : `${Math.round(s.rttMs)} ms`) : "down"
              }</span>`,
          )
          .join("");
        const via = chosen ? "" : `<span class="tx tx-hop">via mesh hops</span>`;
        return `<div class="mv-linkrow"><div class="mv-linkname">${ROLE_ICON[d.role] ?? "📱"} ${esc(d.name)}</div><div class="mv-tx">${cells}${via}</div></div>`;
      });
    el.innerHTML = rows.join("") || `<div class="mv-note">No other devices yet. Open this page on a phone with ?role=driver, rider or police.</div>`;
  }

  private renderDecisions() {
    const el = this.root.querySelector("#mvDecisions");
    if (!el) return;
    const name = (id: string) => String(this.session.node.store.get("presence", id)?.name ?? this.cityById.get(id)?.name ?? id);
    el.innerHTML = this.decisions
      .map((d) => {
        const t = new Date(d.at).toLocaleTimeString([], { hour12: false });
        const arrow = `${d.from ? TRANSPORT_LABEL[d.from] : "—"} → <b>${d.to ? TRANSPORT_LABEL[d.to] : "mesh hops"}</b>`;
        return `<div class="mv-dec"><span class="mv-t">${t}</span> ${esc(name(d.peer))}: ${arrow}<div class="mv-why">${esc(d.reason)}</div></div>`;
      })
      .join("");
  }

  private renderSwitches() {
    const el = this.root.querySelector("#mvSwitches");
    if (!el) return;
    const doc = this.session.node.store.get(OUTAGE_DOC.c, OUTAGE_DOC.id);
    const label: Record<string, string> = { relay: "Internet", p2p: "P2P Wi-Fi", radio: "Mesh radio" };
    el.innerHTML = SWITCHABLE.map((k) => {
      const down = !!doc?.[k];
      return `<button class="mv-switch ${down ? "mv-down" : ""}" data-k="${k}">${down ? "✕" : "●"} ${label[k]} ${down ? "DOWN" : "up"}</button>`;
    }).join("");
    el.querySelectorAll<HTMLButtonElement>("button").forEach((b) =>
      b.addEventListener("click", () => {
        const k = b.dataset.k as TransportKind;
        setOutage(this.session.node, k, !doc?.[k]);
      }),
    );
  }

  private renderLegend() {
    const el = this.root.querySelector("#mvLegend")!;
    const kinds: CityKind[] = ["police", "fire", "hospital", "station", "bus", "school", "university", "library", "stadium", "mall", "venue", "park"];
    el.innerHTML =
      kinds.map((k) => `<span><i style="background:${KIND_COLOR[k]}"></i>${KIND_LABEL[k]}</span>`).join("") +
      (["p2p", "relay", "radio"] as TransportKind[]).map((k) => `<span><i class="line" style="background:${TRANSPORT_STYLE[k].color}"></i>${TRANSPORT_LABEL[k]}</span>`).join("");
  }

  private renderIncident() {
    const el = this.root.querySelector("#mvIncident");
    if (!el || !this.currentIncident) return;
    const inc = this.incidentDoc(this.currentIncident);
    if (!inc) return;
    const where = { lat: Number(inc.lat ?? scene().lat), lng: Number(inc.lng ?? scene().lng) };
    const problem = PROBLEMS[inc.kind as ProblemId];
    this.drawPeople(inc, where);
    // Raised before this console opened: arrival times would measure page
    // load, not the mesh, so don't show them as if they were.
    const old = Number(inc.at) < this.openedAt - 1000;
    const reached = this.city.filter((c) => c.reachedAt !== null);
    const slowest = Math.max(0, ...reached.map((c) => c.reachedAt!));
    const firstOf = (kind: CityKind) =>
      this.city.filter((c) => c.kind === kind).sort((a, b) => metersBetween(a.node.info, where) - metersBetween(b.node.info, where))[0];
    const responder = (kind: CityKind) => {
      const c = firstOf(kind);
      if (!c) return "";
      const state =
        c.reachedAt === null
          ? `<span class="mv-wait">waiting…</span>`
          : old
            ? `<span class="mv-ok">✓ synced</span>`
            : `<span class="mv-ok">✓ ${(c.reachedAt / 1000).toFixed(2)} s · ${c.hops ?? "?"} hop${c.hops === 1 ? "" : "s"}</span>`;
      return `<div class="mv-resp"><i style="background:${KIND_COLOR[kind]}"></i><div><div>${esc(c.name)}</div><div class="mv-why">${KIND_LABEL[kind]} · ${fmtKm(metersBetween(c.node.info, where))} away</div></div>${state}</div>`;
    };
    const responderKinds: CityKind[] = [...(problem?.responders ?? ["police", "fire", "hospital"]), "station"];
    const acks = Object.entries(inc)
      .filter(([k]) => k.startsWith("ack:"))
      .map(
        ([k, v]) =>
          `<div class="mv-ack">✓ ${esc(k.slice(4))}${inc[`status:${k.slice(4)}`] ? ` · ${esc(String(inc[`status:${k.slice(4)}`]))}` : ""} · ${new Date(Number(v)).toLocaleTimeString([], { hour12: false })}</div>`,
      )
      .join("");
    const frames = framesFor(this.session.node.store, inc.id);
    const latest = frames.length ? safeJpeg(frames[frames.length - 1].data) : null;
    el.innerHTML = `
      ${latest ? `<img class="mv-frame" src="${latest}" alt="Latest camera frame"/><div class="mv-why">📸 ${frames.length} frame${frames.length === 1 ? "" : "s"} sent to police · latest ${new Date(Number(frames[frames.length - 1].at)).toLocaleTimeString([], { hour12: false })}</div>` : ""}
      <div class="mv-panel-title mv-alert">${problem?.icon ?? "⚠"} ${esc(problemLabel(inc.kind)).toUpperCase()}${Number(inc.punches ?? 0) > 1 ? ` · ${Number(inc.punches)} DETECTIONS` : ""}</div>
      <div class="mv-inc-where">${esc(String(inc.label ?? scene().label))}</div>
      <div class="mv-why">${esc(String(inc.detail ?? ""))}</div>
      <div class="mv-why">${new Date(Number(inc.at)).toLocaleTimeString([], { hour12: false })} · status <b>${esc(String(inc.status ?? "detected"))}</b></div>
      <div class="mv-panel-title mv-gap">Spread across the city mesh</div>
      <div class="mv-bar"><div style="width:${this.city.length ? (100 * reached.length) / this.city.length : 0}%"></div></div>
      <div class="mv-why">${reached.length} / ${this.city.length} nodes${old ? " · synced when this console opened" : ` · ${(slowest / 1000).toFixed(2)} s`}</div>
      <div class="mv-panel-title mv-gap">Nearest responders</div>
      ${responderKinds.map(responder).join("")}
      ${this.peopleHtml(inc)}
      ${acks ? `<div class="mv-panel-title mv-gap">Acknowledged</div>${acks}` : ""}
    `;
  }

  private peopleHtml(inc: Record<string, unknown>) {
    const rows: string[] = [];
    if (inc.areaNotified !== undefined) {
      const n = Number(inc.areaNotified);
      rows.push(
        n
          ? `<div class="mv-why">⚠️ <b>${n}</b> ${n === 1 ? "person" : "people"} within ${fmtKm(Number(inc.areaRadiusM))} alerted · <span class="mv-ok">${Number(inc.areaSafe ?? 0)} safe</span>${Number(inc.areaHelp ?? 0) ? ` · <span class="mv-help">${Number(inc.areaHelp)} need help</span>` : ""} · ${Math.max(0, n - Number(inc.areaSafe ?? 0) - Number(inc.areaHelp ?? 0))} not answered</div>`
          : `<div class="mv-why">Nobody signed up within ${fmtKm(Number(inc.areaRadiusM))}</div>`,
      );
    }
    if (inc.injured) rows.push(`<div class="mv-why">🩹 <b>${Number(inc.injured)}</b> hurt · ${esc(String(inc.care ?? ""))}</div>`);
    return rows.length ? `<div class="mv-panel-title mv-gap">People nearby · iMessage</div>${rows.join("")}` : "";
  }

  private drawPeople(inc: Record<string, unknown>, where: { lat: number; lng: number }) {
    this.peopleLayer.clearLayers();
    if (inc.areaRadiusM)
      L.circle([where.lat, where.lng], { radius: Number(inc.areaRadiusM), color: "#f59e0b", weight: 1.5, opacity: 0.8, dashArray: "6 6", fillColor: "#f59e0b", fillOpacity: 0.05, interactive: false }).addTo(this.peopleLayer);
    for (const p of this.session.node.store.list("people")) {
      if (p.incident !== this.currentIncident) continue;
      const status = PERSON_COLOR[String(p.status)] ? String(p.status) : "notified";
      // A person pin, not a dot: the city nodes already use colored dots.
      const icon = L.divIcon({ className: "", html: `<div class="mv-person mv-person-${status}">${PERSON_GLYPH[status]}</div>`, iconSize: [26, 26], iconAnchor: [13, 13] });
      L.marker([Number(p.lat), Number(p.lng)], { icon, zIndexOffset: 1000 }).bindTooltip(PERSON_LABEL[status]).addTo(this.peopleLayer);
    }
  }

  // ---- animation ---------------------------------------------------------------

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    const c = this.canvas;
    const rect = this.root.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(rect.width * dpr) || c.height !== Math.round(rect.height * dpr)) {
      c.width = Math.round(rect.width * dpr);
      c.height = Math.round(rect.height * dpr);
    }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const now = performance.now();

    this.ripples = this.ripples.filter((r) => now - r.start < r.ms);
    for (const r of this.ripples) {
      const k = (now - r.start) / r.ms;
      if (k < 0) continue;
      const p = this.map.latLngToContainerPoint(r.at);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4 + k * r.maxR, 0, Math.PI * 2);
      ctx.strokeStyle = r.color;
      ctx.globalAlpha = 1 - k;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    this.packets = this.packets.filter((p) => now - p.start < PACKET_MS);
    for (const pk of this.packets) {
      const k = (now - pk.start) / PACKET_MS;
      const a = this.map.latLngToContainerPoint(pk.from);
      const b = this.map.latLngToContainerPoint(pk.to);
      const x = a.x + (b.x - a.x) * k;
      const y = a.y + (b.y - a.y) * k;
      const tail = Math.max(0, k - 0.18);
      ctx.globalAlpha = pk.incident ? 0.9 : 0.45;
      ctx.strokeStyle = pk.incident ? "#ef4444" : "#5eead4";
      ctx.lineWidth = pk.incident ? 2.5 : 1.2;
      ctx.beginPath();
      ctx.moveTo(a.x + (b.x - a.x) * tail, a.y + (b.y - a.y) * tail);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, pk.incident ? 3.5 : 2, 0, Math.PI * 2);
      ctx.fillStyle = pk.incident ? "#fecaca" : "#ccfbf1";
      ctx.globalAlpha = 1;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };
}

const fmtKm = (m: number) => (m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`);

async function nearestStop(at: { lat: number; lng: number }): Promise<{ name: string; m: number } | null> {
  const { stops } = await loadMarta();
  let best: { name: string; m: number } | null = null;
  for (const [lat, lng, name] of stops) {
    const m = metersBetween({ lat, lng }, at);
    if (!best || m < best.m) best = { name, m };
  }
  return best;
}

// The camera device writes the incident at its own current position (its real
// GPS fix when it has one); the mesh carries it everywhere.
export function raiseIncident(node: MeshNode, detail: string, frame?: string | null, evidence?: string) {
  const id = `inc-${Date.now().toString(36)}`;
  node.write("incidents", id, {
    kind: "assault",
    status: "detected",
    detail,
    ...(evidence ? { evidence } : {}),
    label: scene().label,
    lat: node.info.lat,
    lng: node.info.lng,
    accuracyM: scene().accuracyM ?? null,
    locationSource: scene().source,
    camera: node.id,
    punches: 1,
    at: Date.now(),
  });
  if (frame) addEvidenceFrame(node, id, 1, frame, detail);
  return id;
}

// One evidence frame per punch. Its own document, written after the alert:
// the alert stays tiny and lands first, each frame follows over the same
// mesh, and later punches add frames to the same incident.
export function addEvidenceFrame(node: MeshNode, incidentId: string, seq: number, data: string, caption: string) {
  node.write("frames", `${incidentId}#${String(seq).padStart(3, "0")}`, { incident: incidentId, seq, data, caption, at: Date.now() });
}

