import L from "leaflet";
import type { MeshNode, MeshLink, NodeStatus } from "./mesh";

const STATUS_COLOR: Record<NodeStatus, string> = {
  calm: "#2dd4a7",
  watching: "#f5c344",
  alert: "#f5c344",
  confirmed: "#ef4444",
  dismissed: "#6b7280",
};

const KIND_ICON: Record<MeshNode["kind"], string> = {
  bus: "🚌",
  driver: "🧑‍✈️",
  rider: "🧍",
  security: "🛡️",
  peer: "🚌",
};

export class MeshMap {
  private map: L.Map;
  private markers = new Map<string, L.Marker>();
  private nodeById = new Map<string, MeshNode>();
  private linkLines = new Map<string, L.Polyline>();
  private pulseLayer = L.layerGroup();

  constructor(container: HTMLElement, nodes: MeshNode[], links: MeshLink[]) {
    this.map = L.map(container, {
      zoomControl: false,
    }).setView([33.751, -84.39], 14.5);

    // Standard OSM tiles — free, no API key. Dark look comes from a CSS filter
    // on the tile pane (see style.css) rather than a key-gated dark tile provider.
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap contributors",
    }).addTo(this.map);

    L.control.zoom({ position: "bottomright" }).addTo(this.map);

    nodes.forEach((n) => {
      this.nodeById.set(n.id, n);
      const marker = this.makeMarker(n);
      marker.addTo(this.map);
      this.markers.set(n.id, marker);
    });

    links.forEach((link) => {
      const a = this.nodeById.get(link.from)!;
      const b = this.nodeById.get(link.to)!;
      const line = L.polyline(
        [
          [a.lat, a.lng],
          [b.lat, b.lng],
        ],
        {
          color: link.kind === "mesh" ? "#4fd1c5" : "#8b8ff0",
          weight: link.kind === "mesh" ? 3 : 2,
          opacity: link.kind === "mesh" ? 0.55 : 0.35,
          dashArray: link.kind === "cellular" ? "2 8" : undefined,
        },
      ).addTo(this.map);
      this.linkLines.set(`${link.from}->${link.to}`, line);
    });

    this.pulseLayer.addTo(this.map);
  }

  private makeMarker(n: MeshNode): L.Marker {
    const icon = L.divIcon({
      className: "",
      html: `
        <div class="node-pin node-${n.kind}" data-id="${n.id}">
          <div class="node-ring" style="--ring-color:${STATUS_COLOR[n.status]}"></div>
          <div class="node-emoji">${KIND_ICON[n.kind]}</div>
        </div>
      `,
      iconSize: [40, 40],
      iconAnchor: [20, 20],
    });
    const marker = L.marker([n.lat, n.lng], { icon });
    marker.bindTooltip(`<strong>${n.label}</strong>`, { direction: "top", offset: [0, -16] });
    return marker;
  }

  setNodeStatus(id: string, status: NodeStatus) {
    const n = this.nodeById.get(id);
    if (!n) return;
    n.status = status;
    const marker = this.markers.get(id)!;
    marker.setIcon(
      L.divIcon({
        className: "",
        html: `
          <div class="node-pin node-${n.kind} ${status === "alert" ? "node-pulsing" : ""}" data-id="${n.id}">
            <div class="node-ring" style="--ring-color:${STATUS_COLOR[status]}"></div>
            <div class="node-emoji">${KIND_ICON[n.kind]}</div>
          </div>
        `,
        iconSize: [40, 40],
        iconAnchor: [20, 20],
      }),
    );
  }

  pulseLink(fromId: string, toId: string) {
    const a = this.nodeById.get(fromId);
    const b = this.nodeById.get(toId);
    if (!a || !b) return;
    const key = `${fromId}->${toId}`;
    const line = this.linkLines.get(key) ?? this.linkLines.get(`${toId}->${fromId}`);
    if (line) {
      const el = (line as any)._path as SVGPathElement | undefined;
      if (el) {
        el.classList.remove("link-pulse");
        // force reflow so the animation restarts if triggered twice
        void el.getBoundingClientRect();
        el.classList.add("link-pulse");
      }
    }
    // small traveling dot for extra "wow"
    const dot = L.circleMarker([a.lat, a.lng], {
      radius: 5,
      color: "#fff",
      weight: 0,
      fillOpacity: 1,
    }).addTo(this.pulseLayer);
    const steps = 24;
    let i = 0;
    const iv = setInterval(() => {
      i++;
      const t = i / steps;
      const lat = a.lat + (b.lat - a.lat) * t;
      const lng = a.lng + (b.lng - a.lng) * t;
      dot.setLatLng([lat, lng]);
      if (i >= steps) {
        clearInterval(iv);
        this.pulseLayer.removeLayer(dot);
      }
    }, 20);
  }

  reset(nodes: MeshNode[]) {
    nodes.forEach((n) => this.setNodeStatus(n.id, "calm"));
  }
}
