// Mesh + dispatch simulation for the Eyes demo.
// This is the Step 1 skeleton: nodes, edges, and a scripted event flow.
// Real Ditto peer-to-peer sync replaces the mocked edge propagation in Step 2;
// real Photon dispatch replaces the mocked long-range call in a later step.

export type NodeKind = "bus" | "driver" | "rider" | "security" | "peer";
export type NodeStatus = "calm" | "watching" | "alert" | "confirmed" | "dismissed";

export interface MeshNode {
  id: string;
  kind: NodeKind;
  label: string;
  lat: number;
  lng: number;
  status: NodeStatus;
}

export type LinkKind = "mesh" | "cellular"; // mesh = Ditto BLE/P2P-WiFi, cellular = Photon/SMS

export interface MeshLink {
  from: string;
  to: string;
  kind: LinkKind;
}

// Atlanta / MARTA-anchored demo topology.
// bus/driver/rider are co-located on the vehicle (in local mesh range of each other).
// security is physically elsewhere — reachable only over the cellular/Photon path.
// peer is a second equipped vehicle a few blocks away — shows this is a network, not a pair.
export const NODES: MeshNode[] = [
  { id: "bus", kind: "bus", label: "Route 410 — Bus Camera", lat: 33.749, lng: -84.388, status: "calm" },
  { id: "driver", kind: "driver", label: "Driver — M. Alvarez", lat: 33.7493, lng: -84.3876, status: "calm" },
  { id: "rider", kind: "rider", label: "Rider — seat 14", lat: 33.7488, lng: -84.3883, status: "calm" },
  { id: "security", kind: "security", label: "MARTA Transit Security — Five Points", lat: 33.7537, lng: -84.3926, status: "calm" },
  { id: "peer", kind: "peer", label: "Route 410 — 2nd vehicle (mesh peer)", lat: 33.7555, lng: -84.3845, status: "calm" },
];

export const LINKS: MeshLink[] = [
  { from: "bus", to: "driver", kind: "mesh" },
  { from: "bus", to: "rider", kind: "mesh" },
  { from: "driver", to: "rider", kind: "mesh" },
  { from: "bus", to: "peer", kind: "mesh" },
  { from: "bus", to: "security", kind: "cellular" },
  { from: "driver", to: "security", kind: "cellular" },
];

export interface LogLine {
  t: number; // ms since sequence start
  text: string;
}

export interface SequenceStep {
  delayMs: number;
  run: () => void;
  log?: string;
}

/**
 * Builds the triage -> mesh relay -> Photon dispatch -> rider-confirm sequence.
 * Triggered once a real detection score crosses threshold (see detect.ts) —
 * this function no longer invents the score itself.
 */
export function buildEscalationSequence(opts: {
  score: number;
  onLog: (line: string) => void;
  onNodeStatus: (id: string, status: NodeStatus) => void;
  onLinkPulse: (from: string, to: string) => void;
  onAwaitConfirm: () => void;
}): SequenceStep[] {
  const { score, onLog, onNodeStatus, onLinkPulse, onAwaitConfirm } = opts;
  const steps: SequenceStep[] = [];

  steps.push({
    delayMs: 0,
    run: () => {
      onNodeStatus("bus", "alert");
      onLog(`Threshold crossed (score ${score.toFixed(2)} ≥ 0.75) — triage agent escalating.`);
    },
  });

  steps.push({
    delayMs: 700,
    run: () => {
      onLinkPulse("bus", "driver");
      onLinkPulse("bus", "rider");
      onLinkPulse("bus", "peer");
      onNodeStatus("driver", "alert");
      onLog("Mesh relay (Ditto, local): driver + rider devices notified — 0 internet required.");
    },
  });

  steps.push({
    delayMs: 900,
    run: () => {
      onLinkPulse("bus", "security");
      onLinkPulse("driver", "security");
      onNodeStatus("security", "watching");
      onLog("Photon dispatch (cellular/SMS): security alerted with location + confidence.");
    },
  });

  steps.push({
    delayMs: 900,
    run: () => {
      onNodeStatus("rider", "watching");
      onLog('Rider device prompted: "Everything OK?" — silent confirm/dismiss.');
      onAwaitConfirm();
    },
  });

  return steps;
}

export function runSequence(steps: SequenceStep[]): { cancel: () => void } {
  let cancelled = false;
  let elapsed = 0;
  const timers: number[] = [];
  for (const step of steps) {
    elapsed += step.delayMs;
    const id = window.setTimeout(() => {
      if (!cancelled) step.run();
    }, elapsed);
    timers.push(id);
  }
  return {
    cancel: () => {
      cancelled = true;
      timers.forEach((id) => clearTimeout(id));
    },
  };
}
