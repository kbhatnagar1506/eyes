// Hybrid logical clock: wall time + counter + node id, encoded so that plain
// string comparison orders events. Every device stamps its own writes and all
// devices agree on which write wins, without a central server, even when
// their clocks disagree a little.

const W = 9; // base-36 digits for ms since epoch (good past year 5000)
const C = 4; // base-36 digits for the same-millisecond counter

export class HLC {
  private wall = 0;
  private counter = 0;
  private readonly nodeId: string;

  constructor(nodeId: string) {
    this.nodeId = nodeId;
  }

  now(): string {
    const t = Date.now();
    if (t > this.wall) {
      this.wall = t;
      this.counter = 0;
    } else {
      this.counter++;
    }
    return this.encode();
  }

  // Merge a stamp seen from another device so our next stamp sorts after it.
  observe(stamp: string) {
    const [w, c] = parseStamp(stamp);
    const t = Date.now();
    const next = Math.max(t, this.wall, w);
    if (next === this.wall && next === w) this.counter = Math.max(this.counter, c) + 1;
    else if (next === this.wall) this.counter++;
    else if (next === w) this.counter = c + 1;
    else this.counter = 0;
    this.wall = next;
  }

  private encode() {
    return `${this.wall.toString(36).padStart(W, "0")}:${this.counter.toString(36).padStart(C, "0")}:${this.nodeId}`;
  }
}

export function parseStamp(stamp: string): [number, number] {
  const [w, c] = stamp.split(":");
  return [parseInt(w, 36), parseInt(c, 36)];
}

export function stampTime(stamp: string): number {
  return parseStamp(stamp)[0];
}
