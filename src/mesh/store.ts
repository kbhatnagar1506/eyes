// Replicated document store. Data is collections of documents of fields, and
// every field carries the clock stamp of the write that set it. Merging is
// per field, last writer wins by stamp, so two devices editing different
// fields of one incident (driver acks, police ack) never clobber each other,
// and applying the same ops in any order on any device gives the same state.

export type Op = { c: string; id: string; f: string; v: unknown; t: string };
type Cell = { v: unknown; t: string };
export type Doc = Record<string, unknown> & { id: string };

export class Store {
  private docs = new Map<string, Map<string, Map<string, Cell>>>();
  private listeners = new Set<(c: string, id: string, op: Op) => void>();

  // Returns true if the op changed state (newer than what we had).
  apply(op: Op): boolean {
    let coll = this.docs.get(op.c);
    if (!coll) this.docs.set(op.c, (coll = new Map()));
    let doc = coll.get(op.id);
    if (!doc) coll.set(op.id, (doc = new Map()));
    const cur = doc.get(op.f);
    if (cur && cur.t >= op.t) return false;
    doc.set(op.f, { v: op.v, t: op.t });
    for (const l of this.listeners) l(op.c, op.id, op);
    return true;
  }

  get(c: string, id: string): Doc | undefined {
    const doc = this.docs.get(c)?.get(id);
    if (!doc) return undefined;
    const out: Doc = { id };
    for (const [f, cell] of doc) out[f] = cell.v;
    return out;
  }

  list(c: string): Doc[] {
    const coll = this.docs.get(c);
    if (!coll) return [];
    return [...coll.keys()].map((id) => this.get(c, id)!);
  }

  // Every op that produces the current state; sent to a device on first contact.
  snapshot(): Op[] {
    const ops: Op[] = [];
    for (const [c, coll] of this.docs) for (const [id, doc] of coll) for (const [f, cell] of doc) ops.push({ c, id, f, v: cell.v, t: cell.t });
    return ops;
  }

  onChange(cb: (c: string, id: string, op: Op) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
}
