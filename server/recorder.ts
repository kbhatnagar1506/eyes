// Keeps a durable record of everything that happens on the mesh: incidents,
// who acknowledged, the agents' conversation in Postgres (Cloud SQL eyes-db),
// and every evidence frame as a JPEG in the private media bucket with a row
// pointing to it. Runs inside the agent service, which is always on the mesh.
//
// Writes are idempotent upserts keyed by the mesh document id, so the same
// data arriving again (a device rejoining, a snapshot replay) changes nothing.

import { SQL } from "bun";
import type { MeshNode } from "../src/mesh/node";
import { safeJpeg } from "../src/mesh/frames";
import { uploadObject } from "./gcp";

const SCHEMA = `
create table if not exists incidents (
  id          text primary key,
  kind        text not null,
  status      text,
  label       text,
  detail      text,
  evidence    text,
  lat         double precision,
  lng         double precision,
  camera      text,
  punches     int,
  detected_at timestamptz not null,
  resolved_at timestamptz,
  doc         jsonb not null,
  updated_at  timestamptz not null default now()
);
create table if not exists acknowledgements (
  incident_id text not null,
  who         text not null,
  status      text,
  at          timestamptz not null,
  primary key (incident_id, who)
);
create table if not exists agent_messages (
  id          text primary key,
  incident_id text not null,
  seq         int not null,
  agent       text not null,
  emoji       text,
  text        text not null,
  at          timestamptz not null
);
create index if not exists agent_messages_by_incident on agent_messages (incident_id, seq);
create table if not exists evidence_frames (
  id          text primary key,
  incident_id text not null,
  seq         int not null,
  caption     text,
  gcs_uri     text not null,
  bytes       int,
  captured_at timestamptz not null,
  stored_at   timestamptz not null default now()
);
create index if not exists evidence_frames_by_incident on evidence_frames (incident_id, seq);
`;

// Many fields of one doc land in quick succession: write each doc once they settle.
const SETTLE_MS = 300;

const ts = (v: unknown) => (v === undefined || v === null || v === "" ? null : new Date(Number(v)));
const num = (v: unknown) => (v === undefined || v === null || v === "" || Number.isNaN(Number(v)) ? null : Number(v));
const str = (v: unknown) => (v === undefined || v === null ? null : String(v));

export async function startRecorder(node: MeshNode, databaseUrl: string, log: (...a: unknown[]) => void) {
  const sql = new SQL(databaseUrl);
  await sql.unsafe(SCHEMA);
  log("recorder: writing incidents, acknowledgements and agent messages to Postgres, frames to the media bucket");

  const storedFrames = new Set<string>();
  const pending = new Map<string, ReturnType<typeof setTimeout>>();

  async function writeIncident(id: string) {
    const d = node.store.get("incidents", id);
    if (!d || d.at === undefined) return;
    const { id: _id, ...doc } = d;
    void _id;
    await sql`
      insert into incidents (id, kind, status, label, detail, evidence, lat, lng, camera, punches, detected_at, resolved_at, doc, updated_at)
      values (${id}, ${str(d.kind) ?? "incident"}, ${str(d.status)}, ${str(d.label)}, ${str(d.detail)}, ${str(d.evidence)},
              ${num(d.lat)}, ${num(d.lng)}, ${str(d.camera)}, ${num(d.punches)}, ${ts(d.at)}, ${ts(d.resolvedAt)}, ${JSON.stringify(doc)}::jsonb, now())
      on conflict (id) do update set
        kind = excluded.kind, status = excluded.status, label = excluded.label, detail = excluded.detail,
        evidence = excluded.evidence, lat = excluded.lat, lng = excluded.lng, camera = excluded.camera,
        punches = excluded.punches, resolved_at = excluded.resolved_at, doc = excluded.doc, updated_at = now()`;
    for (const [k, v] of Object.entries(d)) {
      if (!k.startsWith("ack:")) continue;
      const who = k.slice(4);
      await sql`
        insert into acknowledgements (incident_id, who, status, at)
        values (${id}, ${who}, ${str(d[`status:${who}`])}, ${ts(v)})
        on conflict (incident_id, who) do update set status = excluded.status, at = excluded.at`;
    }
  }

  async function writeAgentMessage(id: string) {
    const d = node.store.get("agentchat", id);
    if (!d || d.text === undefined || d.incident === undefined) return;
    await sql`
      insert into agent_messages (id, incident_id, seq, agent, emoji, text, at)
      values (${id}, ${String(d.incident)}, ${num(d.seq) ?? 0}, ${String(d.agent ?? "")}, ${str(d.emoji)}, ${String(d.text)}, ${ts(d.at) ?? new Date()})
      on conflict (id) do nothing`;
  }

  async function writeFrame(id: string) {
    if (storedFrames.has(id)) return;
    const d = node.store.get("frames", id);
    const src = safeJpeg(d?.data);
    if (!d || !src) return;
    storedFrames.add(id);
    const incident = String(d.incident ?? id);
    const seq = num(d.seq) ?? 1;
    const exists = await sql`select 1 from evidence_frames where id = ${id}`;
    if (exists.length) return;
    const bytes = Buffer.from(src.slice(src.indexOf(",") + 1), "base64");
    const name = `evidence/${incident}/${String(seq).padStart(3, "0")}.jpg`;
    const uri = await uploadObject(name, new Blob([bytes], { type: "image/jpeg" }), "image/jpeg");
    await sql`
      insert into evidence_frames (id, incident_id, seq, caption, gcs_uri, bytes, captured_at)
      values (${id}, ${incident}, ${seq}, ${str(d.caption)}, ${uri}, ${bytes.length}, ${ts(d.at) ?? new Date()})
      on conflict (id) do nothing`;
  }

  const writers: Record<string, (id: string) => Promise<void>> = { incidents: writeIncident, agentchat: writeAgentMessage, frames: writeFrame };

  function schedule(c: string, id: string) {
    const write = writers[c];
    if (!write) return;
    const key = `${c}/${id}`;
    clearTimeout(pending.get(key));
    pending.set(
      key,
      setTimeout(() => {
        pending.delete(key);
        write(id).catch((err) => {
          if (c === "frames") storedFrames.delete(id); // retry on the next change
          log(`recorder: ${key} not saved: ${String(err).slice(0, 160)}`);
        });
      }, SETTLE_MS),
    );
  }

  node.store.onChange((c, id) => schedule(c, id));
  // Whatever the mesh already knew before we started.
  for (const c of Object.keys(writers)) for (const d of node.store.list(c)) schedule(c, d.id);
}
