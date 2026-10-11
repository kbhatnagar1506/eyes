// Local-dev only (imported behind import.meta.env.DEV in main.ts): guided
// recordings from the Live tab's recorder and hand-labeled clips, dropped
// into sessions/ (gitignored, never deployed).

export type SessionLabels = {
  name: string;
  video: string;
  durationS: number;
  // Hand-labeled sessions mark the moment of contact, so they use a tighter
  // window than the recorder's countdown cues.
  window?: { before: number; after: number };
  cues: { action: string; kind: "strike" | "harmless" | "idle"; t: number }[];
  liveStrikes: { t: number; desc: string }[];
};

const sessionLabels = import.meta.glob<SessionLabels>("/sessions/*.json", { eager: true, import: "default" });
const sessionVideos = import.meta.glob<string>("/sessions/*.{mp4,webm,mov}", { eager: true, query: "?url", import: "default" });

export const SESSION_CLIPS = Object.values(sessionLabels).flatMap((labels) => {
  const url = sessionVideos[`/sessions/${labels.video}`];
  return url ? [{ id: labels.name, label: `Session: ${labels.name.replace("eyes-session-", "")}`, file: url, labels }] : [];
});
