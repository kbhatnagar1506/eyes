import { defineConfig, type Plugin } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { execSync } from "node:child_process";

const PROJECT_ID = "patchguard-reakon";
const LOCATION = "us-central1";
const MODEL = "gemini-2.5-flash";

let cachedToken: { value: string; expiresAt: number } | null = null;

function getAccessToken(): string {
  const now = Date.now();
  if (cachedToken && cachedToken.expiresAt > now) return cachedToken.value;
  const token = execSync("gcloud auth print-access-token", { encoding: "utf8" }).trim();
  cachedToken = { value: token, expiresAt: now + 45 * 60 * 1000 }; // refresh every ~45 min
  return token;
}

function readBody(req: import("node:http").IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

// Dev-only Gemini (Vertex AI) classification proxy. Keeps the GCP service
// account credentials server-side — the browser only ever talks to this
// local endpoint, never holds a token itself.
function geminiProxyPlugin(): Plugin {
  return {
    name: "gemini-classify-proxy",
    configureServer(server) {
      server.middlewares.use("/api/classify", async (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("POST only");
          return;
        }
        try {
          const body = await readBody(req);
          const { imageBase64, imagesBase64, videoBase64, mimeType, prompt } = JSON.parse(body);
          const parts: Array<Record<string, unknown>> = [{ text: prompt }];
          if (videoBase64) {
            parts.push({ inline_data: { mime_type: mimeType || "video/webm", data: videoBase64 } });
          } else {
            const images: string[] = imagesBase64 ?? (imageBase64 ? [imageBase64] : []);
            for (const data of images) parts.push({ inline_data: { mime_type: "image/jpeg", data } });
          }
          const token = getAccessToken();
          const url = `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL}:generateContent`;
          const resp = await fetch(url, {
            method: "POST",
            headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
            body: JSON.stringify({
              contents: [{ role: "user", parts }],
              // Gemini 2.5 Flash spends tokens "thinking" before answering by
              // default, which ate the whole maxOutputTokens budget and
              // truncated the response. This is a fast yes/no classification,
              // not a reasoning task — thinkingBudget: 0 disables it.
              generationConfig: {
                temperature: 0,
                maxOutputTokens: 200,
                thinkingConfig: { thinkingBudget: 0 },
              },
            }),
          });
          const data = await resp.json();
          const text = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ text, raw: resp.ok ? undefined : data }));
        } catch (err) {
          res.statusCode = 500;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: String(err) }));
        }
      });
    },
  };
}

// Phones need https: browsers only allow geolocation (and the camera) on a
// secure origin, and http://<laptop-ip> isn't one. EYES_HTTPS=1 serves a
// self-signed cert (accept the warning once per device). Plain localhost dev
// stays on http.
const HTTPS = process.env.EYES_HTTPS === "1";
const HUB = "127.0.0.1:8787";

export default defineConfig({
  plugins: [geminiProxyPlugin(), ...(HTTPS ? [basicSsl()] : [])],
  server: {
    host: true, // reachable from phones on the same network
    proxy: {
      // The mesh hub rides on this origin, so an https page gets a wss mesh.
      // xfwd passes each device's real address through, which the hub needs
      // to tell "same machine" links from network ones.
      "/mesh": { target: `ws://${HUB}`, ws: true, xfwd: true },
      "/hub": { target: `http://${HUB}`, xfwd: true, rewrite: (p) => p.replace(/^\/hub/, "") },
    },
  },
});
