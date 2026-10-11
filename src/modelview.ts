// The Model tab: Eyes' main model (AttackNet), how it was trained and how
// well it does, the data, and the infrastructure around it. Training numbers
// come straight from the GPU run's report (scripts/attack/train_rlvs.py ->
// report.json); every score is on clips the model never trained on.

import {
  ArcElement,
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  DoughnutController,
  Filler,
  Legend,
  LinearScale,
  LogarithmicScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
  type ChartConfiguration,
  type ScriptableContext,
} from "chart.js";
import { attackUrl } from "./mesh/media";
import { logoSvg } from "./logo";

Chart.register(ArcElement, BarController, BarElement, CategoryScale, DoughnutController, Filler, Legend, LinearScale, LogarithmicScale, LineController, LineElement, PointElement, Tooltip);
Chart.defaults.color = "#8b93a7";
Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
Chart.defaults.font.family = '"Inter", system-ui, -apple-system, sans-serif';
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = "circle";
Chart.defaults.plugins.legend.labels.boxWidth = 7;
Chart.defaults.plugins.legend.labels.boxHeight = 7;
Chart.defaults.plugins.tooltip.backgroundColor = "rgba(10,13,19,0.95)";
Chart.defaults.plugins.tooltip.borderColor = "rgba(255,255,255,0.12)";
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.titleColor = "#e7e9ee";
Chart.defaults.maintainAspectRatio = false;

const C = { fight: "#f43f5e", normal: "#2dd4bf", accent: "#fbbf24", blue: "#60a5fa", violet: "#a78bfa", pink: "#f472b6", dim: "#4b5466" };

type Pt = [number, number];
type Confusion = { tp: number; fp: number; fn: number; tn: number };
type ClipM = { f1: number; precision: number; recall: number; accuracy: number; auc: number | null; frames: number };
type Stage = { threshold: number; pooled: ClipM; per_clip: Record<string, ClipM> };
interface Report {
  data: {
    rlvs: {
      clips: Record<"train" | "val" | "test", { fight: number; normal: number }>;
      frames: number;
      tracks: number;
      seconds_hist: Record<"0" | "1", number[]>;
      people_hist: Record<"0" | "1", number[]>;
      resolution: Record<string, number>;
    };
    ours: { clip: string; split: string; frames: number; attack_frames: number; people_per_frame: number }[];
  };
  training: { pretrain: { epoch: number; loss: number; val_auc?: number; val_accuracy?: number }[]; finetune: Record<string, { epoch: number; loss: number }[]> };
  rlvs_test: { roc: Pt[]; pr: Pt[]; confusion: Confusion; threshold: number; hist: { fight: number[]; normal: number[] } };
  importance: { base_auc: number; groups: { group: string; auc_drop: number }[] };
  ours: Record<"A" | "B", Record<"ours" | "sparring", { roc: Pt[]; confusion: Confusion }>>;
  params?: Record<string, number>;
  summary: {
    weights: number;
    features: number;
    shown: "A" | "B";
    threshold: number;
    rlvs: { clips: number; train: number; test: number; stage_A_test: { clip_accuracy: number; clip_auc: number; clips: number } };
    stage_A: { ours: Stage; sparring: Stage };
    stage_B: { ours: Stage; sparring: Stage };
    v1_ours_pooled: Partial<ClipM>;
    seconds: number;
  };
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
const num = (v: number | null | undefined, d = 2) => (v === null || v === undefined ? "–" : v.toFixed(d));
// Display names for our clips (file names are internal).
const CLIP_NAME: Record<string, string> = {
  "gf-punch-1": "Demo camera",
  busstop_calm_36467681: "Bus stop, calm",
  Normal_929_x264: "CCTV, normal",
  Fighting042_x264: "CCTV fight 1",
  Fighting049_x264: "CCTV fight 2",
  Assault049_x264: "CCTV assault 1",
  Assault050_x264: "CCTV assault 2",
  Assault051_x264: "CCTV assault 3",
  sparring_6296158: "Sparring 1",
  sparring2_6296587: "Sparring 2",
  boxing_6296696: "Boxing",
  woman_sparring_6296385: "Sparring 3",
};
const short = (clip: string) => CLIP_NAME[clip] ?? clip.replace(/_x264$/, "").replace(/_\d{7,}$/, "").replace(/_/g, " ");

async function loadReport(): Promise<Report | null> {
  const urls = [await attackUrl("report.json")];
  // Local dev: the report straight from disk (Vite serves the project root).
  if (import.meta.env.DEV) urls.push("/data/attack/out/report.json", "/data/attack/out/report.dev.json");
  for (const u of urls) {
    if (!u) continue;
    try {
      const r = await fetch(u, { cache: "no-store" });
      if (r.ok) return (await r.json()) as Report;
    } catch {
      /* next */
    }
  }
  return null;
}

// Vertical gradient under a line, from its color to transparent.
const fade = (hex: string, a = 0.28) => (ctx: ScriptableContext<"line">) => {
  const { chart } = ctx;
  const area = chart.chartArea;
  if (!area) return "transparent";
  const g = chart.ctx.createLinearGradient(0, area.top, 0, area.bottom);
  const [r, gg, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  g.addColorStop(0, `rgba(${r},${gg},${b},${a})`);
  g.addColorStop(1, `rgba(${r},${gg},${b},0)`);
  return g;
};
const rgba = (hex: string, a: number) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${a})`;
};

const reduceMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export class ModelView {
  private charts: Chart[] = [];
  private loaded = false;
  private root: HTMLElement;
  private revealWired = false;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  async show() {
    if (this.loaded) return;
    this.loaded = true;
    // Start right under the app header, which wraps to two rows on phones.
    const fit = () => {
      const bar = document.querySelector("#tabModel")?.closest("#app > *");
      if (bar) this.root.style.top = `${Math.round(bar.getBoundingClientRect().bottom)}px`;
    };
    fit();
    window.addEventListener("resize", fit);
    this.root.innerHTML = staticHtml();
    this.paramsChart(PARAMS);
    this.wireNav();
    this.reveal();
    this.countUp(this.root.querySelector<HTMLElement>("#mdWeights"), Object.values(PARAMS).reduce((a, b) => a + b, 0));
    const r = await loadReport();
    const slot = this.root.querySelector<HTMLElement>("#mdReport")!;
    if (!r) {
      slot.innerHTML = `<div class="md-card md-empty"><span class="md-live"></span>AttackNet is training on the Google Cloud NVIDIA T4 right now. Training curves and results land here when the run finishes.</div>`;
      return;
    }
    if (r.params) {
      for (const [k, v] of Object.entries(r.params)) this.root.querySelectorAll(`[data-param="${k}"]`).forEach((el) => (el.textContent = v.toLocaleString()));
    }
    this.render(r, slot);
    this.reveal();
  }

  // ---- page behaviour --------------------------------------------------------

  private wireNav() {
    const nav = this.root.querySelector<HTMLElement>(".md-nav")!;
    nav.addEventListener("click", (e) => {
      const a = (e.target as HTMLElement).closest<HTMLAnchorElement>("a[data-to]");
      if (!a) return;
      e.preventDefault();
      this.root.querySelector(`#${a.dataset.to}`)?.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
    });
    // Highlight the section being read: the last header scrolled past.
    let queued = false;
    const spy = () => {
      queued = false;
      const links = [...nav.querySelectorAll<HTMLAnchorElement>("a[data-to]")];
      const line = this.root.getBoundingClientRect().top + 140;
      let on: string | null = null;
      for (const a of links) {
        const t = this.root.querySelector(`#${a.dataset.to}`);
        if (t && t.getBoundingClientRect().top <= line) on = a.dataset.to!;
      }
      for (const a of links) a.classList.toggle("on", a.dataset.to === on);
      nav.querySelector("a.on")?.scrollIntoView({ block: "nearest", inline: "nearest" });
    };
    this.root.addEventListener("scroll", () => {
      if (!queued) {
        queued = true;
        setTimeout(spy, 60);
      }
    }, { passive: true });
  }

  // Fade sections in as they scroll into view. A plain position check on
  // scroll (not an observer), so nothing can be left invisible.
  private reveal() {
    const pending = () => [...this.root.querySelectorAll<HTMLElement>(".md-reveal:not(.md-in)")];
    if (reduceMotion()) {
      pending().forEach((el) => el.classList.add("md-in"));
      return;
    }
    const check = () => {
      const bottom = this.root.getBoundingClientRect().bottom + 40;
      for (const el of pending()) if (el.getBoundingClientRect().top < bottom) el.classList.add("md-in");
    };
    check();
    setTimeout(check, 50);
    if (!this.revealWired) {
      this.revealWired = true;
      this.root.addEventListener("scroll", check, { passive: true });
      window.addEventListener("resize", check);
    }
  }

  private countUp(el: HTMLElement | null, to: number) {
    if (!el) return;
    if (reduceMotion()) {
      el.textContent = to.toLocaleString();
      return;
    }
    const start = performance.now();
    const ms = 1600;
    const tick = (now: number) => {
      const k = Math.min(1, (now - start) / ms);
      el.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))).toLocaleString();
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  private chart(id: string, cfg: ChartConfiguration) {
    const el = this.root.querySelector<HTMLCanvasElement>(`#${id}`);
    if (el) this.charts.push(new Chart(el, cfg));
  }

  private paramsChart(p: Record<string, number>) {
    const names: Record<string, string> = { inp: "Input projection", temporal: "Temporal transformer", people: "People transformer", head: "Head", "pos + cls tokens": "Position + CLS tokens" };
    this.chart("mdParams", {
      type: "doughnut",
      data: { labels: Object.keys(p).map((k) => names[k] ?? k), datasets: [{ data: Object.values(p), backgroundColor: [C.blue, C.normal, C.violet, C.accent, C.fight], borderWidth: 0, hoverOffset: 6 }] },
      options: {
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${Number(c.raw).toLocaleString()} weights` } } },
      } as ChartConfiguration<"doughnut">["options"],
    } as ChartConfiguration);
  }

  // ---- the training report ------------------------------------------------------

  private render(r: Report, slot: HTMLElement) {
    const s = r.summary;
    const shown = s.shown === "B" ? s.stage_B : s.stage_A;
    const v1 = s.v1_ours_pooled ?? {};
    const rl = r.data.rlvs;
    const total = (k: "train" | "val" | "test") => rl.clips[k].fight + rl.clips[k].normal;

    const ring = (value: number | null | undefined, label: string, sub: string, color: string, delta?: number | null, asPct = true) => {
      const v = value ?? 0;
      const C2 = 2 * Math.PI * 30;
      const d = delta === null || delta === undefined || Number.isNaN(delta) ? "" : `<span class="md-delta ${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "▲" : "▼"} ${asPct ? `${Math.abs(Math.round(delta * 100))} pts` : Math.abs(delta).toFixed(2)} vs v1</span>`;
      return `<div class="md-kpi md-reveal">
        <svg viewBox="0 0 72 72" class="md-ring"><circle cx="36" cy="36" r="30" class="md-ring-bg"/><circle cx="36" cy="36" r="30" class="md-ring-fg" style="stroke:${color};stroke-dasharray:${(v * C2).toFixed(1)} ${C2.toFixed(1)}"/></svg>
        <div><div class="md-k">${asPct ? pct(value) : num(value)}</div><div class="md-kl">${label}</div><div class="md-ks">${sub}</div>${d}</div>
      </div>`;
    };
    const confusionHtml = (title: string, c: Confusion, unit: string) => {
      const all = c.tp + c.fp + c.fn + c.tn || 1;
      const cell = (v: number, good: boolean) => `<b class="${good ? "md-good" : "md-bad"}" style="--a:${(0.12 + 0.5 * (v / all)).toFixed(2)}">${v}<i>${Math.round((100 * v) / all)}%</i></b>`;
      return `<div class="md-conf">
        <div class="md-sub">${title}</div>
        <div class="md-conf-grid">
          <span></span><span class="md-dim">Said attack</span><span class="md-dim">Said calm</span>
          <span class="md-dim">Attack</span>${cell(c.tp, true)}${cell(c.fn, false)}
          <span class="md-dim">Calm</span>${cell(c.fp, false)}${cell(c.tn, true)}
        </div>
        <div class="md-dim">${all.toLocaleString()} ${unit}</div>
      </div>`;
    };
    const fineFolds = Object.keys(r.training.finetune);
    const clipRows = r.data.ours
      .map((o) => {
        const m = shown.ours.per_clip[o.clip] ?? shown.sparring.per_clip[o.clip];
        const calm = o.attack_frames === 0;
        const score = calm ? (m ? 1 - m.accuracy : 0) : (m?.f1 ?? 0);
        return `<tr>
          <td><b>${esc(short(o.clip))}</b></td>
          <td><span class="md-pill ${o.split === "train" ? "" : "md-pill-hold"}">${o.split === "train" ? "CCTV / demo · held out" : "never trained on"}</span></td>
          <td class="md-num">${o.frames}</td><td class="md-num">${o.attack_frames}</td><td class="md-num">${o.people_per_frame}</td>
          <td><div class="md-bar"><i style="width:${Math.round(score * 100)}%;background:${calm ? C.accent : C.normal}"></i></div><span class="md-num">${calm ? `${pct(score)} flagged` : `F1 ${pct(score)}`}</span></td>
          <td class="md-num">${num(m?.auc)}</td></tr>`;
      })
      .join("");

    slot.innerHTML = `
      ${sec("mdResultsTop", "03", "Headline results", "Every number here is on clips the model never trained on.")}
      <section class="md-kpis">
        ${ring(s.rlvs.stage_A_test.clip_accuracy, "right on unseen clips", `${s.rlvs.stage_A_test.clips} fight / normal clips held back`, C.normal)}
        ${ring(s.rlvs.stage_A_test.clip_auc, "AUC on those clips", "0.5 is a coin flip, 1.0 is perfect", C.blue, null, false)}
        ${ring(shown.ours.pooled.f1, "attack F1 on our CCTV", "each clip scored by a model that never saw it", C.fight, v1.f1 !== undefined ? shown.ours.pooled.f1 - v1.f1 : null)}
        ${ring(shown.ours.pooled.auc, "AUC on our CCTV", "frame by frame, hand-labeled", C.violet, v1.auc !== undefined && shown.ours.pooled.auc !== null ? shown.ours.pooled.auc - (v1.auc ?? 0) : null, false)}
        ${ring(shown.sparring.pooled.f1, "F1 on sparring", "clips kept out of training entirely", C.accent)}
        <div class="md-kpi md-kpi-big md-reveal"><div><div class="md-k">${(rl.frames / 1000).toFixed(0)}k</div><div class="md-kl">frames analysed</div><div class="md-ks">${rl.tracks.toLocaleString()} people tracked · ${rl.clips.train.fight + rl.clips.train.normal + total("val") + total("test")} clips</div></div></div>
      </section>

      ${sec("mdTraining", "04", "Training", "Pretrained on 1,400 real-world clips on a Google Cloud NVIDIA T4, then tested against adapting it to our CCTV.")}
      <section class="md-grid">
        <div class="md-card md-wide md-reveal"><div class="md-title">Pretraining on RLVS <span class="md-dim">· loss falls while held-out accuracy climbs</span></div><div class="md-canvas md-canvas-tall"><canvas id="mdLoss"></canvas></div></div>
        <div class="md-card md-reveal"><div class="md-title">Stage B: fine-tuning on our clips <span class="md-dim">· one run per held-out clip</span></div><div class="md-canvas"><canvas id="mdFine"></canvas></div>
          <div class="md-dim">${fineFolds.length} runs; each leaves one clip out, then is scored on it. ${s.shown === "A" ? "Not shipped: it scored worse on unseen clips than the pretrained model, so the pretrained model is what runs." : "Shipped: it beat the pretrained model on unseen clips."}</div></div>
        <div class="md-card md-reveal"><div class="md-title">Where the weights are</div><div class="md-canvas"><canvas id="mdParams2"></canvas></div>
          <div class="md-dim">AdamW, one-cycle learning rate, mirror / scale / noise / dropped-frame augmentation. ${Math.round(s.seconds / 60)} min on the T4.</div></div>
      </section>

      ${sec("mdResults", "05", "Results on unseen clips", "How often it's right, how sure it is, and where it gets it wrong.")}
      <section class="md-grid">
        <div class="md-card md-reveal"><div class="md-title">ROC curves</div><div class="md-canvas"><canvas id="mdRoc"></canvas></div></div>
        <div class="md-card md-reveal"><div class="md-title">Precision vs recall <span class="md-dim">· RLVS test clips</span></div><div class="md-canvas"><canvas id="mdPr"></canvas></div></div>
        <div class="md-card md-reveal"><div class="md-title">Confidence: fight vs normal clips</div><div class="md-canvas"><canvas id="mdHist"></canvas></div>
          <div class="md-dim">Dashed line: the alert threshold (${num(r.rlvs_test.threshold)}).</div></div>
        <div class="md-card md-confs md-reveal">
          ${confusionHtml("RLVS test clips", r.rlvs_test.confusion, "clips")}
          ${confusionHtml("Our CCTV, held out", r.ours[s.shown].ours.confusion, "frames")}
        </div>
      </section>

      ${sec("mdBody", "06", "What body language it relies on", "Scramble one signal at a time and watch the accuracy fall.")}
      <section class="md-grid">
        <div class="md-card md-wide md-reveal"><div class="md-canvas md-canvas-mid"><canvas id="mdImp"></canvas></div>
          <div class="md-dim">Permutation importance on validation clips: each group of features is shuffled across moments, then the clip AUC is measured again (base ${num(r.importance.base_auc)}). The bigger the drop, the more the model needs that signal.</div></div>
      </section>

      ${sec("mdData", "07", "Data", "What it learned from.")}
      <section class="md-grid">
        <div class="md-card md-reveal"><div class="md-title">RLVS split, by clip</div><div class="md-canvas"><canvas id="mdSplit"></canvas></div>
          <div class="md-dim">${rl.clips.train.fight + rl.clips.train.normal} train · ${total("val")} validation (sets thresholds) · ${total("test")} test (never trained on)</div></div>
        <div class="md-card md-reveal"><div class="md-title">Clip length</div><div class="md-canvas"><canvas id="mdLen"></canvas></div></div>
        <div class="md-card md-reveal"><div class="md-title">People per frame</div><div class="md-canvas"><canvas id="mdPeople"></canvas></div></div>
        <div class="md-card md-reveal"><div class="md-title">Video resolution</div><div class="md-canvas"><canvas id="mdRes"></canvas></div></div>
      </section>
      <div class="md-card md-reveal md-gap">
        <div class="md-title">Our clips <span class="md-dim">· labeled frame by frame, each scored by a model that never saw it</span></div>
        <div class="md-table-wrap"><table class="md-table"><thead><tr><th>Clip</th><th>Role</th><th>Frames</th><th>Attack frames</th><th>People / frame</th><th>Result</th><th>AUC</th></tr></thead><tbody>${clipRows}</tbody></table></div>
      </div>`;

    const pre = r.training.pretrain;
    this.chart("mdLoss", {
      type: "line",
      data: {
        labels: pre.map((p) => p.epoch),
        datasets: [
          { label: "Training loss", data: pre.map((p) => p.loss), borderColor: C.accent, backgroundColor: fade(C.accent), fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2.5, yAxisID: "y" },
          { label: "Held-out AUC", data: pre.map((p) => p.val_auc ?? null), borderColor: C.normal, spanGaps: true, tension: 0.35, pointRadius: 3.5, pointBackgroundColor: C.normal, borderWidth: 2.5, yAxisID: "y1" },
          { label: "Held-out accuracy", data: pre.map((p) => p.val_accuracy ?? null), borderColor: C.blue, spanGaps: true, tension: 0.35, pointRadius: 3.5, pointBackgroundColor: C.blue, borderWidth: 2.5, yAxisID: "y1" },
        ],
      },
      options: {
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "top", align: "end" } },
        scales: {
          x: { grid: { display: false }, title: { display: true, text: "epoch" } },
          // Log scale: one unstable epoch spiked; the curve stays readable around it.
          y: { type: "logarithmic" as never, title: { display: true, text: "loss (log)" } },
          y1: { position: "right", min: 0.4, max: 1, grid: { drawOnChartArea: false }, title: { display: true, text: "held-out" } },
        },
      },
    });
    const palette = [C.fight, C.normal, C.accent, C.blue, C.violet, C.pink, "#34d399", "#fb923c"];
    this.chart("mdFine", {
      type: "line",
      data: {
        labels: (r.training.finetune[fineFolds[0]] ?? []).map((p) => p.epoch),
        datasets: fineFolds.map((f, i) => ({ label: short(f), data: r.training.finetune[f].map((p) => p.loss), borderColor: palette[i % palette.length], pointRadius: 0, tension: 0.35, borderWidth: 1.8 })),
      },
      options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { x: { grid: { display: false }, title: { display: true, text: "epoch" } }, y: { title: { display: true, text: "loss" } } } },
    });
    const p = r.params ?? PARAMS;
    const pnames: Record<string, string> = { inp: "Input projection", temporal: "Temporal transformer", people: "People transformer", head: "Head", "pos + cls tokens": "Tokens" };
    this.chart("mdParams2", {
      type: "bar",
      data: { labels: Object.keys(p).map((k) => pnames[k] ?? k), datasets: [{ data: Object.values(p), backgroundColor: [C.blue, C.normal, C.violet, C.accent, C.fight], borderRadius: 6 }] },
      options: { maintainAspectRatio: false, indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${Number(c.raw).toLocaleString()} weights` } } }, scales: { x: { type: "logarithmic" as never, title: { display: true, text: "weights (log)" } }, y: { grid: { display: false } } } },
    });
    const xy = (pts: Pt[]) => pts.map(([x, y]) => ({ x, y }));
    const line = (label: string, pts: Pt[], color: string, dash?: number[]) => ({ label, data: xy(pts), borderColor: color, borderDash: dash, pointRadius: 0, tension: 0, borderWidth: 2.5, showLine: true });
    this.chart("mdRoc", {
      type: "line",
      data: {
        datasets: [
          { ...line(`RLVS test · ${num(s.rlvs.stage_A_test.clip_auc)}`, r.rlvs_test.roc, C.normal), fill: true, backgroundColor: fade(C.normal, 0.18) },
          line(`Our CCTV · ${num(shown.ours.pooled.auc)}`, r.ours[s.shown].ours.roc, C.fight),
          line(`Sparring · ${num(shown.sparring.pooled.auc)}`, r.ours[s.shown].sparring.roc, C.accent),
          { ...line("Coin flip", [[0, 0], [1, 1]], C.dim, [4, 4]), borderWidth: 1.5 },
        ],
      },
      options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { x: { type: "linear", min: 0, max: 1, grid: { display: false }, title: { display: true, text: "false alarm rate" } }, y: { min: 0, max: 1, title: { display: true, text: "attacks caught" } } } },
    });
    this.chart("mdPr", {
      type: "line",
      data: { datasets: [{ ...line("RLVS test clips", r.rlvs_test.pr, C.violet), fill: true, backgroundColor: fade(C.violet) }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { type: "linear", min: 0, max: 1, grid: { display: false }, title: { display: true, text: "recall" } }, y: { min: 0, max: 1, title: { display: true, text: "precision" } } } },
    });
    const bins = r.rlvs_test.hist.fight.map((_, i) => ((i + 0.5) / r.rlvs_test.hist.fight.length).toFixed(2));
    const thrBin = Math.min(bins.length - 1, Math.floor(r.rlvs_test.threshold * bins.length));
    this.chart("mdHist", {
      type: "bar",
      data: {
        labels: bins,
        datasets: [
          { label: "Fight clips", data: r.rlvs_test.hist.fight, backgroundColor: rgba(C.fight, 0.8), borderRadius: 3 },
          { label: "Normal clips", data: r.rlvs_test.hist.normal, backgroundColor: rgba(C.normal, 0.8), borderRadius: 3 },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { position: "bottom" } },
        scales: { x: { stacked: true, grid: { display: false }, title: { display: true, text: "attack score" } }, y: { stacked: true, title: { display: true, text: "clips" } } },
      },
      plugins: [
        {
          id: "threshold",
          afterDraw: (c) => {
            const x = c.scales.x.getPixelForValue(thrBin);
            const ctx = c.ctx;
            ctx.save();
            ctx.strokeStyle = C.accent;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([5, 4]);
            ctx.beginPath();
            ctx.moveTo(x, c.chartArea.top);
            ctx.lineTo(x, c.chartArea.bottom);
            ctx.stroke();
            ctx.restore();
          },
        },
      ],
    });
    const imp = r.importance.groups;
    const maxDrop = Math.max(0.001, ...imp.map((g) => g.auc_drop));
    this.chart("mdImp", {
      type: "bar",
      data: {
        labels: imp.map((g) => GROUP_LABEL[g.group] ?? g.group),
        datasets: [{ label: "AUC drop", data: imp.map((g) => g.auc_drop), backgroundColor: imp.map((g) => rgba(C.fight, 0.25 + 0.7 * Math.max(0, g.auc_drop) / maxDrop)), borderRadius: 6 }],
      },
      options: { maintainAspectRatio: false, indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "AUC drop when scrambled" } }, y: { grid: { display: false } } } },
    });
    this.chart("mdSplit", {
      type: "bar",
      data: {
        labels: ["Train", "Validation", "Test"],
        datasets: [
          { label: "Fight", data: (["train", "val", "test"] as const).map((k) => rl.clips[k].fight), backgroundColor: rgba(C.fight, 0.8), borderRadius: 4 },
          { label: "Normal", data: (["train", "val", "test"] as const).map((k) => rl.clips[k].normal), backgroundColor: rgba(C.normal, 0.8), borderRadius: 4 },
        ],
      },
      options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, title: { display: true, text: "clips" } } } },
    });
    this.chart("mdLen", {
      type: "line",
      data: {
        labels: rl.seconds_hist["1"].map((_, i) => `${(i * 0.5).toFixed(1)}s`),
        datasets: [
          { label: "Fight", data: rl.seconds_hist["1"], borderColor: C.fight, backgroundColor: fade(C.fight), fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 },
          { label: "Normal", data: rl.seconds_hist["0"], borderColor: C.normal, backgroundColor: fade(C.normal), fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 },
        ],
      },
      options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { x: { grid: { display: false }, title: { display: true, text: "seconds used (cap 8 s)" } }, y: { title: { display: true, text: "clips" } } } },
    });
    const share = (h: number[]) => {
      const t = h.reduce((a, b) => a + b, 0) || 1;
      return h.map((v) => +((100 * v) / t).toFixed(1));
    };
    this.chart("mdPeople", {
      type: "bar",
      data: {
        labels: rl.people_hist["1"].map((_, i) => (i === rl.people_hist["1"].length - 1 ? `${i}+` : String(i))),
        datasets: [
          { label: "Fight", data: share(rl.people_hist["1"]), backgroundColor: rgba(C.fight, 0.8), borderRadius: 3 },
          { label: "Normal", data: share(rl.people_hist["0"]), backgroundColor: rgba(C.normal, 0.8), borderRadius: 3 },
        ],
      },
      options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { x: { grid: { display: false }, title: { display: true, text: "people in frame" } }, y: { title: { display: true, text: "% of frames" } } } },
    });
    this.chart("mdRes", {
      type: "doughnut",
      data: { labels: Object.keys(rl.resolution), datasets: [{ data: Object.values(rl.resolution), backgroundColor: [C.normal, C.blue, C.violet, C.accent, C.fight], borderWidth: 0, hoverOffset: 6 }] },
      options: { maintainAspectRatio: false, cutout: "66%", plugins: { legend: { position: "bottom" } } } as ChartConfiguration<"doughnut">["options"],
    } as ChartConfiguration);
  }
}

// How the report's feature groups read on the page.
const GROUP_LABEL: Record<string, string> = {
  "Strikes (wrist / ankle speed)": "Sudden limb speed (lunges, swings)",
  "Posture (pose, lean, fallen, crouch)": "Posture (lean, someone down, crouch)",
  "Defense (arms up, shielding head)": "Defensive reaction (arms up, shielding)",
  "Contact (distance, closing in, limb inside)": "Contact (distance, closing in)",
  "Crowd (converging, size)": "Crowd (converging, fleeing)",
};

// ---- the parts that don't depend on a training run --------------------------

// AttackNet's weights by module (the GPU report overwrites these if they differ).
const PARAMS: Record<string, number> = { inp: 13768, temporal: 1237120, people: 618560, head: 26241, "pos + cls tokens": 3520 };

const sec = (id: string, n: string, title: string, sub: string) =>
  `<div class="md-sec md-reveal" id="${id}"><span class="md-sec-n">${n}</span><div><h2>${title}</h2><p>${sub}</p></div></div>`;
const node = (icon: string, title: string, body: string, tag = "") =>
  `<div class="md-node"><div class="md-node-t"><span class="md-ico">${icon}</span>${title}${tag ? `<span class="md-tag">${tag}</span>` : ""}</div><div class="md-node-b">${body}</div></div>`;
const arrow = (label: string) => `<div class="md-arrow"><span>${label}</span></div>`;
const step = (n: number, icon: string, title: string, body: string) =>
  `<div class="md-step md-reveal"><div class="md-step-n">${icon}<i>${n}</i></div><div class="md-step-t">${title}</div><div class="md-step-b">${body}</div></div>`;
const layer = (title: string, body: string, opts: { param?: string; kind?: "io" | "net" | "mil"; stack?: number; shape?: string } = {}) =>
  `<div class="md-layer md-layer-${opts.kind ?? "io"} ${opts.stack ? "md-stack" : ""}" ${opts.stack ? `data-stack="×${opts.stack}"` : ""}>
    <span class="md-spine-dot"></span>
    <div class="md-layer-t">${title}${opts.shape ? `<code>${opts.shape}</code>` : ""}</div>
    <div class="md-layer-b">${body}</div>
    ${opts.param ? `<div class="md-layer-p"><b data-param="${opts.param}">${PARAMS[opts.param].toLocaleString()}</b> weights</div>` : ""}
  </div>`;
const model = (name: string, role: string, size: string, where: string, ours = false) =>
  `<div class="md-model ${ours ? "md-model-ours" : ""}"><div class="md-model-top"><b>${name}</b>${ours ? `<span class="md-tag md-tag-ours">ours · trained</span>` : ""}</div><div class="md-model-size">${size}</div><div class="md-node-b">${role}</div><div class="md-model-where">${where}</div></div>`;

function staticHtml() {
  const total = Object.values(PARAMS).reduce((a, b) => a + b, 0);
  return `
  <div class="md-page">
    <div class="md-hero">
      <div class="md-hero-glow"></div>
      <div class="md-wrap md-hero-in">
        <div class="md-hero-text">
          <div class="md-eyebrow"><span class="md-live"></span>Main model · trained on a Google Cloud NVIDIA T4</div>
          <h1><span class="md-grad">Detect. Dispatch. Act.</span><br/>The model that knows when a scene turns dangerous.</h1>
          <p class="md-lede">AttackNet watches how people move and react, for everyone in the frame, two seconds at a time: aggression, someone defending themselves, a person going down, a crowd closing in. The moment a situation turns into an attack, Eyes raises the incident, dispatches the nearest responders and warns the people around it. Below: the architecture, the data, the training, honest scores on footage it never saw, and the system that acts on it.</p>
        </div>
        <div class="md-hero-card">
          <div class="md-hc-mascot">${logoSvg(76)}</div>
          <div class="md-hc-label">Trainable weights</div>
          <div class="md-hc-num" id="mdWeights">${total.toLocaleString()}</div>
          <div class="md-hc-grid">
            <div><b>84</b><span>body-language signals</span></div>
            <div><b>2 s</b><span>memory per person</span></div>
            <div><b>8</b><span>people at once</span></div>
            <div><b>6</b><span>transformer layers</span></div>
          </div>
          <div class="md-hc-foot">PyTorch · CUDA · YOLOv8x-pose · NVIDIA T4</div>
        </div>
      </div>
    </div>

    <nav class="md-nav">
      <div class="md-wrap md-nav-in">
        <a href="#" data-to="mdArch">Model</a><a href="#" data-to="mdPipe">Data pipeline</a><a href="#" data-to="mdResultsTop">Results</a><a href="#" data-to="mdTraining">Training</a>
        <a href="#" data-to="mdBody">Body language</a><a href="#" data-to="mdData">Data</a><a href="#" data-to="mdInfra">Infrastructure</a><a href="#" data-to="mdFlow">Response flow</a>
      </div>
    </nav>

    <div class="md-wrap">
      ${sec("mdArch", "01", "The main model", "From raw video to an attack probability for each person.")}
      <section class="md-grid md-grid-arch">
        <div class="md-card md-reveal">
          <div class="md-arch">
            <span class="md-pulse"></span>
            ${layer("Video", "Any camera: CCTV, phone, webcam", { shape: "10 fps" })}
            ${layer("YOLOv8x-pose", "Every person's box and 17 keypoints; an IoU tracker keeps their identity frame to frame", { shape: "17 × (x, y, conf)" })}
            ${layer("Body-language features", "Posture, lean, someone down · sudden limb speed · arms up, shielding the head · distance, closing in, contact with another person · crowd converging or fleeing", { shape: "84 per person" })}
            ${layer("Window", "Each person's last two seconds, up to eight people", { shape: "8 × 20 × 84" })}
            ${layer("Input projection", "LayerNorm, then Linear into the model width", { param: "inp", kind: "net", shape: "84 → 160" })}
            ${layer("Temporal transformer", "Reads how each person moves over two seconds: building aggression, a sudden attack, someone recoiling. CLS token, 4 heads", { param: "temporal", kind: "net", stack: 4, shape: "d = 160" })}
            ${layer("People transformer", "Everyone in the frame attends to everyone else: attacker, victim and bystanders only make sense together", { param: "people", kind: "net", stack: 2, shape: "8 people" })}
            ${layer("Head", "MLP to one attack probability per person", { param: "head", kind: "net", shape: "160 → 1" })}
            ${layer("Multiple-instance learning", "Frame score = its most violent person · clip score = its most violent moment. Labels say when; the model learns who", { kind: "mil" })}
          </div>
        </div>
        <div class="md-arch-side">
          <div class="md-card md-reveal"><div class="md-title">${(total / 1e6).toFixed(2)}M weights by part</div><div class="md-canvas"><canvas id="mdParams"></canvas></div></div>
          <div class="md-card md-reveal"><div class="md-title">Why this shape</div>
            <ul class="md-list">
              <li><b>Pose, not pixels.</b> It sees skeletons, so it works on grainy 320×240 CCTV and never needs faces.</li>
              <li><b>Time matters.</b> Danger is a sequence: tension builds, someone lunges, someone recoils. One frame can't tell an attack from a hug or a wave.</li>
              <li><b>People together.</b> A fight is two bodies converging and one shielding; the people transformer sees both.</li>
              <li><b>Small on purpose.</b> 1.9M weights for 2,000 clips learns patterns instead of memorising videos.</li>
            </ul>
          </div>
        </div>
      </section>

      <div class="md-card md-reveal md-gap">
        <div class="md-title">Every model in the system</div>
        <div class="md-models">
          ${model("AttackNet", "Decides, per person, whether this is an attack. The model on this page.", "1.9M weights", "Trained on a GCP NVIDIA T4", true)}
          ${model("YOLOv8x-pose", "Finds every person and their 17 joints, feeding AttackNet.", "69.4M parameters", "Pretrained (Ultralytics) · runs on the T4")}
          ${model("MediaPipe Pose", "33 landmarks in the browser for the live camera: spots an attack in the same frame, with no server round trip.", "On-device", "Runs on the laptop, no server")}
          ${model("DensePose R50", "Maps every visible body's surface for the GPU renders in the Dataset tool.", "Detectron2", "Pretrained (Meta) · runs on the T4")}
          ${model("Gemini 2.5 Flash", "Words the agents' iMessages from grounded facts only.", "Hosted", "Vertex AI · size not published")}
        </div>
      </div>

      ${sec("mdPipe", "02", "Data pipeline", "What goes in, how it's processed, and how it's kept honest.")}
      <div class="md-card md-reveal">
        <div class="md-pipe">
          <div class="md-col">
            <div class="md-col-h">Sources</div>
            ${node("🎞", "Real Life Violence Situations", "1,000 fight + 1,000 normal real-world clips, one label per clip", "2,000")}
            ${node("📹", "Our clips", "UCF-Crime CCTV (fights, assaults, normal), a calm bus stop, and footage from our own camera setup. Labeled frame by frame", "8")}
            ${node("🥊", "Sparring clips", "Kept out of training entirely", "4")}
          </div>
          ${arrow("GPU")}
          <div class="md-col">
            <div class="md-col-h">Processing</div>
            ${node("🦴", "Pose + tracking", "YOLOv8x-pose on the T4 at 10 fps, up to 8 s a clip, half precision, batches of 32")}
            ${node("🧮", "Features", "84 per person per frame; 2 s windows of up to 8 people")}
          </div>
          ${arrow("splits")}
          <div class="md-col">
            <div class="md-col-h">Honest evaluation</div>
            ${node("🅰", "Stage A · pretrain", "RLVS 70% train · 10% thresholds · 20% test, split by clip")}
            ${node("🅱", "Stage B · adapt", "Fine-tune on our clips, leave-one-clip-out: each is scored by a model that never saw it")}
            ${node("🔒", "Never trained on", "The sparring clips and the RLVS test clips")}
          </div>
        </div>
      </div>

      <div id="mdReport"><div class="md-card md-empty"><span class="md-live"></span>Loading the training report…</div></div>

      ${sec("mdInfra", "08", "Infrastructure", "What runs where, from the camera to the iMessage on a phone.")}
      <div class="md-card md-reveal">
        <div class="md-band">${node("▲", "Vercel", "The web app (camera console, map, phone screens) + <code>/api/buses</code>: live MARTA bus positions (GTFS-realtime)", "hosting")}</div>
        <div class="md-infra">
          <div class="md-col">
            <div class="md-col-h">Edge</div>
            ${node("📷", "CCTV camera / webcam", "MediaPipe pose in the browser spots an attack as it happens and captures the evidence frame, on the device itself.", "on device")}
            ${node("📱", "Phones", "Officer, bus driver and rider screens. Join by QR. Alert, camera frame, map, Respond / Injured / Resolved.", "web app")}
          </div>
          ${arrow("mesh ops")}
          <div class="md-col">
            <div class="md-col-h">Eyes mesh</div>
            ${node("🕸", "Replicated store", "Hybrid logical clock + field-level last-writer-wins. Gossips fresh ops only, in 48 KB chunks. Every device holds the incident.", "own build")}
            ${node("🧭", "Networking agent", "Pings every link (RTT, loss), sends each message over the best one with the reason, fails over and resyncs.")}
            <div class="md-chips"><span>Same device</span><span>P2P Wi-Fi · WebRTC</span><span>Mesh radio · simulated</span><span>Internet relay</span></div>
          </div>
          ${arrow("WebSocket")}
          <div class="md-col">
            <div class="md-col-h">Gateway laptop</div>
            ${node("🔌", "Hub", "Bun: presence, WebRTC signaling, relay, bus feed", ":8787")}
            ${node("🚇", "Cloudflare tunnel", "Public <code>wss://</code> address; phones join over it")}
            ${node("🤖", "Dispatch AI", "The agents, the public side (area alerts, care) and the recorder, on the mesh as a device", "Bun")}
          </div>
          ${arrow("HTTPS")}
          <div class="md-col">
            <div class="md-col-h">Cloud</div>
            ${node("✨", "Vertex AI · Gemini 2.5 Flash", "Words each agent's message from grounded facts only", "GCP")}
            ${node("💬", "Photon Spectrum → iMessage", "Responder thread, police photos, EMS pages, area alerts with polls, replies back in", "Photon")}
            ${node("🗄", "Cloud SQL Postgres", "Incidents, acknowledgements, agent messages, evidence frames, sign-ups", "GCP")}
            ${node("🪣", "Cloud Storage", "Private: clips, DensePose renders, evidence JPEGs, model outputs, via 7-day signed links", "GCP")}
            ${node("⚡", "Compute Engine · NVIDIA T4", "DensePose, YOLOv8x-pose, AttackNet training. Created per run, deleted after", "GPU")}
          </div>
        </div>
        <div class="md-band">${node("🌐", "Open data", "MARTA GTFS + GTFS-realtime · OpenStreetMap places and Nominatim · Real Life Violence Situations · UCF-Crime", "inputs")}</div>
      </div>

      ${sec("mdFlow", "09", "Response flow", "From danger on camera to help on the way, in seconds.")}
      <div class="md-steps">
        ${step(1, "👁", "Detect", "The camera sees a situation turn violent and scores everyone's body language. The incident is raised instantly, with the evidence frame, and updated as it unfolds.")}
        ${step(2, "🕸", "Spread", "The incident gossips across the mesh over whatever works: internet, P2P Wi-Fi or radio. 120+ city nodes have it in seconds.")}
        ${step(3, "🧭", "Coordinate", "Agents take turns in one thread: Sentinel, Network, Dispatch, Police / EMS, Safety, Care. Facts only; nobody is “on the way” until a human says so.")}
        ${step(4, "💬", "Notify", "Over iMessage: the responder thread, live camera frames to police as photos, loud EMS pages with a map link.")}
        ${step(5, "🛡", "Protect", "People signed up within 800 m get a loud alert, a safe spot and an “Are you safe?” poll. “I need help” opens care.")}
        ${step(6, "✅", "Close", "“On my way” marks responders. Police resolve; everyone alerted gets the all clear. All of it lands in Postgres and the bucket.")}
      </div>
      <p class="md-foot">Eyes · Hacklanta 2026 · every number on this page comes from the training run's own report.</p>
    </div>
  </div>`;
}
