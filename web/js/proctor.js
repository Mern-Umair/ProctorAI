// ProctorAI - live proctoring engine + demo UI.
// Everything runs in the browser: webcam frames never leave the device.

import { MODEL_FILES, THRESHOLDS, PENALTIES, LABELS } from "./config.js";
import { CLASSES, landmarksToFeatures, standardise, softmax } from "./features.js";
import { YoloDetector } from "./yolo.js";
import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const VIOLATIONS = ["looking_away", "phone", "multiple_faces", "no_face"];
// Camera-view -> student-view (the preview is mirrored, so left/right swap)
const MIRROR = { left: "right", right: "left" };

// ------------------------------------------------------------------ violation tracker
class Tracker {
  constructor(type) {
    this.type = type;
    this.activeSince = null;
    this.clearSince = null;
    this.event = null;
    this.events = [];
  }
  update(flag, now) {
    if (flag) {
      this.clearSince = null;
      if (this.activeSince === null) this.activeSince = now;
      if (!this.event && now - this.activeSince >= THRESHOLDS.minEventMs) {
        this.event = { type: this.type, start: this.activeSince, end: null };
        this.events.push(this.event);
        return "opened";
      }
    } else {
      this.activeSince = null;
      if (this.event) {
        if (this.clearSince === null) this.clearSince = now;
        if (now - this.clearSince >= THRESHOLDS.endEventMs) {
          this.event.end = this.clearSince;
          this.event = null;
          this.clearSince = null;
          return "closed";
        }
      }
    }
    return null;
  }
  close(now) {
    if (this.event) {
      this.event.end = now;
      this.event = null;
    }
  }
}

function eventPenalty(ev, now) {
  const p = PENALTIES[ev.type];
  const dur = ((ev.end ?? now) - ev.start) / 1000;
  return p.perEvent + p.perSecond * Math.max(0, dur - p.perSecondAfter);
}

function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function verdict(score) {
  if (score >= 85) return { text: "Clean", cls: "ok" };
  if (score >= 60) return { text: "Suspicious", cls: "warn" };
  return { text: "High risk", cls: "bad" };
}

// ------------------------------------------------------------------ app
class ProctorApp {
  constructor() {
    this.video = $("video");
    this.canvas = $("overlay");
    this.ctx = this.canvas.getContext("2d");
    this.running = false;
    this.session = null;
    this.trackers = {};
    this.attnProbs = new Array(CLASSES.length).fill(0);
    this.attnProbs[0] = 1;
    this.lastPhoneRun = 0;
    this.phoneBoxes = [];
    this.personBoxes = [];
    this.phoneBusy = false;
    this.fps = { frames: 0, last: performance.now(), value: 0 };
    this.bindUI();
  }

  bindUI() {
    $("btn-start").addEventListener("click", () => this.start());
    $("btn-end").addEventListener("click", () => this.end());
    $("btn-restart").addEventListener("click", () => location.reload());
    $("btn-json").addEventListener("click", () => this.downloadJSON());
    $("btn-print").addEventListener("click", () => window.print());
    $("mirror").addEventListener("change", (e) => {
      document.querySelector(".stage").classList.toggle("mirrored", e.target.checked);
    });
  }

  setStatus(text, cls = "") {
    const el = $("status");
    el.textContent = text;
    el.className = `status ${cls}`;
  }

  async loadModels() {
    this.setStatus("Loading face landmark model…", "loading");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm",
    );
    this.landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_FILES.faceLandmarker, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: THRESHOLDS.maxFaces,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false,
    });

    this.setStatus("Loading attention classifier…", "loading");
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1;
    this.attn = await ort.InferenceSession.create(MODEL_FILES.attentionOnnx, {
      executionProviders: ["wasm"],
    });
    const norm = await (await fetch(MODEL_FILES.attentionNorm)).json();
    this.mean = Float32Array.from(norm.mean);
    this.std = Float32Array.from(norm.std);

    this.setStatus("Loading object detector (YOLOv8n)…", "loading");
    this.detector = await YoloDetector.load(ort, MODEL_FILES.detectorOnnx, 320);
    this.detClasses = new Set(Object.values(MODEL_FILES.detectorClasses));
  }

  async start() {
    $("btn-start").disabled = true;
    try {
      if (!this.landmarker) await this.loadModels();
      this.setStatus("Requesting camera…", "loading");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      this.video.srcObject = stream;
      await new Promise((r) => (this.video.onloadedmetadata = r));
      await this.video.play();
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    } catch (err) {
      console.error(err);
      this.setStatus(`Could not start: ${err.message}`, "bad");
      $("btn-start").disabled = false;
      return;
    }
    this.session = { start: performance.now(), startedAt: new Date().toISOString() };
    for (const t of VIOLATIONS) this.trackers[t] = new Tracker(t);
    this.running = true;
    document.body.classList.add("live");
    $("btn-end").disabled = false;
    this.setStatus("Monitoring – looking at screen", "ok");
    this.loop();
    this.timer = setInterval(() => this.renderPanel(), 250);
  }

  async loop() {
    if (!this.running) return;
    const now = performance.now();
    const flags = { looking_away: false, phone: false, multiple_faces: false, no_face: false };
    let attnClass = "center", attnConf = 1;

    // ---- faces + attention
    const res = this.landmarker.detectForVideo(this.video, now);
    const faces = res.faceLandmarks || [];
    if (faces.length === 0) {
      flags.no_face = true;
    } else {
      if (faces.length > 1) flags.multiple_faces = true;
      const feats = standardise(landmarksToFeatures(faces[0]), this.mean, this.std);
      const out = await this.attn.run({ features: new ort.Tensor("float32", feats, [1, feats.length]) });
      const probs = softmax(out.logits.data);
      const a = THRESHOLDS.attentionEma;
      for (let i = 0; i < probs.length; i++) this.attnProbs[i] = a * probs[i] + (1 - a) * this.attnProbs[i];
      let best = 0;
      for (let i = 1; i < this.attnProbs.length; i++) if (this.attnProbs[i] > this.attnProbs[best]) best = i;
      attnClass = CLASSES[best];
      attnConf = this.attnProbs[best];
      if (attnClass !== "center") flags.looking_away = true;
    }

    // ---- object detector: phone + person (throttled, non-blocking)
    if (!this.phoneBusy && now - this.lastPhoneRun > THRESHOLDS.phoneEveryMs) {
      this.phoneBusy = true;
      this.lastPhoneRun = now;
      const { phone: PHONE, person: PERSON } = MODEL_FILES.detectorClasses;
      const minConf = Math.min(THRESHOLDS.phoneConf, THRESHOLDS.personConf);
      this.detector
        .detect(this.video, this.video.videoWidth, this.video.videoHeight, minConf, THRESHOLDS.phoneIou, this.detClasses)
        .then((boxes) => {
          this.phoneBoxes = boxes.filter((b) => b.cls === PHONE && b.conf >= THRESHOLDS.phoneConf);
          this.personBoxes = boxes.filter((b) => b.cls === PERSON && b.conf >= THRESHOLDS.personConf);
        })
        .catch(console.error)
        .finally(() => (this.phoneBusy = false));
    }
    if (this.phoneBoxes.length > 0) flags.phone = true;
    if (this.personBoxes.length > 1) flags.multiple_faces = true;

    // ---- trackers
    for (const t of VIOLATIONS) {
      const change = this.trackers[t].update(flags[t], now);
      if (change === "opened") this.logEvent(this.trackers[t].events.at(-1));
    }

    this.draw(faces, attnClass);
    this.renderLive(flags, faces.length, attnClass, attnConf);
    this.tickFps(now);
    requestAnimationFrame(() => this.loop());
  }

  draw(faces, attnClass) {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const W = canvas.width, H = canvas.height;
    faces.forEach((lm, i) => {
      let minX = 1, minY = 1, maxX = 0, maxY = 0;
      for (const p of lm) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
      }
      const ok = i === 0 && attnClass === "center" && faces.length === 1;
      ctx.strokeStyle = ok ? "#22c55e" : "#f59e0b";
      ctx.lineWidth = 2;
      ctx.strokeRect(minX * W, minY * H, (maxX - minX) * W, (maxY - minY) * H);
      ctx.fillStyle = ok ? "rgba(34,197,94,.7)" : "rgba(245,158,11,.7)";
      for (let k = 0; k < 468; k += 6) {
        ctx.fillRect(lm[k].x * W - 1, lm[k].y * H - 1, 2, 2);
      }
    });
    if (this.personBoxes.length > 1) {
      for (const b of this.personBoxes) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 6]);
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.setLineDash([]);
      }
    }
    for (const b of this.phoneBoxes) {
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 3;
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.fillStyle = "rgba(239,68,68,.25)";
      ctx.fillRect(b.x, b.y, b.w, b.h);
    }
  }

  renderLive(flags, nFaces, attnClass, attnConf) {
    const active = VIOLATIONS.filter((t) => this.trackers[t].event);
    if (active.length) {
      const l = LABELS[active[0]];
      this.setStatus(`${l.icon} ${l.title}${active.length > 1 ? ` +${active.length - 1}` : ""}`, "bad");
    } else if (Object.values(flags).some(Boolean)) {
      this.setStatus("Checking…", "warn");
    } else {
      this.setStatus("Monitoring – looking at screen", "ok");
    }
    $("chip-faces").textContent = `Faces: ${nFaces} · Persons: ${this.personBoxes.length}`;
    const shown = MIRROR[attnClass] ?? attnClass;
    $("chip-attn").textContent = nFaces ? `Gaze: ${shown} (${Math.round(attnConf * 100)}%)` : "Gaze: –";
    $("chip-phone").textContent = `Phone: ${this.phoneBoxes.length ? "yes" : "no"}`;
  }

  renderPanel() {
    if (!this.session) return;
    const now = performance.now();
    $("timer").textContent = fmtTime(now - this.session.start);
    let total = 0;
    for (const t of VIOLATIONS) {
      const tr = this.trackers[t];
      $(`count-${t}`).textContent = tr.events.length;
      for (const ev of tr.events) total += eventPenalty(ev, now);
    }
    const score = Math.max(0, Math.round(100 - total));
    const v = verdict(score);
    const el = $("score");
    el.textContent = score;
    el.className = `score ${v.cls}`;
    $("verdict").textContent = v.text;
  }

  logEvent(ev) {
    const l = LABELS[ev.type];
    const li = document.createElement("li");
    li.innerHTML = `<span class="t">${fmtTime(ev.start - this.session.start)}</span> ${l.icon} ${l.title}`;
    $("log").prepend(li);
  }

  tickFps(now) {
    this.fps.frames++;
    if (now - this.fps.last >= 1000) {
      this.fps.value = this.fps.frames;
      this.fps.frames = 0;
      this.fps.last = now;
      $("chip-fps").textContent = `${this.fps.value} fps`;
    }
  }

  end() {
    if (!this.running) return;
    this.running = false;
    clearInterval(this.timer);
    const now = performance.now();
    for (const t of VIOLATIONS) this.trackers[t].close(now);
    this.video.srcObject.getTracks().forEach((tr) => tr.stop());
    this.session.end = now;
    this.session.endedAt = new Date().toISOString();
    this.report = this.buildReport();
    this.renderReport();
    document.body.classList.remove("live");
    document.body.classList.add("done");
    $("btn-end").disabled = true;
  }

  buildReport() {
    const s = this.session;
    const events = VIOLATIONS.flatMap((t) => this.trackers[t].events).sort((a, b) => a.start - b.start);
    const summary = {};
    let totalPenalty = 0;
    for (const t of VIOLATIONS) {
      const evs = this.trackers[t].events;
      const dur = evs.reduce((a, e) => a + (e.end - e.start), 0) / 1000;
      const pen = evs.reduce((a, e) => a + eventPenalty(e, s.end), 0);
      totalPenalty += pen;
      summary[t] = { count: evs.length, totalSeconds: +dur.toFixed(1), penalty: +pen.toFixed(1) };
    }
    const score = Math.max(0, Math.round(100 - totalPenalty));
    return {
      project: "ProctorAI",
      startedAt: s.startedAt,
      endedAt: s.endedAt,
      durationSeconds: +((s.end - s.start) / 1000).toFixed(1),
      integrityScore: score,
      verdict: verdict(score).text,
      summary,
      timeline: events.map((e) => ({
        type: e.type,
        startSec: +((e.start - s.start) / 1000).toFixed(1),
        endSec: +((e.end - s.start) / 1000).toFixed(1),
        durationSec: +((e.end - e.start) / 1000).toFixed(1),
      })),
    };
  }

  renderReport() {
    const r = this.report;
    const v = verdict(r.integrityScore);
    $("r-score").textContent = r.integrityScore;
    $("r-score").className = `big-score ${v.cls}`;
    $("r-verdict").textContent = v.text;
    $("r-duration").textContent = fmtTime(r.durationSeconds * 1000);
    $("r-total").textContent = r.timeline.length;
    const tb = $("r-summary");
    tb.innerHTML = "";
    for (const t of VIOLATIONS) {
      const s = r.summary[t], l = LABELS[t];
      tb.insertAdjacentHTML(
        "beforeend",
        `<tr><td>${l.icon} ${l.title}</td><td>${s.count}</td><td>${s.totalSeconds}s</td><td>-${s.penalty}</td></tr>`,
      );
    }
    const tl = $("r-timeline");
    tl.innerHTML = r.timeline.length
      ? r.timeline
          .map((e) => `<li><span class="t">${fmtTime(e.startSec * 1000)} – ${fmtTime(e.endSec * 1000)}</span> ${LABELS[e.type].icon} ${LABELS[e.type].title} <em>(${e.durationSec}s)</em></li>`)
          .join("")
      : "<li>No violations recorded 🎉</li>";
    $("report").scrollIntoView({ behavior: "smooth" });
  }

  downloadJSON() {
    const blob = new Blob([JSON.stringify(this.report, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `proctorai-report-${Date.now()}.json`;
    a.click();
  }
}

window.addEventListener("DOMContentLoaded", () => new ProctorApp());
