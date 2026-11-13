// Fills the landing page with real numbers from the exported metrics files
// (models/attention_metrics.json, models/phone_metrics.json) and the links in config.js.
import { LINKS, AUTHOR } from "./config.js";

const $ = (id) => document.getElementById(id);
const pct = (x) => `${(x * 100).toFixed(1)}%`;

function setLinks() {
  for (const [id, key] of [
    ["hero-github", "github"], ["res-github", "github"], ["res-report", "report"],
    ["res-slides", "slides"], ["res-youtube", "youtube"], ["foot-linkedin", "linkedin"],
    ["ds-headpose", "datasetHeadPose"], ["ds-phone", "datasetPhone"],
  ]) {
    const el = $(id);
    if (el) el.href = LINKS[key];
  }
  if (AUTHOR?.name) $("author").textContent = `ProctorAI · ${AUTHOR.name} · ${AUTHOR.course || "Deep Learning semester project"}`;

  // embed YouTube if a real video id is configured
  const m = LINKS.youtube.match(/(?:v=|youtu\.be\/)([\w-]{11})/);
  if (m) {
    $("yt").src = `https://www.youtube.com/embed/${m[1]}`;
    $("video-wrap").hidden = false;
  }
}

async function loadMetrics() {
  try {
    const a = await (await fetch("./models/attention_metrics.json")).json();
    $("s-attn-acc").textContent = pct(a.test_accuracy);
    $("r-attn-acc").textContent = pct(a.test_accuracy);
    $("r-attn-f1").textContent = a.test_macro_f1.toFixed(3);
    $("r-attn-split").textContent = `${a.n_train} / ${a.n_val} / ${a.n_test}`;
    if (a.test_binary_attentive_vs_away_accuracy) $("r-attn-bin").textContent = pct(a.test_binary_attentive_vs_away_accuracy);
    if (a.config?.hidden) $("r-attn-arch").textContent = `${a.feature_dim} → ${a.config.hidden.join(" → ")} → 5`;
    const total = a.n_train + a.n_val + a.n_test;
    $("s-headpose-n").textContent = total.toLocaleString();
    $("ds-headpose-n").textContent = `${total.toLocaleString()} landmark samples`;
  } catch (e) {
    console.warn("attention metrics not available yet", e);
  }
  try {
    const p = await (await fetch("./models/phone_metrics.json")).json();
    $("s-phone-map").textContent = pct(p.test_mAP50);
    $("r-phone-map50").textContent = p.coco_baseline ? `${pct(p.test_mAP50)} vs ${pct(p.coco_baseline.test_mAP50)}` : pct(p.test_mAP50);
    $("r-phone-map").textContent = pct(p.test_mAP50_95);
    $("r-phone-pr").textContent = `${pct(p.test_precision)} / ${pct(p.test_recall)}`;
    $("r-phone-cfg").textContent = `${p.imgsz} px · ${p.epochs} epochs`;
  } catch (e) {
    console.warn("phone metrics not available yet", e);
  }
}

setLinks();
loadMetrics();
