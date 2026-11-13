// Cross-check that web/js/features.js reproduces scripts/common.py exactly.
import { readFileSync } from "node:fs";
import { landmarksToFeatures } from "../web/js/features.js";
const d = JSON.parse(readFileSync(new URL("../data/processed/feature_check.json", import.meta.url)));
let maxDiff = 0;
d.landmarks.forEach((lm, i) => {
  const js = landmarksToFeatures(lm);
  const py = d.features[i];
  if (js.length !== py.length) throw new Error(`length mismatch ${js.length} vs ${py.length}`);
  for (let k = 0; k < js.length; k++) maxDiff = Math.max(maxDiff, Math.abs(js[k] - py[k]));
});
console.log(`JS vs Python features: ${d.landmarks.length} samples, dim ${d.features[0].length}, max abs diff = ${maxDiff.toExponential(2)}`);
if (maxDiff > 1e-4) { console.error("MISMATCH"); process.exit(1); }
