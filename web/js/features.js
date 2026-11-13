// JavaScript twin of scripts/common.py :: landmarks_to_features
// Must stay numerically identical to the Python version used at training time.

export const CLASSES = ["center", "left", "right", "up", "down"];
const NOSE_TIP = 1;
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_OUTER = 263;
const FOREHEAD = 10, CHIN = 152, LEFT_CHEEK = 234, RIGHT_CHEEK = 454, MOUTH_LEFT = 61, MOUTH_RIGHT = 291;
export const N_LANDMARKS = 468;
export const N_GEO = 10;
export const FEATURE_DIM = N_LANDMARKS * 3 + N_GEO;

const norm2 = (p) => Math.hypot(p[0], p[1]);

/**
 * @param {Array<{x:number,y:number,z:number}>} lm  MediaPipe normalised landmarks (>=468)
 * @returns {Float32Array} 1414 features: 1404 nose-centred / eye-scaled coords + 10 geometry cues
 */
export function landmarksToFeatures(lm) {
  const out = new Float32Array(FEATURE_DIM);
  const nx = lm[NOSE_TIP].x, ny = lm[NOSE_TIP].y, nz = lm[NOSE_TIP].z;
  const dx = lm[RIGHT_EYE_OUTER].x - lm[LEFT_EYE_OUTER].x;
  const dy = lm[RIGHT_EYE_OUTER].y - lm[LEFT_EYE_OUTER].y;
  const scale = Math.hypot(dx, dy) + 1e-6;
  for (let i = 0; i < N_LANDMARKS; i++) {
    out[i * 3] = (lm[i].x - nx) / scale;
    out[i * 3 + 1] = (lm[i].y - ny) / scale;
    out[i * 3 + 2] = (lm[i].z - nz) / scale;
  }
  // ---- geometry cues (mirror of scripts/common.py::geometric_features)
  const P = (i) => [out[i * 3], out[i * 3 + 1], out[i * 3 + 2]];
  const le = P(LEFT_EYE_OUTER), re = P(RIGHT_EYE_OUTER);
  const eyeMid = [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2, (le[2] + re[2]) / 2];
  const lc = P(LEFT_CHEEK), rc = P(RIGHT_CHEEK), fh = P(FOREHEAD), ch = P(CHIN);
  const ml = P(MOUTH_LEFT), mr = P(MOUTH_RIGHT);
  const mouthMid = [(ml[0] + mr[0]) / 2, (ml[1] + mr[1]) / 2];
  const faceW = Math.hypot(rc[0] - lc[0], rc[1] - lc[1]) + 1e-6;
  const faceH = Math.hypot(ch[0] - fh[0], ch[1] - fh[1]) + 1e-6;
  const g = N_LANDMARKS * 3;
  out[g + 0] = -eyeMid[0];
  out[g + 1] = (norm2(rc) - norm2(lc)) / faceW;
  out[g + 2] = re[2] - le[2];
  out[g + 3] = rc[2] - lc[2];
  out[g + 4] = -eyeMid[1];
  out[g + 5] = mouthMid[1];
  out[g + 6] = (norm2(ch) - norm2(fh)) / faceH;
  out[g + 7] = fh[2] - ch[2];
  out[g + 8] = Math.atan2(re[1] - le[1], re[0] - le[0]);
  out[g + 9] = faceW / faceH;
  return out;
}

/** Standardise in-place with the training mean / std. */
export function standardise(features, mean, std) {
  for (let i = 0; i < features.length; i++) features[i] = (features[i] - mean[i]) / std[i];
  return features;
}

export function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
