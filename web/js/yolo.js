// Minimal YOLOv8 inference helper for onnxruntime-web.
// Works for any Ultralytics export with output shape [1, 4 + nc, anchors]
// (e.g. the fine-tuned single-class phone model, or the 80-class COCO model).

export class YoloDetector {
  constructor(ort, session, size = 320) {
    this.ort = ort;
    this.session = session;
    this.size = size;
    this.canvas = document.createElement("canvas");
    this.canvas.width = size;
    this.canvas.height = size;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
  }

  static async load(ort, url, size = 320) {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new YoloDetector(ort, session, size);
  }

  /** Letterbox `source` (video/canvas) into a square tensor. Returns scale + padding for un-mapping. */
  preprocess(source, sw, sh) {
    const s = this.size;
    const r = Math.min(s / sw, s / sh);
    const nw = Math.round(sw * r), nh = Math.round(sh * r);
    const px = (s - nw) / 2, py = (s - nh) / 2;
    this.ctx.fillStyle = "#727272";
    this.ctx.fillRect(0, 0, s, s);
    this.ctx.drawImage(source, 0, 0, sw, sh, px, py, nw, nh);
    const { data } = this.ctx.getImageData(0, 0, s, s);
    const chw = new Float32Array(3 * s * s);
    const area = s * s;
    for (let i = 0; i < area; i++) {
      chw[i] = data[i * 4] / 255;
      chw[area + i] = data[i * 4 + 1] / 255;
      chw[2 * area + i] = data[i * 4 + 2] / 255;
    }
    return { tensor: new this.ort.Tensor("float32", chw, [1, 3, s, s]), r, px, py };
  }

  /**
   * @param {Set<number>|null} classes  class ids to keep (null = all)
   * @returns {Promise<Array<{cls:number,conf:number,x:number,y:number,w:number,h:number}>>} boxes in source pixels
   */
  async detect(source, sw, sh, confThr = 0.45, iouThr = 0.45, classes = null) {
    const { tensor, r, px, py } = this.preprocess(source, sw, sh);
    const out = await this.session.run({ [this.inputName]: tensor });
    const o = out[this.outputName];
    const [, ch, n] = o.dims; // [1, 4+nc, anchors]
    const d = o.data;
    const nc = ch - 4;
    const cands = [];
    for (let i = 0; i < n; i++) {
      let best = 0, bestCls = -1;
      for (let c = 0; c < nc; c++) {
        if (classes && !classes.has(c)) continue;
        const v = d[(4 + c) * n + i];
        if (v > best) { best = v; bestCls = c; }
      }
      if (best < confThr) continue;
      const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
      cands.push({
        cls: bestCls,
        conf: best,
        x: (cx - w / 2 - px) / r,
        y: (cy - h / 2 - py) / r,
        w: w / r,
        h: h / r,
      });
    }
    return nmsPerClass(cands, iouThr);
  }
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9);
}

export function nmsPerClass(boxes, thr) {
  boxes.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => k.cls !== b.cls || iou(k, b) < thr)) keep.push(b);
  }
  return keep;
}
