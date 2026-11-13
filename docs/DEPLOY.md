# Deploying ProctorAI (GitHub + Vercel) – step by step

## 1. Push to GitHub (public repo)

```powershell
cd D:\deep-learning\ProctorAI
git init
git add .
git commit -m "ProctorAI: real-time AI exam proctoring (attention MLP + YOLOv8n, ONNX web app)"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/ProctorAI.git
git push -u origin main
```

What gets pushed (already arranged by `.gitignore`): source code, `models/` (trained weights `.pt`/`.onnx` + metrics),
`web/` (site + copies of the ONNX models), `docs/` (report PDF/DOCX, slides, figures). Raw datasets are **not**
pushed – `data/README.md` + `scripts/download_data.py` re-create them.

> Weights are ~ 50 MB in total (`models/` + `web/models/`). GitHub accepts files up to 100 MB each, so no LFS is needed.

## 2. Deploy on Vercel

1. Go to https://vercel.com → **Add New… → Project** → import the `ProctorAI` repository.
2. **Root Directory → Edit → `web`** (important: the site lives in the `web/` folder).
3. Framework preset: **Other**. Build command: *(leave empty)*. Output directory: *(leave empty)*.
4. Click **Deploy**. You get a URL like `https://proctorai-xxxx.vercel.app`.
5. Open `https://<your-url>/demo.html`, allow the camera and test it (HTTPS is required for webcam access – Vercel provides it).

## 3. Put the real links into the site

Edit `web/js/config.js`:

```js
export const LINKS = {
  github:  "https://github.com/<YOUR_USERNAME>/ProctorAI",
  report:  "https://github.com/<YOUR_USERNAME>/ProctorAI/blob/main/docs/ProctorAI_Report.pdf",
  slides:  "https://github.com/<YOUR_USERNAME>/ProctorAI/blob/main/docs/ProctorAI_Presentation.pptx",
  youtube: "https://www.youtube.com/watch?v=<VIDEO_ID>",
  linkedin:"https://www.linkedin.com/in/<YOUR_PROFILE>",
  ...
};
```

Also update the **Live demo** line at the top of `README.md` with the Vercel URL, then:

```powershell
git add .
git commit -m "Add deployment, video and report links"
git push
```

Vercel redeploys automatically on every push.

## 4. Record the presentation video

1. Open `docs/ProctorAI_Presentation.pptx` in PowerPoint → **Slide Show**.
2. Record screen + microphone (Windows: `Win + Alt + R` via Xbox Game Bar, or OBS Studio).
3. Read `docs/presentation_script.md` slide by slide (≈ 6–8 min). For slide 13 switch to the browser and do the live demo.
4. Upload to YouTube as **Unlisted** (or Public), copy the link into `web/js/config.js` and `README.md`.

## 5. LinkedIn

Use `docs/linkedin_post.md`, replace the `{{…}}` links, attach a short screen-recording GIF/clip of the demo if possible.
