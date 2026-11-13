# LinkedIn post draft

🛡️ I built an AI exam proctor that catches cheating in real time — and it runs 100% in your browser.

Over the past weeks, for my Deep Learning semester project, I built **ProctorAI**: open the link, allow your webcam, and it flags the four most common online-exam cheating signals live:

👀 Looking away (left / right / up / down)
📱 Phone in view
👥 A second person in the frame
🚫 Student leaving the camera

What's under the hood:
• MediaPipe face landmarks → a custom PyTorch MLP I trained on 4.6k samples derived from the Head-Pose Image Database (89.5% test accuracy, 95.4% attentive-vs-away)
• YOLOv8n for phone + person detection (exported to ONNX at 320 px)
• A temporal event engine that debounces predictions (≥1 s) and produces a timestamped timeline + integrity score
• Everything exported to ONNX and run with ONNX Runtime Web (WebAssembly) → no server, no video upload, deployable as a static site on Vercel

The hardest parts weren't the models — they were making the Python and JavaScript feature pipelines numerically identical, and designing the temporal logic so a blink or a quick glance doesn't become a false alarm.

🔗 Live demo: {{VERCEL_URL}}
💻 Code + weights: {{GITHUB_URL}}
▶️ Walkthrough video: {{YOUTUBE_URL}}
📄 Report: in the repo

Try it, try to "cheat", and tell me how it did. Feedback and ideas are very welcome — especially from people working on EdTech, assessment or computer vision.

#DeepLearning #ComputerVision #MachineLearning #PyTorch #YOLO #MediaPipe #ONNX #EdTech #WebDevelopment #Vercel #AI #OpenSource
