# ProctorAI – Presentation voice-over script

**Presenter:** Umair Tahir · Roll No SU92-BSCSM-F23-393 · Section 7B
**Target length:** 6–8 minutes (≈ 40 s per slide). Read naturally; pause at "—".

---

### Slide 1 – Title
"Assalam-o-Alaikum. My name is Umair Tahir, roll number SU92-BSCSM-F23-393, section 7B. Today I am presenting my Deep Learning semester project: **ProctorAI – a real-time AI exam proctoring system** that runs completely inside the web browser."

### Slide 2 – The problem
"Since online exams became common, one thing never moved online — the invigilator. A teacher cannot watch forty webcam tiles at once. Commercial proctoring services are expensive, and most of them stream the student's camera to a server, which creates privacy and bandwidth problems. — But cheating behaviours are actually very predictable: students look away at notes, pick up a phone, let someone else join them, or simply leave the camera."

### Slide 3 – Objectives
"So my goal was a system that: one — detects those four behaviours in real time; two — ignores innocent movements like a blink or a quick glance by reasoning over time; three — runs fully client-side, so it opens from a link with nothing to install and no video ever leaves the device; and four — produces an auditable timeline and an integrity score that a teacher can review."

### Slide 4 – Datasets
"I used two public datasets from the Hugging Face Hub. The first is the Head-Pose Image Database — fifteen people photographed at ninety-three known head angles, from minus ninety to plus ninety degrees of pan and tilt. I kept two thousand seven hundred and ninety unique poses. The second is a phone-detection dataset in COCO format with about six hundred annotated images, which I converted to YOLO format."

### Slide 5 – Label engineering
"Head angles were converted into five attention classes. If both pan and tilt are within fifteen degrees, the student is looking at the screen — that's *center*. Otherwise the larger angle decides left, right, up or down. I verified the sign convention visually on the images. Every image was also flipped horizontally with left and right swapped, which doubles the data and makes the classifier symmetric."

### Slide 6 – System architecture
"Here is the full pipeline. Each webcam frame goes through MediaPipe FaceLandmarker, which gives four hundred and sixty-eight 3-D face points. Those points are converted into a feature vector and classified by my PyTorch attention model. In parallel, every three hundred and fifty milliseconds, YOLOv8-nano looks for a phone and counts people. A temporal event engine then turns these per-frame flags into debounced violations and a score."

### Slide 7 – Feature engineering
"The feature vector must not depend on where the face is or how big it appears. So I centre all landmarks on the nose tip and divide by the distance between the outer eye corners. That gives one thousand four hundred and four numbers, plus ten explicit geometry cues for yaw, pitch and roll, that describe only the *shape and orientation* of the face. The same code exists in Python for training and in JavaScript for the browser — they must match exactly."

### Slide 8 – Attention classifier
"The classifier is a compact multilayer perceptron: one thousand four hundred and fourteen inputs, two hidden layers of five hundred twelve and two hundred fifty-six units with batch-norm, ReLU and dropout, and five outputs. I trained it with AdamW, a cosine learning-rate schedule, label smoothing and class weights for one hundred epochs, and picked the hyper-parameters with a small grid search scored on the validation split only. The split was done by *source image*, so a flipped copy can never leak from training into testing."

### Slide 9 – Results: attention classifier
"On the held-out test set the model reached **89.5% accuracy** and a macro-F1 of **0.880**. The confusion matrix shows that the few errors are between neighbouring classes — for example *up* versus *center* at fifteen degrees — which is exactly where a human would also hesitate."

### Slide 10 – Object detector
"For phones and people I use YOLOv8-nano, exported to ONNX at three-hundred-twenty pixels so it runs in real time on a CPU. I also fine-tuned it on the phone dataset as an experiment. On the same test split, fine-tuning lifted mAP at fifty from 78 to 93 percent. But that dataset is close-up product photos, and the general COCO model also detects people, so the demo ships the COCO model, with the fine-tuned one available as a one-line swap."

### Slide 11 – Temporal event engine & integrity score
"Frame-level predictions are noisy, so a flag must stay active for at least one second before it becomes a violation, and clear for point-six seconds before it closes. Each violation carries a penalty — fifteen points for a phone, ten for a second person, and so on — and the integrity score is one hundred minus the total. Above eighty-five is clean, below sixty is high risk."

### Slide 12 – Deployment
"The web app is a static site on Vercel. The trained models are exported to ONNX and executed with ONNX Runtime Web — WebAssembly in the browser. MediaPipe also runs as WebAssembly. That is why anyone — a teacher, a recruiter — can open the link and test it in thirty seconds, and why the student's video never leaves the laptop."

### Slide 13 – Live demo
"Let me show it. — I start a session. The status is green: monitoring. Now I look to the side — after one second the banner turns red: *looking away*, and the counter goes up. I pick up my phone — *phone detected*, with a red box. My friend steps in — *multiple persons*. I leave the frame — *face missing*. When I end the session, I get a full report: counts, durations, a timeline, and an integrity score I can download as JSON or PDF."

### Slide 14 – Limitations & future work
"Some honest limitations. The model reads head pose, not eye gaze, so eyes-only glances are missed — iris landmarks would fix that. A phone hidden below the desk is invisible. And automated proctoring must never be the sole basis for an accusation — that's why the output is a timeline for human review. Future work: audio-based talking detection, a multi-student teacher dashboard, and a larger webcam-domain phone dataset."

### Slide 15 – Conclusion
"To summarise: ProctorAI is an end-to-end deep learning project — public datasets, a purpose-trained PyTorch model with **89.5%** test accuracy, a YOLOv8 detector, ONNX export, and a live web deployment that anyone can try from a link. The code, weights, report and this presentation are all on GitHub. Thank you — I'm happy to take questions."
