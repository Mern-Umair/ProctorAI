// ProctorAI - project links & tunable thresholds.
// Fill the links in once the repo / video / report are published.
export const AUTHOR = {
  name: "",
  course: "Deep Learning – Semester Project",
};

export const LINKS = {
  github: "https://github.com/Mern-Umair/ProctorAI",
  report: "https://github.com/Mern-Umair/ProctorAI/blob/main/docs/ProctorAI_Report.pdf",
  slides: "https://github.com/Mern-Umair/ProctorAI/blob/main/docs/ProctorAI_Presentation.pptx",
  youtube: "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
  linkedin: "https://www.linkedin.com/in/YOUR_PROFILE",
  datasetHeadPose: "https://huggingface.co/datasets/StevenLe456/head-pose",
  datasetPhone: "https://huggingface.co/datasets/harshdadiya-wappnet/phone_detection",
};

export const MODEL_FILES = {
  faceLandmarker:
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  attentionOnnx: "./models/attention_mlp.onnx",
  attentionNorm: "./models/attention_norm.json",
  // Object detector (YOLOv8n, ONNX, 320 px). `classes` maps the model's class ids to
  // the roles the proctor needs. Swap the file + ids to use a differently-trained model.
  detectorOnnx: "./models/detector_yolov8n.onnx",
  detectorClasses: { person: 0, phone: 67 },
};

// Detection / event thresholds
export const THRESHOLDS = {
  phoneConf: 0.40,          // YOLO confidence to accept a phone box
  personConf: 0.50,         // YOLO confidence to accept a person box
  phoneIou: 0.45,           // NMS IoU
  phoneEveryMs: 350,        // run YOLO at most this often (CPU friendly)
  attentionEma: 0.35,       // EMA factor for attention probabilities (higher = faster)
  minEventMs: 1000,         // a state must persist this long before it becomes a violation
  endEventMs: 600,          // state must be back to normal this long to close the event
  maxFaces: 4,
};

// Integrity-score penalties
export const PENALTIES = {
  looking_away: { perEvent: 3, perSecondAfter: 3, perSecond: 0.5 },
  phone:        { perEvent: 15, perSecondAfter: 0, perSecond: 1.0 },
  multiple_faces: { perEvent: 10, perSecondAfter: 0, perSecond: 0.5 },
  no_face:      { perEvent: 5, perSecondAfter: 3, perSecond: 0.5 },
};

export const LABELS = {
  looking_away: { title: "Looking away", icon: "👀" },
  phone: { title: "Phone detected", icon: "📱" },
  multiple_faces: { title: "Multiple persons", icon: "👥" },
  no_face: { title: "Face missing", icon: "🚫" },
};
