# fiap-ai-tech-challenge-04

Python script that analyzes a video and overlays:
- **Face detection** (MediaPipe BlazeFace)
- **Emotion recognition** (DeepFace)
- **Activity detection** (TensorFlow)
- **Anomaly detection** (pose/gesture)

## How to run

1. Put your video at **`videos/input.mp4`**.
2. From the project root, run:

```bash
python main.py
```

Models are downloaded automatically on first run (into `models/`).

## Output

- **Annotated video:** `analysis_results/video_analisado.mp4`
- **Summary:** printed in the terminal (duration, faces, top emotions/activities, anomaly count)
