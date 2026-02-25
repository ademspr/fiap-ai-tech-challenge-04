# fiap-ai-tech-challenge-04

Multimodal analysis of consultation and healthcare videos: **video analysis** (YOLOv8-pose + body language heuristics) and **voice recording processing** (transcription with faster-whisper and text analysis). Applicable to medical, psychological, physiotherapy consultations, births, surgeries, and postpartum sessions. Runs on a personal computer without GPU or cloud services.

## Requirements

- Python 3.12
- Poetry

## Installation

```bash
poetry install
```

On first run, YOLOv8 (`yolov8n-pose.pt`) and faster-whisper (`tiny`) are downloaded automatically. After that, the pipeline runs **offline**.

## Usage

Place a consultation video in `data/` (e.g. `data/teste.mp4`). Two subcommands:

**Analysis (JSON reports):**

```bash
poetry run python main.py run data/teste.mp4
```

Options: `--output-dir DIR`, `--skip-audio`, `--skip-video`.

**Annotated video (boxes and labels, with audio and correct timing):**

```bash
poetry run python main.py annotate-video data/teste.mp4
```

The annotated video is written to `output/video_annotated_<name>.mp4`. Original audio and duration metadata are included via MoviePy; annotation text (e.g. "Hands near face") is drawn with Pillow. No manual ffmpeg required.

Reports (JSON) are written to `output/`:

- `video_report_<name>.json` — Pose analysis and discomfort/defensive indicators (16 body language patterns).
- `audio_report_<name>.json` — English transcription and text analysis (anxiety, hesitation, discomfort, postpartum depression, domestic violence, hormonal fatigue).
- `consolidated_<name>.json` — Summary of both pipelines.

## Sample video

Use **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** (install with `pip install yt-dlp` or `brew install yt-dlp`) to download a demo video:

```bash
yt-dlp -f mp4 -o data/teste.mp4 "https://www.youtube.com/watch?v=snG5323GGQ4"
```

If you already downloaded elsewhere, move to `data/` and rename to `teste.mp4`:

```bash
cd data
mv "DownloadedFileName.mp4" teste.mp4
```

Use a short video (30 s to 2 min) of a consultation scene (doctor + patient) with **English** speech (transcription and audio analysis use English). Free sources: **Pexels** ("Doctor Talking to a Patient Sitting on Bed"), **Mixkit** ("female doctor consultation").

Then run:

```bash
poetry run python main.py run data/teste.mp4
poetry run python main.py annotate-video data/teste.mp4
```

## Multimodal flow

1. **Video:** OpenCV reads video → YOLOv8-pose (nano/small, CPU) extracts keypoints → heuristics detect 16 body language patterns (head lowered, hands near face, defensive arms, arms crossed, hands on hips, shoulders raised, hand on chest, hand to neck, leaning, legs closed, body turned away, hands clasped, etc.) → report with segments and counts.
2. **Audio:** MoviePy extracts audio to WAV → faster-whisper (tiny/base, CPU) transcribes in English → term-list analysis (anxiety, hesitation, discomfort, postpartum depression, domestic violence, hormonal fatigue) → report with transcript and indicators.

## Configuration

Edit `config.py` to change:

- `YOLO_POSE_MODEL` — `yolov8n-pose.pt` (nano) or `yolov8s-pose.pt` (small).
- `VIDEO_SAMPLE_EVERY_N_FRAMES` — Sample 1 every N frames (default 5; lower = more patterns, more CPU).
- `WHISPER_MODEL_SIZE` — `tiny` or `base` for CPU.
- `WHISPER_LANGUAGE` — `en` (English) or `pt` (Portuguese).

## Project structure

- `main.py` — CLI orchestrator.
- `config.py` — Configuration (models, paths).
- `video/` — Video pipeline (YOLOv8-pose, heuristics, report, annotated video with Pillow and MoviePy).
- `audio/` — Audio pipeline (MoviePy, faster-whisper, text analysis, report).
- `data/` — Demo video(s).
- `output/` — Generated reports (created automatically).
