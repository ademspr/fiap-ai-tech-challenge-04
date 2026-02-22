"""Pipeline de análise de vídeo: leitura, YOLOv8-pose e heurísticas."""

from pathlib import Path
from typing import Any, cast

import cv2
from ultralytics import YOLO

import config
from video.heuristics import compute_frame_indicators


def _load_model():
    return YOLO(config.YOLO_POSE_MODEL)


def _frame_to_keypoints(model, frame, device: str):
    """Executa YOLOv8-pose no frame; retorna keypoints da primeira pessoa (x,y,conf)."""
    results = model(frame, device=device, verbose=False)
    if not results or len(results) == 0:
        return None
    r = results[0]
    if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
        return None
    # Primeira pessoa detectada: (17, 3) = x, y, confidence
    kpts = r.keypoints.data[0].cpu().numpy()
    return kpts


def analyze_video(  # noqa: C901
    video_path: str,
    sample_every_n: int | None = None,
    device: str | None = None,
):
    """
    Analisa vídeo com YOLOv8-pose e heurísticas de desconforto/medo/defensivo.
    Retorna dict: fps, total_frames, sampled_frames, segments, summary.
    """
    sample_every_n = sample_every_n or config.VIDEO_SAMPLE_EVERY_N_FRAMES
    device = device or config.YOLO_DEVICE
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise OSError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = _load_model()
    segments: list[dict[str, Any]] = []
    current_segment: dict[str, Any] | None = None
    frame_idx = 0
    sampled = 0
    discomfort_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n != 0:
            frame_idx += 1
            continue
        t = frame_idx / fps
        kpts = _frame_to_keypoints(model, frame, device)
        indicators = compute_frame_indicators(kpts)
        sampled += 1
        if indicators["discomfort"]:
            discomfort_count += 1
            reasons: list[str] = indicators["reasons"]
            if current_segment is None:
                current_segment = {
                    "start_time": t,
                    "end_time": t,
                    "reasons": list(reasons),
                }
            else:
                current_segment["end_time"] = t
                reas = cast(list[str], current_segment["reasons"])
                for r in reasons:
                    if r not in reas:
                        reas.append(r)
        else:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
        frame_idx += 1
    cap.release()
    if current_segment is not None:
        segments.append(current_segment)

    return {
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames": sampled,
        "discomfort_frames": discomfort_count,
        "segments": segments,
        "summary": {
            "total_discomfort_segments": len(segments),
            "discomfort_ratio": round(discomfort_count / sampled, 4) if sampled else 0,
        },
    }


def run_video_pipeline(video_path: str) -> dict:
    """
    Executa o pipeline de vídeo (análise apenas).
    Retorna o resultado da análise; relatório JSON fica a cargo do chamador.
    """
    result: dict = analyze_video(video_path)
    return result
