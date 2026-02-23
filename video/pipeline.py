"""Pipeline de análise de vídeo: leitura, YOLOv8-pose e heurísticas."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import config
from video.heuristics import compute_frame_indicators

# Fonte embutida no projeto (UTF-8/acentos); igual em qualquer SO
_BUNDLED_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"

REASON_TO_LABEL = {
    "head_lowered": "Cabeça baixa",
    "hands_near_face": "Mãos no rosto",
    "arms_defensive": "Braços defensivos",
    "closed_posture": "Postura fechada",
    "arms_raised": "Braço(s) levantado(s)",
    "shoulders_contracted": "Ombros contraídos",
}


def _load_model():
    return YOLO(config.YOLO_POSE_MODEL)


def _frame_to_keypoints_and_bbox(model, frame, device: str):
    """
    Executa YOLOv8-pose no frame.
    Retorna (keypoints, bbox_xyxy) da primeira pessoa; bbox é [x1,y1,x2,y2] ou None.
    """
    results = model(frame, device=device, verbose=False)
    if not results or len(results) == 0:
        return None, None
    r = results[0]
    if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
        return None, None
    kpts = r.keypoints.data[0].cpu().numpy()
    bbox = None
    if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes.xyxy) > 0:
        bbox = r.boxes.xyxy[0].cpu().numpy().tolist()
    return kpts, bbox


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
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = _load_model()
    segments: list[dict[str, Any]] = []
    frame_annotations: list[dict[str, Any]] = []
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
        kpts, bbox = _frame_to_keypoints_and_bbox(model, frame, device)
        indicators = compute_frame_indicators(kpts)
        sampled += 1
        if indicators["discomfort"]:
            discomfort_count += 1
            reasons: list[str] = indicators["reasons"]
            if bbox is not None:
                labels = [REASON_TO_LABEL.get(reason, reason) for reason in reasons]
                label = ", ".join(labels)
                frame_annotations.append(
                    {
                        "frame_index": frame_idx,
                        "time": round(t, 2),
                        "bbox": [round(float(x), 1) for x in bbox],
                        "reasons": list(reasons),
                        "label": label,
                    }
                )
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
        "video_width": video_width,
        "video_height": video_height,
        "video_sample_every_n_frames": sample_every_n,
        "sampled_frames": sampled,
        "discomfort_frames": discomfort_count,
        "segments": segments,
        "frame_annotations": frame_annotations,
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


def _find_utf8_font(size: int = 20):
    """Retorna ImageFont que suporta UTF-8 (fonte do projeto) ou None."""
    if _BUNDLED_FONT_PATH.is_file():
        try:
            return ImageFont.truetype(str(_BUNDLED_FONT_PATH), size)
        except (OSError, IOError):
            pass
    return None


def _draw_label_pillow(
    text: str,
    color_bgr: tuple[int, int, int],
    font_scale: float,
) -> tuple[np.ndarray, int, int] | None:
    """
    Desenha texto com acentos (UTF-8) usando Pillow.
    Retorna (array BGR, largura, altura) ou None para fallback em cv2.putText.
    """
    if not text.strip():
        return None
    size = max(12, int(22 * font_scale))
    font = _find_utf8_font(size)
    if font is None:
        return None
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    try:
        img = Image.new("RGB", (1, 1), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = int(bbox[2] - bbox[0] + 4)
        h = int(bbox[3] - bbox[1] + 4)
        img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), text, font=font, fill=color_rgb)
        arr = np.array(img)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr_bgr, w, h
    except Exception:
        return None


def _get_active_annotation(
    frame_idx: int,
    sorted_annotations: list[dict[str, Any]],
    sample_every_n: int,
) -> dict[str, Any] | None:
    """Retorna a anotação ativa para frame_idx (persiste por sample_every_n frames)."""
    for ann in reversed(sorted_annotations):
        fi = ann["frame_index"]
        if fi <= frame_idx and (frame_idx - fi) < sample_every_n:
            return ann
    return None


def render_annotated_video(  # noqa: C901
    video_path: str, report: dict, output_path: str
) -> None:
    """
    Gera vídeo com caixas e rótulos (UTF-8 com Pillow); persiste anotação por
    video_sample_every_n frames. Grava em temp e usa MoviePy para muxar áudio.
    """
    frame_annotations = report.get("frame_annotations")
    if not frame_annotations:
        raise ValueError(
            "Relatório sem frame_annotations; execute antes o comando de análise."
        )
    sample_every_n = report.get(
        "video_sample_every_n_frames", config.VIDEO_SAMPLE_EVERY_N_FRAMES
    )
    sorted_annotations = sorted(frame_annotations, key=lambda a: a["frame_index"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    except Exception:
        os.unlink(temp_path)
        raise

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ann = _get_active_annotation(frame_idx, sorted_annotations, sample_every_n)
        if ann is not None:
            bbox = ann["bbox"]
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                config.ANNOTATION_BOX_COLOR,
                config.ANNOTATION_BOX_THICKNESS,
            )
            label = ann.get("label", "")
            text_y = max(y1 - 10, 25)
            patch_result = _draw_label_pillow(
                label,
                config.ANNOTATION_TEXT_COLOR,
                config.ANNOTATION_FONT_SCALE,
            )
            if patch_result is not None:
                patch, pw, ph = patch_result
                fy1 = max(0, text_y)
                fy2 = min(frame.shape[0], text_y + ph)
                fx1 = max(0, x1)
                fx2 = min(frame.shape[1], x1 + pw)
                ph_act = fy2 - fy1
                pw_act = fx2 - fx1
                if ph_act > 0 and pw_act > 0:
                    frame[fy1:fy2, fx1:fx2] = patch[:ph_act, :pw_act]
            else:
                cv2.putText(
                    frame,
                    label,
                    (x1, text_y),
                    config.ANNOTATION_FONT,
                    config.ANNOTATION_FONT_SCALE,
                    config.ANNOTATION_TEXT_COLOR,
                    config.ANNOTATION_TEXT_THICKNESS,
                )
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    try:
        clip_annotated = VideoFileClip(temp_path)
        clip_original = VideoFileClip(str(video_path))
        try:
            if clip_original.audio is not None:
                clip_out = clip_annotated.set_audio(clip_original.audio)
            else:
                clip_out = clip_annotated
            clip_out.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None,
            )
        finally:
            clip_annotated.close()
            clip_original.close()
    except Exception:
        shutil.copy2(temp_path, output_path)
    finally:
        if os.path.isfile(temp_path):
            os.unlink(temp_path)
