"""Video analysis pipeline: YOLOv8-pose and body language heuristics."""

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

from config import (
    ANNOTATION_BBOX_PADDING_PX,
    ANNOTATION_BOX_COLOR,
    ANNOTATION_BOX_THICKNESS,
    ANNOTATION_DISPLAY_FRAMES,
    ANNOTATION_FONT,
    ANNOTATION_FONT_SCALE,
    ANNOTATION_TEXT_COLOR,
    ANNOTATION_TEXT_THICKNESS,
    DETECTION_CONFIDENCE_THRESHOLD,
    VIDEO_SAMPLE_EVERY_N_FRAMES,
    YOLO_DEVICE,
    YOLO_POSE_MODEL,
)
from video.heuristics import MIN_CONF, compute_frame_indicators

_BUNDLED_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"

REASON_TO_LABEL = {
    "head_lowered": "Head lowered",
    "hands_near_face": "Hands near face",
    "arms_defensive": "Defensive arms",
    "closed_posture": "Closed posture",
    "arms_raised": "Arms raised",
    "shoulders_contracted": "Shoulders contracted",
    "arms_crossed": "Arms crossed",
    "hands_on_hips": "Hands on hips",
    "shoulders_raised": "Shoulders raised",
    "hand_on_chest": "Hand on chest",
    "hand_to_neck": "Hand to neck",
    "leaning_back": "Leaning back",
    "leaning_forward": "Leaning forward",
    "legs_closed": "Legs closed",
    "body_turned_away": "Body turned away",
    "hands_clasped": "Hands clasped",
}


def _load_model():
    return YOLO(YOLO_POSE_MODEL)


def _frame_to_all_persons(model, frame, device: str):
    """Return list of (keypoints, bounding_box) per person above threshold"""
    threshold = DETECTION_CONFIDENCE_THRESHOLD
    results = model(frame, device=device, verbose=False)
    if not results or len(results) == 0:
        return []
    result = results[0]
    if (
        result.keypoints is None
        or result.keypoints.data is None
        or len(result.keypoints.data) == 0
    ):
        return []
    persons = []
    num_keypoints = len(result.keypoints.data)
    if (
        result.boxes is not None
        and hasattr(result.boxes, "conf")
        and result.boxes.conf is not None
    ):
        confidences = result.boxes.conf.cpu().numpy()
    else:
        confidences = np.ones(num_keypoints)
    for i in range(num_keypoints):
        confidence = float(confidences[i]) if i < len(confidences) else 1.0
        if confidence < threshold:
            continue
        keypoints = result.keypoints.data[i].cpu().numpy()
        bounding_box = None
        if (
            result.boxes is not None
            and result.boxes.xyxy is not None
            and i < len(result.boxes.xyxy)
        ):
            bounding_box = result.boxes.xyxy[i].cpu().numpy().tolist()
        persons.append((keypoints, bounding_box))
    return persons


def _keypoints_to_bbox(
    keypoints: np.ndarray,
    padding: int | None = None,
    upper_body_only: bool = True,
) -> list[float] | None:
    """Compute tight bounding box from keypoints (upper body by default)."""
    pad = padding if padding is not None else ANNOTATION_BBOX_PADDING_PX
    if keypoints is None or keypoints.shape[0] < 13:
        return None
    end_idx = 13 if upper_body_only else keypoints.shape[0]
    pts = []
    for i in range(end_idx):
        if i >= keypoints.shape[0]:
            break
        row = keypoints[i]
        if len(row) >= 3 and float(row[2]) >= MIN_CONF:
            pts.append((float(row[0]), float(row[1])))
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = max(xs) + pad
    y2 = max(ys) + pad
    return [x1, y1, x2, y2]


def analyze_video(  # noqa: C901
    video_path: str,
    sample_every_n: int | None = None,
    device: str | None = None,
):
    """Analyze video with YOLOv8-pose and discomfort/defensive heuristics."""
    sample_every_n = sample_every_n or VIDEO_SAMPLE_EVERY_N_FRAMES
    device = device or YOLO_DEVICE
    video_path_resolved = Path(video_path)
    if not video_path_resolved.exists():
        raise FileNotFoundError(f"File not found: {video_path_resolved}")

    video_capture = cv2.VideoCapture(str(video_path_resolved))
    if not video_capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = video_capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = _load_model()
    segments: list[dict[str, Any]] = []
    frame_annotations: list[dict[str, Any]] = []
    current_segment: dict[str, Any] | None = None
    frame_idx = 0
    sampled = 0
    discomfort_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_idx % sample_every_n != 0:
            frame_idx += 1
            continue
        timestamp_sec = frame_idx / fps
        persons = _frame_to_all_persons(model, frame, device)
        sampled += 1
        frame_has_discomfort = False
        all_reasons: list[str] = []
        for keypoints, bounding_box in ((p[0], p[1]) for p in persons):
            indicators = compute_frame_indicators(keypoints)
            if indicators["discomfort"]:
                frame_has_discomfort = True
                reasons: list[str] = indicators["reasons"]
                all_reasons.extend(r for r in reasons if r not in all_reasons)
                annotation_bbox = (
                    _keypoints_to_bbox(keypoints) if keypoints is not None else None
                )
                if annotation_bbox is None:
                    annotation_bbox = bounding_box
                if annotation_bbox is not None:
                    labels = [REASON_TO_LABEL.get(reason, reason) for reason in reasons]
                    label = ", ".join(labels)
                    frame_annotations.append(
                        {
                            "frame_index": frame_idx,
                            "time": round(timestamp_sec, 2),
                            "bbox": [round(float(x), 1) for x in annotation_bbox],
                            "reasons": list(reasons),
                            "label": label,
                        }
                    )
        if frame_has_discomfort:
            discomfort_count += 1
            if current_segment is None:
                current_segment = {
                    "start_time": timestamp_sec,
                    "end_time": timestamp_sec,
                    "reasons": list(all_reasons),
                }
            else:
                current_segment["end_time"] = timestamp_sec
                segment_reasons = cast(list[str], current_segment["reasons"])
                for reason in all_reasons:
                    if reason not in segment_reasons:
                        segment_reasons.append(reason)
        else:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
        frame_idx += 1
    video_capture.release()
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
    """Run video pipeline (analysis only). Return analysis result dict."""
    result: dict = analyze_video(video_path)
    return result


def _find_utf8_font(size: int = 20):
    """Return ImageFont for UTF-8 (bundled) or None."""
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
    """Draw UTF-8 text with Pillow. Return (BGR array, width, height) or None."""
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


def _get_active_annotations(
    frame_idx: int,
    sorted_annotations: list[dict[str, Any]],
    display_frames: int,
) -> list[dict[str, Any]]:
    """Return all annotations active for frame_idx (persist for display_frames)."""
    best_frame_index: int | None = None
    for annotation in reversed(sorted_annotations):
        frame_index = annotation["frame_index"]
        if frame_index <= frame_idx and (frame_idx - frame_index) < display_frames:
            best_frame_index = frame_index
            break
    if best_frame_index is None:
        return []
    return [a for a in sorted_annotations if a["frame_index"] == best_frame_index]


def render_annotated_video(  # noqa: C901
    video_path: str, report: dict, output_path: str
) -> None:
    """Generate video with boxes and labels (via Pillow). Use MoviePy for audio."""
    frame_annotations = report.get("frame_annotations")
    if not frame_annotations:
        raise ValueError("Report missing frame_annotations; run analysis first.")
    display_frames = ANNOTATION_DISPLAY_FRAMES
    sorted_annotations = sorted(frame_annotations, key=lambda a: a["frame_index"])

    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = video_capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        ret, frame = video_capture.read()
        if not ret:
            break
        annotations = _get_active_annotations(
            frame_idx, sorted_annotations, display_frames
        )
        for annotation in annotations:
            bounding_box = annotation["bbox"]
            x1, y1 = int(bounding_box[0]), int(bounding_box[1])
            x2, y2 = int(bounding_box[2]), int(bounding_box[3])
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                ANNOTATION_BOX_COLOR,
                ANNOTATION_BOX_THICKNESS,
            )
            label = annotation.get("label", "")
            text_y = max(y1 - 10, 25)
            patch_result = _draw_label_pillow(
                label,
                ANNOTATION_TEXT_COLOR,
                ANNOTATION_FONT_SCALE,
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
                    ANNOTATION_FONT,
                    ANNOTATION_FONT_SCALE,
                    ANNOTATION_TEXT_COLOR,
                    ANNOTATION_TEXT_THICKNESS,
                )
        out.write(frame)
        frame_idx += 1

    video_capture.release()
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
