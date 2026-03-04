"""Audio pipeline: extract from video, transcribe (faster-whisper), analyze."""

from pathlib import Path

from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip

from audio.analyzer import analyze_transcript_text
from config import (
    OUTPUT_DIR,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)


def extract_audio(video_path: str, audio_path: str | None = None) -> str:
    """Extract audio track from video to WAV. Return path to generated file."""
    video_path_resolved = Path(video_path)
    if not video_path_resolved.exists():
        raise FileNotFoundError(f"Video not found: {video_path_resolved}")
    if audio_path is None:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_audio_path = output_dir / f"{video_path_resolved.stem}_audio.wav"
    else:
        output_audio_path = Path(audio_path)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(video_path_resolved))
    if clip.audio is None:
        clip.close()
        raise ValueError("Video has no audio track.")
    clip.audio.write_audiofile(
        str(output_audio_path), codec="pcm_s16le", verbose=False, logger=None
    )
    clip.close()
    return str(output_audio_path)


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio with faster-whisper. Return transcribed text."""
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    segments, _ = model.transcribe(audio_path, language=WHISPER_LANGUAGE)
    parts = [s.text.strip() for s in segments if s.text.strip()]
    return " ".join(parts).strip()


def analyze_transcript(transcript: str) -> dict:
    """Analyze transcript with rules/heuristics. Return dict."""
    return analyze_transcript_text(transcript)


def run_audio_pipeline(
    video_path: str,
    output_dir: str | None = None,
) -> dict:
    """Run audio pipeline: extract, transcribe, analyze.
    Return dict with transcript and analysis."""
    video_path_resolved = Path(video_path)
    if not video_path_resolved.exists():
        raise FileNotFoundError(f"File not found: {video_path_resolved}")

    output_dir_resolved = Path(output_dir or OUTPUT_DIR)
    output_dir_resolved.mkdir(parents=True, exist_ok=True)

    wav_path = extract_audio(
        str(video_path_resolved),
        str(output_dir_resolved / f"{video_path_resolved.stem}_audio.wav"),
    )
    transcript = transcribe_audio(wav_path)

    return {
        "video_path": str(video_path_resolved),
        "audio_path": wav_path,
        "transcript": transcript,
        "analysis": analyze_transcript(transcript),
    }
