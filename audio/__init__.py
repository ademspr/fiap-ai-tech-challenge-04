"""Pipeline de áudio: extração, transcrição (faster-whisper) e análise do texto."""

from audio.pipeline import (
    analyze_transcript,
    extract_audio,
    run_audio_pipeline,
    transcribe_audio,
)

__all__ = [
    "run_audio_pipeline",
    "extract_audio",
    "transcribe_audio",
    "analyze_transcript",
]
