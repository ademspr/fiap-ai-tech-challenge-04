"""Pipeline de áudio: extração do vídeo, transcrição (faster-whisper), análise."""

from pathlib import Path

from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip

import config
from audio.analyzer import analyze_transcript_text


def extract_audio(video_path: str, audio_path: str | None = None) -> str:
    """
    Extrai a trilha de áudio do vídeo em WAV.
    Retorna o caminho do arquivo de áudio gerado.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {vpath}")
    if audio_path is None:
        out_dir = Path(config.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_audio = out_dir / f"{vpath.stem}_audio.wav"
    else:
        out_audio = Path(audio_path)
        out_audio.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(vpath))
    if clip.audio is None:
        clip.close()
        raise ValueError("O vídeo não possui trilha de áudio.")
    clip.audio.write_audiofile(
        str(out_audio), codec="pcm_s16le", verbose=False, logger=None
    )
    clip.close()
    return str(out_audio)


def transcribe_audio(audio_path: str) -> str:
    """
    Transcreve o áudio com faster-whisper (modelo tiny/base, CPU).
    Retorna o texto transcrito.
    """
    model = WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    segments, _ = model.transcribe(audio_path, language="pt")
    parts = [s.text.strip() for s in segments if s.text.strip()]
    return " ".join(parts).strip()


def analyze_transcript(transcript: str) -> dict:
    """Analisa o texto transcrito com regras/heurísticas. Retorna dict."""
    return analyze_transcript_text(transcript)


def run_audio_pipeline(
    video_path: str,
    output_dir: str | None = None,
) -> dict:
    """
    Executa o pipeline de áudio: extrai áudio, transcreve e analisa.
    Retorna dict com transcript e analysis; relatório JSON e diretório
    de saída ficam a cargo do chamador.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {vpath}")

    out_dir = Path(output_dir or config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = extract_audio(str(vpath), str(out_dir / f"{vpath.stem}_audio.wav"))
    transcript = transcribe_audio(wav_path)

    return {
        "video_path": str(vpath),
        "audio_path": wav_path,
        "transcript": transcript,
        "analysis": analyze_transcript(transcript),
    }
