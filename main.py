"""
Orquestrador: analisa vídeo de consulta (YOLOv8-pose + áudio com faster-whisper)
e gera relatórios consolidados.
"""

import argparse
import sys
from pathlib import Path

import config
import report
from audio.pipeline import run_audio_pipeline
from video.pipeline import run_video_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Análise multimodal de vídeo de consulta (saúde da mulher)."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Caminho do arquivo de vídeo (ex.: data/demo_consultation.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Pasta para relatórios de saída",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Executar apenas o pipeline de vídeo",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Executar apenas o pipeline de áudio",
    )
    args = parser.parse_args()
    video_path = Path(args.video_path)
    report_video = None
    report_audio = None

    try:
        output_dir_path = report.ensure_output_dir(args.output_dir)

        if not args.skip_video:
            print("Executando pipeline de vídeo (YOLOv8-pose + heurísticas)...")

            report_video = run_video_pipeline(str(video_path))

            report_path = report.write_json_report(
                report_video,
                output_dir_path,
                f"video_report_{video_path.stem}.json",
            )
            print(f"Relatório de vídeo: {report_path}")

            n = report_video.get("summary", {}).get("total_discomfort_segments", 0)
            print(f"  Segmentos com indicadores: {n}")

        if not args.skip_audio:
            print(
                "Executando pipeline de áudio (MoviePy + faster-whisper + análise)..."
            )

            report_audio = run_audio_pipeline(
                str(video_path), output_dir=str(output_dir_path)
            )

            report_path = report.write_json_report(
                report_audio,
                output_dir_path,
                f"audio_report_{video_path.stem}.json",
            )
            print(f"Relatório de áudio: {report_path}")

            summary = report_audio.get("analysis", {}).get("summary", "")
            print(f"  Resumo: {summary}")

        consolidated = {
            "video_path": str(video_path),
            "output_dir": str(output_dir_path),
            "video_report": report_video,
            "audio_report": report_audio,
        }
        consolidated_path = report.write_json_report(
            consolidated,
            output_dir_path,
            f"consolidated_{video_path.stem}.json",
        )
        print(f"Relatório consolidado: {consolidated_path}")
        return 0
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
