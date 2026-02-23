"""
Orquestrador: analisa vídeo de consulta (YOLOv8-pose + áudio com faster-whisper)
e gera relatórios consolidados. Subcomando annotate-video gera vídeo com overlay.
"""

import argparse
import json
import sys
from pathlib import Path

import config
import report
from audio.pipeline import run_audio_pipeline
from video.pipeline import render_annotated_video, run_video_pipeline


def _cmd_run(args: argparse.Namespace) -> int:
    """Subcomando run: análise multimodal e relatórios."""
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


def _cmd_annotate_video(args: argparse.Namespace) -> int:
    """Subcomando annotate-video: gera vídeo com caixas e rótulos."""
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Erro: vídeo não encontrado: {video_path}", file=sys.stderr)
        return 1

    output_dir_path = report.ensure_output_dir(args.output_dir)
    report_path = (
        Path(args.report)
        if args.report
        else output_dir_path / f"video_report_{video_path.stem}.json"
    )
    if not report_path.exists():
        print(f"Erro: relatório não encontrado: {report_path}", file=sys.stderr)
        return 1

    try:
        with open(report_path, encoding="utf-8") as f:
            report_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Erro ao ler relatório: {e}", file=sys.stderr)
        return 1

    output_video_path = (
        Path(args.output_video)
        if args.output_video
        else output_dir_path / f"video_annotated_{video_path.stem}.mp4"
    )

    try:
        render_annotated_video(str(video_path), report_data, str(output_video_path))
    except (ValueError, OSError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    print(f"Vídeo anotado: {output_video_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Análise multimodal de vídeo de consulta (saúde da mulher)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Executar análise e gerar relatórios"
    )
    run_parser.add_argument(
        "video_path",
        type=str,
        help="Caminho do arquivo de vídeo (ex.: data/demo_consultation.mp4)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Pasta para relatórios de saída",
    )
    run_parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Executar apenas o pipeline de vídeo",
    )
    run_parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Executar apenas o pipeline de áudio",
    )
    run_parser.set_defaults(func=_cmd_run)

    annotate_parser = subparsers.add_parser(
        "annotate-video",
        help="Gerar vídeo com caixas e rótulos a partir do relatório de vídeo",
    )
    annotate_parser.add_argument(
        "video_path",
        type=str,
        help="Caminho do vídeo original",
    )
    annotate_parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Pasta do relatório e do vídeo de saída (se --output-video omitido)",
    )
    annotate_parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="JSON do relatório (default: output_dir/video_report_<stem>.json)",
    )
    annotate_parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Vídeo de saída (default: output_dir/video_annotated_<stem>.mp4)",
    )
    annotate_parser.set_defaults(func=_cmd_annotate_video)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
