"""CLI: multimodal analysis (YOLOv8-pose+faster-whisper)
and annotated video generation."""

import argparse
import json
import sys
from pathlib import Path

import report
from audio.pipeline import run_audio_pipeline
from config import OUTPUT_DIR
from video.pipeline import render_annotated_video, run_video_pipeline


def _cmd_run(args: argparse.Namespace) -> int:
    """Run subcommand: multimodal analysis and reports."""
    video_path = Path(args.video_path)
    report_video = None
    report_audio = None

    try:
        output_dir_path = report.ensure_output_dir(args.output_dir)

        if not args.skip_video:
            print("Running video pipeline (YOLOv8-pose + heuristics)...")

            report_video = run_video_pipeline(str(video_path))

            report_path = report.write_json_report(
                report_video,
                output_dir_path,
                f"video_report_{video_path.stem}.json",
            )
            print(f"Video report: {report_path}")

            num_segments = report_video.get("summary", {}).get(
                "total_discomfort_segments", 0
            )
            print(f"  Segments with indicators: {num_segments}")

        if not args.skip_audio:
            print("Running audio pipeline (MoviePy + faster-whisper + analysis)...")

            report_audio = run_audio_pipeline(
                str(video_path), output_dir=str(output_dir_path)
            )

            report_path = report.write_json_report(
                report_audio,
                output_dir_path,
                f"audio_report_{video_path.stem}.json",
            )
            print(f"Audio report: {report_path}")

            summary = report_audio.get("analysis", {}).get("summary", "")
            print(f"  Summary: {summary}")

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
        print(f"Consolidated report: {consolidated_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_annotate_video(args: argparse.Namespace) -> int:
    """Annotate-video subcommand: generate video with boxes and labels."""
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    output_dir_path = report.ensure_output_dir(args.output_dir)
    report_path = (
        Path(args.report)
        if args.report
        else output_dir_path / f"video_report_{video_path.stem}.json"
    )
    if not report_path.exists():
        print(f"Error: report not found: {report_path}", file=sys.stderr)
        return 1

    try:
        with open(report_path, encoding="utf-8") as f:
            report_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading report: {e}", file=sys.stderr)
        return 1

    output_video_path = (
        Path(args.output_video)
        if args.output_video
        else output_dir_path / f"video_annotated_{video_path.stem}.mp4"
    )

    try:
        render_annotated_video(str(video_path), report_data, str(output_video_path))
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Annotated video: {output_video_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multimodal analysis of consultation videos "
        "(medical, psychological, physiotherapy)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run analysis and generate reports")
    run_parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file (e.g. data/demo_consultation.mp4)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for reports",
    )
    run_parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Run video pipeline only",
    )
    run_parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Run audio pipeline only",
    )
    run_parser.set_defaults(func=_cmd_run)

    annotate_parser = subparsers.add_parser(
        "annotate-video",
        help="Generate video with boxes and labels from video report",
    )
    annotate_parser.add_argument(
        "video_path",
        type=str,
        help="Path to original video",
    )
    annotate_parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Report and output video directory (if --output-video omitted)",
    )
    annotate_parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Report JSON (default: output_dir/video_report_<stem>.json)",
    )
    annotate_parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video (default: output_dir/video_annotated_<stem>.mp4)",
    )
    annotate_parser.set_defaults(func=_cmd_annotate_video)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
