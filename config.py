"""Configuração do projeto: modelos, thresholds e caminhos."""

# YOLOv8 Pose: variante leve para CPU (nano ou small)
YOLO_POSE_MODEL = (
    "yolov8n-pose.pt"  # nano = mais rápido; use yolov8s-pose.pt para mais precisão
)
YOLO_DEVICE = "cpu"

# Processamento de vídeo: amostrar 1 a cada N frames para acelerar em CPU
VIDEO_SAMPLE_EVERY_N_FRAMES = 5

# faster-whisper: modelo pequeno para CPU (tiny ou base)
WHISPER_MODEL_SIZE = "tiny"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"  # int8 para CPU

# Pastas
DATA_DIR = "data"
OUTPUT_DIR = "output"
