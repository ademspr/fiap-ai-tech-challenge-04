"""Configuração do projeto: modelos, thresholds e caminhos."""

# YOLOv8 Pose: variante leve para CPU (nano ou small)
YOLO_POSE_MODEL = (
    "yolov8n-pose.pt"  # nano = mais rápido; use yolov8s-pose.pt para mais precisão
)
YOLO_DEVICE = "cpu"

# Processamento de vídeo: amostrar 1 a cada N frames (3 = mais padrões, mais CPU)
VIDEO_SAMPLE_EVERY_N_FRAMES = 3

# Heurísticas: distância máxima (px) pulso–nariz para "mãos no rosto"
HEURISTIC_HANDS_NEAR_FACE_MAX_DIST_PX = 120

# faster-whisper: modelo pequeno para CPU (tiny ou base)
WHISPER_MODEL_SIZE = "tiny"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"  # int8 para CPU

# Pastas
DATA_DIR = "data"
OUTPUT_DIR = "output"

# Vídeo anotado: cores e fontes para overlay (OpenCV BGR)
ANNOTATION_BOX_COLOR = (0, 255, 0)  # verde
ANNOTATION_BOX_THICKNESS = 2
ANNOTATION_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
ANNOTATION_FONT_SCALE = 0.6
ANNOTATION_TEXT_COLOR = (255, 255, 255)
ANNOTATION_TEXT_THICKNESS = 2
