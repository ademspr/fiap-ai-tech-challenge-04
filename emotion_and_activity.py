import time
import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
import os
import json
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import warnings
import urllib.request
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks

warnings.filterwarnings("ignore")

# Cores para emoções (BGR)
EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (255, 165, 0),
    "neutral": (128, 128, 128),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.txt")

# Classes COCO que representam pessoas/animais — ignoradas no detector
# de atividade (pessoas já são tratadas via face detection).
IGNORED_ACTIVITY_CLASSES = {1, 3, 17, 37, 43, 45, 46, 47, 59, 65, 74, 77, 78, 79, 80}

# Limiar de velocidade de landmarks (px/frame) para anomalia
# (gestos bruscos ou comportamentos atípicos).
ANOMALY_VELOCITY_THRESHOLD = 35.0


class VideoAnalyzer:
    """Analisa um vídeo: reconhecimento facial, emoções, atividades e anomalias."""

    def __init__(self, video_path: str, output_dir: str = "analysis_results"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Video state
        self.cap = None
        self.fps = 30
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0

        # Analysis data
        self.frame_analysis: list[dict] = []
        self.emotion_timeline: list[dict] = []
        self.activity_timeline: list[dict] = []
        self.anomaly_timeline: list[dict] = []

        # Load models ONCE at init
        self.face_detector = self._load_face_detector()
        self.pose = self._load_mediapipe_pose()
        self.activity_session, self.activity_tensors, self.activity_labels, self.activity_colors = (
            self._load_activity_model()
        )

        # Previous pose landmarks for anomaly detection
        self._prev_landmarks = None

    # ------------------------------------------------------------------
    # Model loading (executed once)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_model(filename: str, url: str) -> str:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"⬇️  Baixando {filename}...")
            urllib.request.urlretrieve(url, path)
        return path

    def _load_face_detector(self) -> vision.FaceDetector:
        model_path = self._ensure_model(
            "blaze_face_short_range.tflite",
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        )
        options = vision.FaceDetectorOptions(
            base_options=tasks.BaseOptions(model_asset_path=model_path),
        )
        return vision.FaceDetector.create_from_options(options)

    @staticmethod
    def _load_mediapipe_pose():
        try:
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            return None

    def _load_activity_model(self):
        model_path = self._ensure_model(
            "frozen_inference_graph.pb",
            "https://github.com/visiongeeklabs/human-activity-detection/"
            "releases/download/v0.1.0/frozen_inference_graph.pb",
        )

        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines()]

        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, "rb") as fid:
                graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name="")

            sess = tf.compat.v1.Session(graph=graph)
            all_tensor_names = {
                output.name
                for op in tf.compat.v1.get_default_graph().get_operations()
                for output in op.outputs
            }
            tensor_dict = {}
            for key in ("num_detections", "detection_boxes", "detection_scores", "detection_classes"):
                tname = key + ":0"
                if tname in all_tensor_names:
                    tensor_dict[key] = graph.get_tensor_by_name(tname)

            image_tensor = graph.get_tensor_by_name("image_tensor:0")

        colors = np.random.RandomState(42).uniform(0, 255, size=(len(labels), 3))
        return sess, {**tensor_dict, "image_tensor": image_tensor}, labels, colors

    # ------------------------------------------------------------------
    # Face preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_face(face_region: np.ndarray) -> np.ndarray:
        """Normaliza iluminação, remove ruído e realça nitidez."""
        face = cv2.resize(face_region, (152, 152))

        if len(face.shape) == 3 and face.shape[2] == 3:
            for i in range(3):
                face[:, :, i] = cv2.equalizeHist(face[:, :, i])
        else:
            face = cv2.equalizeHist(face)

        face = cv2.GaussianBlur(face, (3, 3), 0)
        face = cv2.convertScaleAbs(face, alpha=1.2, beta=10)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        face = cv2.filter2D(face, -1, kernel)

        return cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ------------------------------------------------------------------
    # 1 + 2. Reconhecimento facial + Análise de expressões emocionais
    # ------------------------------------------------------------------

    def analyze_faces(self, frame: np.ndarray) -> list[dict]:
        """Detecta rostos e analisa emoções de cada rosto."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detections = self.face_detector.detect(mp_image).detections
        except Exception:
            return []

        results = []
        for det in detections:
            bb = det.bounding_box
            x, y, w, h = int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)

            # Clamp dentro dos limites do frame
            x, y = max(0, x), max(0, y)
            x2 = min(x + w, frame.shape[1])
            y2 = min(y + h, frame.shape[0])
            face_roi = frame[y:y2, x:x2]

            if face_roi.size == 0:
                continue

            try:
                preprocessed = self.preprocess_face(face_roi)
                analysis = DeepFace.analyze(
                    preprocessed,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                )
                analysis = analysis if isinstance(analysis, list) else [analysis]
                for a in analysis:
                    results.append(
                        {
                            "region": {"x": x, "y": y, "w": x2 - x, "h": y2 - y},
                            "emotion": a.get("dominant_emotion", "neutral"),
                            "emotions": a.get("emotion", {}),
                            "confidence": max(a.get("emotion", {}).values(), default=0),
                        }
                    )
            except Exception:
                results.append(
                    {
                        "region": {"x": x, "y": y, "w": x2 - x, "h": y2 - y},
                        "emotion": "unknown",
                        "emotions": {},
                        "confidence": 0,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # 3. Detecção de atividades
    # ------------------------------------------------------------------

    def analyze_activity(self, frame: np.ndarray) -> list[dict]:
        """Detecta atividades/objetos no frame."""
        tensors = self.activity_tensors
        frame_exp = np.expand_dims(frame, axis=0)

        feed = {tensors["image_tensor"]: frame_exp}
        fetch = {k: v for k, v in tensors.items() if k != "image_tensor"}

        try:
            out = self.activity_session.run(fetch, feed_dict=feed)
        except Exception:
            return []

        num = int(out["num_detections"][0])
        classes = out["detection_classes"][0].astype(np.uint8)
        boxes = out["detection_boxes"][0]
        scores = out["detection_scores"][0]

        fh, fw = frame.shape[:2]
        threshold = 0.5
        detected: list[dict] = []

        for i in range(num):
            cls = int(classes[i])
            if cls in IGNORED_ACTIVITY_CLASSES:
                continue
            if scores[i] < threshold:
                continue

            bbox = boxes[i].copy()
            bbox[0] *= fh
            bbox[1] *= fw
            bbox[2] *= fh
            bbox[3] *= fw

            idx = cls - 1
            label = self.activity_labels[idx] if idx < len(self.activity_labels) else f"class_{cls}"
            if label == "N/A":
                continue

            detected.append(
                {
                    "type": label,
                    "confidence": float(scores[i]),
                    "bbox": [int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])],
                    "color_idx": idx,
                }
            )
        return detected

    # ------------------------------------------------------------------
    # 4. Detecção de anomalias (movimentos bruscos / atípicos)
    # ------------------------------------------------------------------

    def detect_anomaly(self, frame: np.ndarray) -> dict | None:
        """Detecta movimentos anômalos via variação brusca de pose landmarks."""
        if self.pose is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            self._prev_landmarks = None
            return None

        fh, fw = frame.shape[:2]
        current = np.array(
            [(lm.x * fw, lm.y * fh) for lm in result.pose_landmarks.landmark]
        )

        anomaly = None
        if self._prev_landmarks is not None:
            velocities = np.linalg.norm(current - self._prev_landmarks, axis=1)
            max_vel = float(velocities.max())
            mean_vel = float(velocities.mean())

            if max_vel > ANOMALY_VELOCITY_THRESHOLD:
                anomaly = {
                    "max_velocity": round(max_vel, 2),
                    "mean_velocity": round(mean_vel, 2),
                    "description": "Movimento brusco / atípico detectado",
                }

        self._prev_landmarks = current
        return anomaly

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_overlay(
        self,
        frame: np.ndarray,
        faces: list[dict],
        activities: list[dict],
        anomaly: dict | None,
    ) -> np.ndarray:
        # Faces + emoções
        for f in faces:
            r = f["region"]
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            color = EMOTION_COLORS.get(f["emotion"], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{f['emotion']}: {f['confidence']:.0f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Atividades
        for act in activities:
            bx = act["bbox"]
            idx = act["color_idx"]
            c = tuple(int(v) for v in self.activity_colors[idx % len(self.activity_colors)])
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), c, 2)
            cv2.putText(
                frame,
                f"{act['type']} {act['confidence']:.0%}",
                (bx[0], bx[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c,
                2,
            )

        # Anomalia
        if anomaly:
            cv2.putText(
                frame,
                f"ANOMALIA (vel={anomaly['max_velocity']:.0f})",
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Info
        ts = self.frame_count / self.fps
        cv2.putText(
            frame,
            f"Frame: {self.frame_count} | {ts:.1f}s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return frame

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def _init_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir: {self.video_path}")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Vídeo: {self.total_frames} frames, {self.total_frames / self.fps:.1f}s, {self.width}x{self.height}")

    def process(self, skip_frames: int = 2, save_video: bool = True, show_preview: bool = True):
        self._init_video()

        writer = None
        if save_video:
            out_path = os.path.join(self.output_dir, "video_analisado.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )

        pbar = tqdm(total=self.total_frames, desc="Processando", unit="frame")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                pbar.update(1)

                # Skip frames para performance
                if self.frame_count % (skip_frames + 1) != 0:
                    if writer:
                        writer.write(frame)
                    continue

                ts = self.frame_count / self.fps

                # 1 + 2. Reconhecimento facial + emoções
                faces = self.analyze_faces(frame)

                # 3. Detecção de atividades
                activities = self.analyze_activity(frame)

                # 4. Detecção de anomalias
                anomaly = self.detect_anomaly(frame)

                # Armazenar resultados
                self.frame_analysis.append(
                    {
                        "frame": self.frame_count,
                        "timestamp": ts,
                        "faces": faces,
                        "activities": [{"type": a["type"], "confidence": a["confidence"]} for a in activities],
                        "anomaly": anomaly is not None,
                    }
                )

                for f in faces:
                    self.emotion_timeline.append(
                        {"timestamp": ts, "emotion": f["emotion"], "confidence": f["confidence"]}
                    )

                for a in activities:
                    self.activity_timeline.append(
                        {"timestamp": ts, "activity": a["type"], "confidence": a["confidence"]}
                    )

                if anomaly:
                    self.anomaly_timeline.append({"timestamp": ts, **anomaly})

                # Desenhar anotações
                annotated = self.draw_overlay(frame, faces, activities, anomaly)
                if writer:
                    writer.write(annotated)

                if show_preview:
                    cv2.imshow("Analise", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário.")
        finally:
            pbar.close()
            self.cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"✓ {len(self.frame_analysis)} frames analisados")

    # ------------------------------------------------------------------
    # Geração de resumo / relatório
    # ------------------------------------------------------------------

    def generate_report(self):
        if not self.frame_analysis:
            print("Nenhuma análise disponível.")
            return

        duration = self.total_frames / self.fps
        total_faces = sum(len(f["faces"]) for f in self.frame_analysis)
        total_anomalies = len(self.anomaly_timeline)

        sep = "=" * 55
        print(f"\n{sep}")
        print("           RELATÓRIO DE ANÁLISE DE VÍDEO")
        print(f"{sep}")

        print(f"\n📊 Estatísticas Gerais:")
        print(f"   Duração do vídeo ........... {duration:.1f}s")
        print(f"   Total de frames ............ {self.total_frames}")
        print(f"   Frames analisados .......... {len(self.frame_analysis)}")
        print(f"   Detecções de rosto ......... {total_faces}")
        print(f"   Anomalias detectadas ....... {total_anomalies}")

        # Emoções
        if self.emotion_timeline:
            emotions = Counter(e["emotion"] for e in self.emotion_timeline)
            print(f"\n😊 Emoções Detectadas:")
            for emotion, count in emotions.most_common():
                pct = count / len(self.emotion_timeline) * 100
                avg_conf = np.mean(
                    [e["confidence"] for e in self.emotion_timeline if e["emotion"] == emotion]
                )
                print(f"   {emotion:<12} {count:>4}x  ({pct:5.1f}%)  confiança média: {avg_conf:.0f}%")

        # Atividades
        if self.activity_timeline:
            acts = Counter(a["activity"] for a in self.activity_timeline)
            print(f"\n🏃 Atividades Detectadas:")
            for act, count in acts.most_common():
                pct = count / len(self.activity_timeline) * 100
                print(f"   {act.replace('_', ' '):<30} {count:>4}x  ({pct:5.1f}%)")

        # Anomalias
        if self.anomaly_timeline:
            print(f"\n⚠️  Anomalias (movimentos bruscos / atípicos):")
            for i, an in enumerate(self.anomaly_timeline, 1):
                print(
                    f"   {i:>3}. t={an['timestamp']:6.1f}s  |  "
                    f"vel_max={an['max_velocity']:6.1f}  vel_média={an['mean_velocity']:5.1f}"
                )
        else:
            print(f"\n✅ Nenhuma anomalia detectada.")

        # Cronologia por segmentos de 10s
        print(f"\n⏱️  Cronologia (segmentos de 10s):")
        for i in range(0, int(duration) + 1, 10):
            end = min(i + 10, int(duration))
            seg_emotions = [
                e["emotion"] for e in self.emotion_timeline if i <= e["timestamp"] < end
            ]
            seg_activities = [
                a["activity"] for a in self.activity_timeline if i <= a["timestamp"] < end
            ]
            seg_anomalies = sum(
                1 for a in self.anomaly_timeline if i <= a["timestamp"] < end
            )

            dom_emotion = Counter(seg_emotions).most_common(1)
            dom_emotion = dom_emotion[0][0] if dom_emotion else "—"
            dom_activity = Counter(seg_activities).most_common(1)
            dom_activity = dom_activity[0][0] if dom_activity else "—"
            anom_str = f"  ⚠{seg_anomalies}" if seg_anomalies else ""

            print(f"   {i:03d}-{end:03d}s  emoção: {dom_emotion:<12} atividade: {dom_activity}{anom_str}")

        self._save_data()
        print(f"\n💾 Resultados salvos em: {self.output_dir}/")
        print(sep)

    def _save_data(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        def default_serializer(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        data = {
            "video": {
                "path": self.video_path,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "resolution": f"{self.width}x{self.height}",
            },
            "summary": {
                "frames_analyzed": len(self.frame_analysis),
                "total_face_detections": sum(len(f["faces"]) for f in self.frame_analysis),
                "total_anomalies": len(self.anomaly_timeline),
            },
            "analysis": self.frame_analysis,
            "emotions": self.emotion_timeline,
            "activities": self.activity_timeline,
            "anomalies": self.anomaly_timeline,
        }
        json_path = os.path.join(self.output_dir, f"analise_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=default_serializer)

        if self.emotion_timeline:
            pd.DataFrame(self.emotion_timeline).to_csv(
                os.path.join(self.output_dir, f"emocoes_{ts}.csv"), index=False
            )
        if self.activity_timeline:
            pd.DataFrame(self.activity_timeline).to_csv(
                os.path.join(self.output_dir, f"atividades_{ts}.csv"), index=False
            )
        if self.anomaly_timeline:
            pd.DataFrame(self.anomaly_timeline).to_csv(
                os.path.join(self.output_dir, f"anomalias_{ts}.csv"), index=False
            )


# ======================================================================
# Entry point
# ======================================================================

def main():
    video_path = os.path.join(SCRIPT_DIR, "videos", "input.mp4")

    if not os.path.exists(video_path):
        print(f"❌ Vídeo não encontrado: {video_path}")
        print("   Coloque o vídeo em ./videos/input.mp4")
        return

    print("🎬 Sistema de Análise de Vídeo")
    print("   • Reconhecimento Facial")
    print("   • Análise de Expressões Emocionais")
    print("   • Detecção de Atividades")
    print("   • Detecção de Anomalias")
    print("=" * 45)

    analyzer = VideoAnalyzer(video_path)
    analyzer.process(skip_frames=2, save_video=True, show_preview=True)
    analyzer.generate_report()

    print("\n✅ Análise concluída!")


if __name__ == "__main__":
    main()
