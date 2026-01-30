import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from datetime import datetime
import warnings
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
warnings.filterwarnings("ignore")

# Cores para emoções (BGR)
EMOTION_COLORS = {
    'angry': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (128, 0, 128),
    'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 165, 0), 'neutral': (128, 128, 128)
}


class VideoAnalyzer:
    def __init__(self, video_path, output_dir="analysis_results"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = None
        self.frame_analysis = []
        self.emotion_timeline = []
        self.activity_timeline = []
        self.frame_count = 0
        self.fps = 30
        self.total_frames = 0
        self.pose = self._init_mediapipe()
        os.makedirs(output_dir, exist_ok=True)

    def _download_mediapipe_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'blaze_face_short_range.tflite')

        if not os.path.exists(model_path):
            model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(model_url, model_path)

        return model_path


    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            return mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        except Exception:
            return None

    def _init_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir: {self.video_path}")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 Vídeo: {self.total_frames} frames, {self.total_frames/self.fps:.1f}s")

    def analyze_faces(self, frame):
        try:
            model_path = self._download_mediapipe_model()
            
            base_options = tasks.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            face_detector = vision.FaceDetector.create_from_options(options)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = face_detector.detect(mp_image)
            
            result = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)


                face_region = frame[y:y+h, x:x+w]

                if face_region.size > 0:
                    try:
                        results = DeepFace.analyze(
                            face_region,
                            actions=['emotion', 'age', 'gender'],
                            enforce_detection=False,
                            detector_backend='opencv'
                        )

                        results = results if isinstance(results, list) else [results]

                        for r in results:
                            r['m_region'] = {'x': x, 'y': y, 'w': w, 'h': h}
                            result.append(r)

                        
                    except:
                        return []
                
                
                
            return [{
                        'region': r.get('m_region', {}),
                        'emotion': r.get('dominant_emotion', 'neutral'),
                        'confidence': max(r.get('emotion', {}).values(), default=0),
                        'age': r.get('age', 0),
                        'gender': r.get('dominant_gender', 'N/A')
                    } for r in result]

        except Exception:
            return []

    def analyze_activity(self, frame):
        if not self.pose:
            return {'detected': False, 'type': 'unknown', 'confidence': 0}
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            if not results.pose_landmarks:
                return {'detected': False, 'type': 'unknown', 'confidence': 0}
            
            lm = results.pose_landmarks.landmark
            arms_up = lm[15].y < lm[11].y and lm[16].y < lm[12].y
            torso = abs((lm[23].y + lm[24].y)/2 - (lm[11].y + lm[12].y)/2)
            
            if arms_up:
                return {'detected': True, 'type': 'gesticulando', 'confidence': 0.7}
            elif torso < 0.3:
                return {'detected': True, 'type': 'sentado', 'confidence': 0.6}
            return {'detected': True, 'type': 'em_pe', 'confidence': 0.6}
        except Exception:
            return {'detected': False, 'type': 'unknown', 'confidence': 0}

    def draw_overlay(self, frame, faces, activity):
        for f in faces:
            r = f.get('region', {})
            if not r:
                continue
            x, y, w, h = r.get('x', 0), r.get('y', 0), r.get('w', 0), r.get('h', 0)
            color = EMOTION_COLORS.get(f['emotion'], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{f['emotion']}: {f['confidence']:.0f}%", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"{f['gender']}, {int(f['age'])}", (x, y+h+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if activity.get('detected'):
            cv2.putText(frame, f"Atividade: {activity['type']}", (10, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        ts = self.frame_count / self.fps
        cv2.putText(frame, f"Frame: {self.frame_count} | {ts:.1f}s", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def process(self, skip_frames=2, save_video=True, show_preview=True):
        self._init_video()
        
        writer = None
        if save_video:
            w, h = int(self.cap.get(3)), int(self.cap.get(4))
            writer = cv2.VideoWriter(
                f"{self.output_dir}/video_analisado.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h)
            )
        
        print("Processando... (pressione 'q' para parar)")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                if self.frame_count % (skip_frames + 1) != 0:
                    continue
                
                ts = self.frame_count / self.fps
                faces = self.analyze_faces(frame)
                activity = self.analyze_activity(frame)
                
                self.frame_analysis.append({
                    'frame': self.frame_count, 'timestamp': ts,
                    'faces': faces, 'activity': activity
                })
                
                for f in faces:
                    self.emotion_timeline.append({
                        'timestamp': ts, 'emotion': f['emotion'], 'confidence': f['confidence']
                    })
                
                if activity.get('detected'):
                    self.activity_timeline.append({
                        'timestamp': ts, 'activity': activity['type'], 'confidence': activity['confidence']
                    })
                
                annotated = self.draw_overlay(frame, faces, activity)
                if writer:
                    writer.write(annotated)
                
                progress = (self.frame_count / self.total_frames) * 100
                print(f"\r{progress:.1f}% ({self.frame_count}/{self.total_frames})", end="")
                
                if show_preview:
                    cv2.imshow('Analise', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nInterrompido")
        finally:
            self.cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"\n✓ {len(self.frame_analysis)} frames analisados")

    def generate_report(self):
        if not self.frame_analysis:
            print("Nenhuma análise disponível")
            return
        
        duration = self.total_frames / self.fps
        total_faces = sum(len(f['faces']) for f in self.frame_analysis)
        
        print(f"\n{'='*50}")
        print("RELATÓRIO DE ANÁLISE")
        print(f"{'='*50}")
        print(f"\n📊 Estatísticas:")
        print(f"   Duração: {duration:.1f}s | Frames: {len(self.frame_analysis)} | Faces: {total_faces}")
        
        if self.emotion_timeline:
            emotions = Counter(e['emotion'] for e in self.emotion_timeline)
            print(f"\n😊 Emoções:")
            for emotion, count in emotions.most_common():
                pct = count / len(self.emotion_timeline) * 100
                avg_conf = np.mean([e['confidence'] for e in self.emotion_timeline if e['emotion'] == emotion])
                print(f"   {emotion}: {count} ({pct:.1f}%) - {avg_conf:.0f}% confiança")
        
        if self.activity_timeline:
            activities = Counter(a['activity'] for a in self.activity_timeline)
            print(f"\n🏃 Atividades:")
            for act, count in activities.most_common():
                pct = count / len(self.activity_timeline) * 100
                print(f"   {act.replace('_', ' ')}: {count} ({pct:.1f}%)")
        
        print(f"\n⏱️ Cronologia (por 10s):")
        for i in range(0, int(duration), 10):
            seg_emotions = [e['emotion'] for e in self.emotion_timeline if i <= e['timestamp'] < i+10]
            dom_emotion = Counter(seg_emotions).most_common(1)[0][0] if seg_emotions else "N/A"
            print(f"   {i:02d}-{min(i+10, int(duration)):02d}s: {dom_emotion}")
        
        self._save_data()
        print(f"\n💾 Resultados salvos em: {self.output_dir}/")
        print(f"{'='*50}")

    def _save_data(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        data = {
            'video': {'path': self.video_path, 'fps': self.fps, 'frames': self.total_frames},
            'analysis': self.frame_analysis,
            'emotions': self.emotion_timeline,
            'activities': self.activity_timeline
        }
        with open(f"{self.output_dir}/analise_{ts}.json", 'w') as f:
            json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
        
        if self.emotion_timeline:
            pd.DataFrame(self.emotion_timeline).to_csv(f"{self.output_dir}/emocoes_{ts}.csv", index=False)
        if self.activity_timeline:
            pd.DataFrame(self.activity_timeline).to_csv(f"{self.output_dir}/atividades_{ts}.csv", index=False)
        
        self._create_charts(ts)

    def _create_charts(self, ts):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Resumo da Análise', fontsize=14)
            
            if self.emotion_timeline:
                emotions = Counter(e['emotion'] for e in self.emotion_timeline)
                axes[0, 0].pie(emotions.values(), labels=emotions.keys(), autopct='%1.1f%%')
                axes[0, 0].set_title('Emoções')
            
            if self.emotion_timeline:
                for i, em in enumerate(set(e['emotion'] for e in self.emotion_timeline)):
                    times = [e['timestamp'] for e in self.emotion_timeline if e['emotion'] == em]
                    axes[0, 1].scatter(times, [i]*len(times), label=em, alpha=0.6)
                axes[0, 1].set_title('Timeline de Emoções')
                axes[0, 1].legend(fontsize=8)
            
            if self.activity_timeline:
                acts = Counter(a['activity'] for a in self.activity_timeline)
                axes[1, 0].bar(acts.keys(), acts.values())
                axes[1, 0].set_title('Atividades')
            
            times = [f['timestamp'] for f in self.frame_analysis]
            faces = [len(f['faces']) for f in self.frame_analysis]
            axes[1, 1].plot(times, faces, alpha=0.7)
            axes[1, 1].fill_between(times, faces, alpha=0.3)
            axes[1, 1].set_title('Faces Detectadas')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/graficos_{ts}.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"Erro nos gráficos: {e}")


def main():
    video_path = "./videos/input.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Vídeo não encontrado: {video_path}")
        return
    
    print("🎬 Sistema de Análise de Vídeo com DeepFace")
    print("=" * 45)
    
    analyzer = VideoAnalyzer(video_path)
    analyzer.process(skip_frames=15, save_video=True, show_preview=True)
    analyzer.generate_report()
    
    print("\n✅ Concluído!")


if __name__ == "__main__":
    main()
