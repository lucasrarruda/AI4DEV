from player_strategy import AlgorithmStrategy
from deepface import DeepFace
import cv2

class FaceEmotionStrategy(AlgorithmStrategy):
    def process_frame(self, frame, video_capture):
        faces = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return self.draw_faces_emotions(faces, frame)
    
    def draw_faces_emotions(self, faces, frame):
        for face in faces:
            emotion = face['dominant_emotion']
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return frame