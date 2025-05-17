from player_strategy import AlgorithmStrategy
from deepface import DeepFace
import cv2

class FaceEmotionStrategy(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.trackers = []

    def process_frame(self, frame, video_capture):
        height, width = frame.shape[:2]
        if width > 640:
            new_width = 640
            new_height = int((640 / width) * height)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        frame_id = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        faces = []
        if frame_id % 10 == 0:
            faces = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
            self.trackers = []
            for face in faces:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                bbox = (int(x), int(y), int(w), int(h))

                tracker = cv2.TrackerKCF.create()
                tracker.init(frame, bbox)
                self.trackers.append(tracker)
        else:
            new_trackers = []
            for tracker in self.trackers:
                success, box = tracker.update(frame)
                if success:
                    new_trackers.append(tracker)
                    x, y, w, h = [int(v) for v in box]
                    face = {
                        'region': {'x': x, 'y': y, 'w': w, 'h': h},
                        'dominant_emotion': 'unknown'
                        # 'dominant_emotion': face['dominant_emotion']
                    }
                    faces.append(face)
            self.trackers = new_trackers
        return self.draw_faces_emotions(faces, frame)
    
    def draw_faces_emotions(self, faces, frame):
        for face in faces:
            emotion = face['dominant_emotion']
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return frame