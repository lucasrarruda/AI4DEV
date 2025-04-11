import cv2
from player_strategy import PlayerStrategy

class FaceDetectStrategy(PlayerStrategy):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def process_frame(self, frame, _):
        faces = self.detect_faces(frame)
        return self.draw_faces_rectangle(faces, frame)

    def detect_faces(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return faces

    def draw_faces_rectangle(self, faces, frame):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame