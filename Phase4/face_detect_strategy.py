import cv2
import numpy as np
from player_strategy import AlgorithmStrategy
import keras

class FaceDetectStrategy(AlgorithmStrategy):
    def __init__(self):
        self.face_dnn = cv2.dnn.readNetFromCaffe(
            'Phase4/models/deploy.prototxt.txt',
            'Phase4/models/res10_300x300_ssd_iter_140000.caffemodel'
        )
        self.model = keras.models.load_model('Phase4/models/fer2013_mini_XCEPTION.119-0.65.hdf5', compile=False)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def process_frame(self, frame, _):
        self.trackers = []
        faces = self.detect_faces(frame)
        return self.draw_faces_rectangle(faces, frame)
    
    def detect_faces(self, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        self.face_dnn.setInput(blob)
        detections = self.face_dnn.forward()
        faces = self.get_faces(detections, frame)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size > 0:
                emotion = self.predict_emotion(face_img)
                cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return self.get_faces(detections, frame)
    
    def get_faces(self, detections, frame):
        h, w = frame.shape[:2]
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                x = startX
                y = startY
                width = endX - startX
                height = endY - startY
                faces.append((x, y, width, height))
        return faces

    def draw_faces_rectangle(self, faces, frame):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame
    
    def predict_emotion(self, face_img):
        # Pré-processamento: cinza, resize, normalização
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        img = resized.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        preds = self.model.predict(img)
        emotion_idx = np.argmax(preds)
        return self.emotion_labels[emotion_idx]