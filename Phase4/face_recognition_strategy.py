import cv2
import face_recognition
import os
import numpy as np
from player_strategy import AlgorithmStrategy

class FaceRecognitionStrategy(AlgorithmStrategy):
    def __init__(self):
        self.face_locations = []
        self.face_names = []
        load_known_faces_dir = 'Phase4/known_faces'
        self.known_faces, self.known_names = self.load_known_faces(load_known_faces_dir)

    def process_frame(self, frame, video_capture):
        frame_id = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % 6 == 0:
            self.face_locations, self.face_names = self.recognize_faces(frame, self.known_faces, self.known_names)
        return self.draw_face_rectangle(frame, self.face_locations, self.face_names)
    
    def load_known_faces(self, known_faces_dir):
        known_faces = []
        known_names = []
        
        for filename in os.listdir(known_faces_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(os.path.splitext(filename)[0])
        
        return known_faces, known_names

    def prepare_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        return face_locations, face_encodings

    def recognize_faces(self, frame, known_faces, known_names):
        face_locations, face_encodings = self.prepare_faces(frame)
        face_names = []
        
        for face_encoding in face_encodings:
            name = "Unknown"
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
            face_names.append(name)
        
        return face_locations, face_names

    def draw_face_rectangle(self, frame, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return frame