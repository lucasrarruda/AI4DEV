import cv2
import os

def detect_faces(frame, haar_cascade):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=6, minSize=(100, 100))
    return faces

def draw_face_rectangle(faces, frame):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

def face_detect(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(video_path)        
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        faces = detect_faces(frame, face_cascade)
        frame = draw_face_rectangle(faces, frame)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'Phase4/videoplayback.mp4'
    face_detect(video_path)
    