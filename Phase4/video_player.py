import cv2
from face_detector import detect_faces, draw_faces_rectangle

def start_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    # load_known_faces_dir = 'Phase4/known_faces'
    # known_faces, known_names = recognize_faces.load_known_faces(load_known_faces_dir)        
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        faces = detect_faces(frame)
        frame = draw_faces_rectangle(faces, frame)
        # frame = draw_face_rectangle(faces, frame)
        # if len(faces) > 0:
        #     face_locations, face_names = recognize_faces(frame, known_faces, known_names)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'Phase4/videoplayback.mp4'
    start_video(video_path)
    