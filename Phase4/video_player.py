import cv2
from face_detector import detect_faces, draw_faces_rectangle
from face_recog import load_known_faces, recognize_faces, draw_face_rectangle

def start_video_with_face_detect(video_path):
    video_capture = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        faces = detect_faces(frame)
        frame = draw_faces_rectangle(faces, frame)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def start_video_with_face_regonition(video_path):
    load_known_faces_dir = 'Phase4/known_faces'
    known_faces, known_names = load_known_faces(load_known_faces_dir)        
    video_capture = cv2.VideoCapture(video_path)
    
    face_locations, face_names = [], []
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        frame_id = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % 6 == 0:
            face_locations, face_names = recognize_faces(frame, known_faces, known_names)
        frame = draw_face_rectangle(frame, face_locations, face_names)
        
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'Phase4/videoplayback.mp4'
    # start_video_with_face_detect(video_path)
    start_video_with_face_regonition(video_path)
    