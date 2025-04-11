from face_emotion_strategy import FaceEmotionStrategy
from face_detect_strategy import FaceDetectStrategy
from face_recognition_strategy  import FaceRecognitionStrategy
from player import Player

if __name__ == "__main__":
    video_path = 'Phase4/videoplayback.mp4'

    face_detect_strategy = FaceDetectStrategy()
    player = Player(video_path, face_detect_strategy, window_title="Face Detection")
    player.play()

    face_recognition_strategy = FaceRecognitionStrategy()
    player.set_strategy(face_recognition_strategy)
    player.window_title = "Face Recognition"
    player.play()

    face_emotion_strategy = FaceEmotionStrategy()
    player.set_strategy(face_emotion_strategy)
    player.window_title = "Face Emotion Recognition"
    player.play()