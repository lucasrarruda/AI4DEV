import cv2
from player_strategy import AlgorithmStrategy

class Player:
    def __init__(self, video_path, strategy: AlgorithmStrategy, window_title="Video"):
        self.video_path = video_path
        self.strategy = strategy
        self.window_title = window_title

    def set_strategy(self, strategy: AlgorithmStrategy):
        self.strategy = strategy

    def play(self):
        video_capture = cv2.VideoCapture(self.video_path)
        
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            frame = self.strategy.process_frame(frame, video_capture)
            
            cv2.imshow(self.window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()