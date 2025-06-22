from ultralytics import YOLO

# Carrega modelo pré-treinado YOLOv8 nano
model = YOLO('yolov8n.pt')  # Você pode trocar por 'yolov8s.pt', 'yolov8m.pt', etc.

# Treina com o dataset customizado
model.train(data='../aws-components.yaml', epochs=50, imgsz=640, project='aws-components', name='yolov8n-custom')
