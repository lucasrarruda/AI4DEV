import os
from ultralytics import YOLO

# Configurações
DATASET_DIR = r'C:/Projects/AI4DEV/Hackathon2/datasets/synthetic_yolo'
MODEL_DIR = r'C:/Projects/AI4DEV/Hackathon2/models'
PRETRAINED_MODEL = 'yolov8n.pt'  # Pode ser yolov8n.pt, yolov8s.pt, etc.
EPOCHS = 50
IMG_SIZE = 640

# Cria diretório de modelos se não existir
os.makedirs(MODEL_DIR, exist_ok=True)

# Treinamento YOLO
model = YOLO(PRETRAINED_MODEL)
model.train(
    data=os.path.join(DATASET_DIR, 'data.yaml'),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    project=MODEL_DIR,
    name='yolo_sintetico',
    exist_ok=True,
    save=True,  # Salva checkpoints a cada época
    save_period=1  # Salva a cada época
)
print('Treinamento YOLO concluído!')
