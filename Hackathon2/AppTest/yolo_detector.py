import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Caminho do modelo YOLO treinado (ajuste conforme necessário)
MODEL_PATH = r'diagram-detector.pt'
IMG_SIZE = 640

# Carrega o modelo YOLO apenas uma vez
_model = None
def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

def detect_components_pil(pil_image):
    """
    Recebe uma imagem PIL, salva temporariamente, roda YOLO e retorna lista de componentes detectados.
    Retorna: lista de dicts: {"bbox": [x1, y1, x2, y2], "classe": str, "conf": float}
    """
    # Salva imagem PIL em arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        pil_image.save(tmp.name, format='JPEG')
        img_path = tmp.name
    model = get_model()
    results = model(img_path, imgsz=IMG_SIZE)
    detected = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls]
            detected.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "classe": label,
                "conf": conf
            })
    return detected
