import os
from ultralytics import YOLO
import cv2

# Configurações
test_img_dir = r'C:/Projects/AI4DEV/Hackathon2/test_images'
model_path = r'C:/Projects/AI4DEV/Hackathon2/models/yolo_sintetico/weights/best.pt'  # Ajuste se necessário
img_size = 640

# Carrega modelo treinado
model = YOLO(model_path)

# Lista imagens de teste
img_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in img_files:
    img_path = os.path.join(test_img_dir, fname)
    results = model(img_path, imgsz=img_size)
    img = cv2.imread(img_path)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = f'{model.names[cls]} ({conf:.2f})'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow('YOLO Detecção', img)
    key = cv2.waitKey(0)
    if key == 27:
        break
cv2.destroyAllWindows()
print('Teste concluído!')
