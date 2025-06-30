import cv2
from ultralytics import YOLO
import sys
sys.path.append('C:/Projects/AI4DEV/Hackathon/scripts')
from classifier import GroupClassifierResNet50

# Caminho para o modelo treinado
MODEL_PATH = 'C:/Projects/AI4DEV/Hackathon/scripts/aws-components/yolov8n-custom/weights/best.pt'  # ajuste conforme o caminho real do melhor peso

# Pasta com imagens para teste (pode ser a validação)
TEST_IMAGES_FOLDER = 'C:/Projects/AI4DEV/Hackathon/test-diagrams'

def main():
    model = YOLO(MODEL_PATH)
    classifier = GroupClassifierResNet50()  # Usa a estratégia ResNet50
    import os
    image_files = [f for f in os.listdir(TEST_IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGES_FOLDER, img_file)
        results = model(img_path)

        # Pega a imagem para desenhar
        img = cv2.imread(img_path)

        # Itera pelas detecções
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                if conf < 0.7:
                    continue

                # Recorta a região detectada para classificar
                crop = img[y1:y2, x1:x2].copy()
                group = classifier.classify(crop)
                label = f'{group} | C {conf:.2f}'
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                # Calcula tamanho do texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4  # Fonte menor
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > th else y1 + th + 2
                cv2.rectangle(img, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), (30, 30, 30), -1)
                cv2.putText(img, label, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

        cv2.imshow('Detecção', img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC para sair
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
