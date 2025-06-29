import cv2
from ultralytics import YOLO
import pytesseract

# Caminho para o modelo treinado
MODEL_PATH = 'C:/Projects/AI4DEV/Hackathon/scripts/aws-components/yolov8n-custom/weights/best.pt'  # ajuste conforme o caminho real do melhor peso

# Pasta com imagens para teste (pode ser a validação)
TEST_IMAGES_FOLDER = 'C:/Projects/AI4DEV/Hackathon/test-diagrams'

def ocr_on_expanded_region(img, x1, y1, x2, y2, expand_ratio=0.3):
    h, w = img.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    # Expande 30% para cada lado
    expand_w = int(box_w * expand_ratio / 2)
    expand_h = int(box_h * expand_ratio / 2)
    nx1 = max(0, x1 - expand_w)
    ny1 = max(0, y1 - expand_h)
    nx2 = min(w, x2 + expand_w)
    ny2 = min(h, y2 + expand_h)
    crop = img[ny1:ny2, nx1:nx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Opcional: binarização
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    return text.strip()

def main():
    model = YOLO(MODEL_PATH)

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

                label = f'Component {conf:.2f}'
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                # OCR na região expandida
                ocr_text = ocr_on_expanded_region(img, x1, y1, x2, y2)
                # Calcula tamanho do texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4  # Fonte menor
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > th else y1 + th + 2
                cv2.rectangle(img, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), (30, 30, 30), -1)
                cv2.putText(img, label, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
                # Mostra OCR logo abaixo
                if ocr_text:
                    ocr_lines = ocr_text.splitlines()
                    for i, line in enumerate(ocr_lines):
                        if not line.strip():
                            continue
                        ocr_y = text_y + (i+1)*(th+4)
                        cv2.rectangle(img, (text_x, ocr_y - th - baseline), (text_x + tw, ocr_y + baseline), (30, 30, 30), -1)
                        cv2.putText(img, line, (text_x, ocr_y), font, font_scale, (255,255,0), thickness, cv2.LINE_AA)

        cv2.imshow('Detecção', img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC para sair
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
