import cv2
from ultralytics import YOLO

# Caminho para o modelo treinado
MODEL_PATH = 'C:/Projects/AI4DEV/Hackaton/DetectComponents/scripts/aws-components/yolov8n-custom/weights/best.pt'  # ajuste conforme o caminho real do melhor peso

# Pasta com imagens para teste (pode ser a validação)
TEST_IMAGES_FOLDER = 'C:/Projects/AI4DEV/Hackaton/DetectComponents/test-diagrams'

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

                label = f'Component {conf:.2f}'
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow('Detecção', img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC para sair
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
