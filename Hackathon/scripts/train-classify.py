import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.utils import save_image
import shutil
from collections import Counter

# === CONFIG ===
DATASET_DIR = "C:/Projects/AI4DEV/Hackathon/aws_icons_basic_class"  # Estrutura: grouped_icons/Compute/*.png etc.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10

# === TRANSFORMAÇÕES ===
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === DADOS ===
dataset = ImageFolder(DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = dataset.classes

# === MODELO ===
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === DATA AUGMENTATION PARA BALANCEAMENTO ===
def balancear_dataset_com_augment(dataset_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # Conta imagens por classe
    class_counts = {}
    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path): continue
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[cls] = len(imgs)
    max_count = max(class_counts.values())
    # Transforms de augmentação
    aug_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomResizedCrop(128, scale=(0.8, 1.0)),
        T.ToTensor()
    ])
    base_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path): continue
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        out_cls_path = os.path.join(output_dir, cls)
        os.makedirs(out_cls_path, exist_ok=True)
        # Copia imagens originais
        for img_name in imgs:
            src = os.path.join(cls_path, img_name)
            dst = os.path.join(out_cls_path, img_name)
            shutil.copy2(src, dst)
        # Gera augmentações se necessário
        n_to_add = max_count - len(imgs)
        if n_to_add > 0:
            for i in range(n_to_add):
                img_name = imgs[i % len(imgs)]
                img_path = os.path.join(cls_path, img_name)
                img = default_loader(img_path)
                aug_img = aug_transforms(img)
                save_image(aug_img, os.path.join(out_cls_path, f'aug_{i}_{img_name}'))

# Caminho para base balanceada
BALANCED_DATASET_DIR = "C:/Projects/AI4DEV/Hackathon/aws_icons_basic_class_balanced"
balancear_dataset_com_augment(DATASET_DIR, BALANCED_DATASET_DIR)

# Usa a base balanceada para o treinamento
DATASET_DIR = BALANCED_DATASET_DIR
dataset = ImageFolder(DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = dataset.classes

# === TREINAMENTO ===
print("⏳ Treinando classificador de grupo...")
model.train()
for epoch in range(EPOCHS):
    total, correct, loss_sum = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        _, predicted = pred.max(1)
        correct += predicted.eq(y).sum().item()
        total += x.size(0)
    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss_sum/total:.4f} - Acc: {acc:.2f}%")

torch.save(model.state_dict(), "modelo_group_classifier.pth")

# Salva a lista de classes
with open("modelo_group_classifier_classes.txt", "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(f"{c}\n")

# === FUNÇÃO DE CLASSIFICAÇÃO ===
# def classify_group(img_bgr):
#     model.eval()
#     img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#     tensor = transform(img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         output = model(tensor)
#         pred = output.argmax(1).item()
#     return class_names[pred]
