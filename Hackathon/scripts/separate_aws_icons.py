import os
import shutil
from PIL import Image
import numpy as np
from collections import Counter

# Caminhos
ICON_DIR = "C:/Projects/AI4DEV/Hackathon/aws_icons"
OUTPUT_DIR = "C:/Projects/AI4DEV/Hackathon/aws_icons_classified"

# Cores principais por categoria AWS (valores aproximados, ajuste conforme necessário)
AWS_CATEGORY_COLORS = {
    "compute": (255, 153, 0),        # Laranja
    "storage": (0, 102, 204),        # Azul
    "database": (0, 153, 0),         # Verde
    "networking": (153, 0, 204),     # Roxo
    "application": (255, 102, 204),  # Rosa
    "security": (255, 0, 0),         # Vermelho
    "observability": (0, 204, 204),  # Ciano
    "messaging": (255, 204, 0),      # Amarelo
    "devtools": (102, 102, 102),     # Cinza
    "ml": (0, 0, 0),                 # Preto
}

# Nomes das categorias AWS (padrão oficial)
AWS_CATEGORY_NAMES = {
    "compute": "Compute",
    "storage": "Storage",
    "database": "Database",
    "networking": "Networking & Content Delivery",
    "application": "Application Integration",
    "security": "Security, Identity, & Compliance",
    "observability": "Management & Governance",
    "messaging": "Messaging",
    "devtools": "Developer Tools",
    "ml": "Machine Learning"
}

# Cria as pastas de saída com nomes de tags (chaves do dicionário)
for categoria in AWS_CATEGORY_NAMES.keys():
    os.makedirs(os.path.join(OUTPUT_DIR, categoria), exist_ok=True)

# Lista de caminhos e cores predominantes por ícone
file_paths = []
predominant_colors = []

# Função para encontrar a cor predominante ignorando pixels transparentes e quase brancos
def get_predominant_color(img):
    arr = np.array(img)
    if arr.shape[2] == 4:
        # RGBA: remove pixels transparentes
        arr = arr[arr[:, :, 3] > 10]
        arr = arr[:, :3]
    else:
        arr = arr.reshape(-1, 3)
    # Remove pixels quase brancos
    arr = arr[np.all(arr < 240, axis=1)]
    if len(arr) == 0:
        return np.array([255, 255, 255])
    # Conta as cores
    arr_tuples = [tuple(x) for x in arr]
    most_common = Counter(arr_tuples).most_common(1)[0][0]
    return np.array(most_common)

# Extrai a cor predominante de cada imagem
for file in os.listdir(ICON_DIR):
    if not file.lower().endswith(".png"):
        continue
    path = os.path.join(ICON_DIR, file)
    img = Image.open(path).convert("RGBA").resize((32, 32))
    pred_color = get_predominant_color(img)
    predominant_colors.append(pred_color)
    file_paths.append(path)

# Classifica cada ícone pela cor predominante
for path, pred_color in zip(file_paths, predominant_colors):
    min_dist = float('inf')
    best_cat = None
    for cat, ref_color in AWS_CATEGORY_COLORS.items():
        dist = np.linalg.norm(pred_color - np.array(ref_color))
        if dist < min_dist:
            min_dist = dist
            best_cat = cat
    dst = os.path.join(OUTPUT_DIR, best_cat, os.path.basename(path))
    shutil.copy2(path, dst)

print("✅ Ícones classificados por cor conforme categorias AWS!")
