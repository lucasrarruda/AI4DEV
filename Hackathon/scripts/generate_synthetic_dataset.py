import os
import random
from PIL import Image, ImageDraw
import numpy as np

# Configurações do dataset
ICON_FOLDER = "C:/Projects/AI4DEV/Hackathon/icons"  # Pasta com os ícones AWS (PNG com fundo transparente)
OUTPUT_ROOT = "C:/Projects/AI4DEV/Hackathon"
OUTPUT_IMAGES_TRAIN = os.path.join(OUTPUT_ROOT, "images/train")
OUTPUT_LABELS_TRAIN = os.path.join(OUTPUT_ROOT, "labels/train")
OUTPUT_IMAGES_VAL = os.path.join(OUTPUT_ROOT, "images/val")
OUTPUT_LABELS_VAL = os.path.join(OUTPUT_ROOT, "labels/val")

BACKGROUND_SIZE = (640, 640)  # Tamanho das imagens (w, h)
NUM_IMAGES = 2000  # Total de imagens a gerar (serão divididas em treino/val)
VAL_SPLIT = 0.2  # Percentual de imagens para validação
ICONS_PER_IMAGE = (6, 20)  # Mínimo e máximo de ícones por imagem
CLASS_ID = 0  # ID da classe "Component"

# Cria pastas de saída, se não existirem
os.makedirs(OUTPUT_IMAGES_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_LABELS_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_VAL, exist_ok=True)
os.makedirs(OUTPUT_LABELS_VAL, exist_ok=True)

# Lista de arquivos de ícones
icon_files = [f for f in os.listdir(ICON_FOLDER) if f.lower().endswith(".png")]
if not icon_files:
    raise RuntimeError(f"Nenhum ícone PNG encontrado em {ICON_FOLDER}")

# Lista de palavras para rótulos aleatórios
LABEL_WORDS = [
    'API', 'Gateway', 'Service', 'Node', 'DB', 'Auth', 'Cache', 'Proxy', 'Worker', 'Queue',
    'Monitor', 'Storage', 'Compute', 'Analytics', 'Lambda', 'EC2', 'S3', 'RDS', 'IAM', 'ECS', 'EKS',
    'SNS', 'SQS', 'CloudFront', 'Redshift', 'Elastic', 'Beanstalk', 'Route53', 'VPC', 'DynamoDB'
]

for idx in range(NUM_IMAGES):
    bg = Image.new("RGB", BACKGROUND_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(bg)
    boxes = []
    icon_positions = []  # Para armazenar as posições dos ícones

    num_icons = random.randint(*ICONS_PER_IMAGE)
    attempts = 0
    placed = 0
    max_attempts = num_icons * 20  # Limite de tentativas para evitar loop infinito
    while placed < num_icons and attempts < max_attempts:
        icon_name = random.choice(icon_files)
        icon = Image.open(os.path.join(ICON_FOLDER, icon_name)).convert("RGBA")
        scale = random.uniform(0.3, 1.0)
        w, h = icon.size
        icon = icon.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        icon_w, icon_h = icon.size
        bg_w, bg_h = BACKGROUND_SIZE
        max_x = bg_w - icon_w
        max_y = bg_h - icon_h
        if max_x <= 0 or max_y <= 0:
            attempts += 1
            continue
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        # Verifica sobreposição
        overlap = False
        for (_, _, _, bx, by, bw, bh) in icon_positions:
            if not (x + icon_w < bx or x > bx + bw or y + icon_h < by or y > by + bh):
                overlap = True
                break
        if overlap:
            attempts += 1
            continue
        # Desenha retângulo claro atrás do ícone
        rect_w = int(icon_w * random.uniform(1.1, 1.5))
        rect_h = int(icon_h * random.uniform(1.1, 1.5))
        rect_x = x - (rect_w - icon_w) // 2
        rect_y = y - (rect_h - icon_h) // 2
        rect_x = max(0, rect_x)
        rect_y = max(0, rect_y)
        rect_w = min(rect_w, bg_w - rect_x)
        rect_h = min(rect_h, bg_h - rect_y)
        color = tuple([random.randint(200, 255) for _ in range(3)])
        draw.rectangle([rect_x, rect_y, rect_x + rect_w, rect_y + rect_h], fill=color)
        # Cola o ícone
        bg.paste(icon, (x, y), icon)
        # Bounding box normalizado
        xc = (x + icon_w / 2) / bg_w
        yc = (y + icon_h / 2) / bg_h
        w_norm = icon_w / bg_w
        h_norm = icon_h / bg_h
        boxes.append((CLASS_ID, xc, yc, w_norm, h_norm))
        icon_positions.append((icon_name, placed, (x + icon_w // 2, y + icon_h // 2), x, y, icon_w, icon_h))
        placed += 1
        attempts += 1

    # Desenha linhas conectando os centros dos ícones
    if len(icon_positions) > 1:
        centers = [pos[2] for pos in icon_positions]
        draw.line(centers, fill=(150, 150, 150), width=3)

    # Sorteia grupos de ícones para receberem um retângulo claro (nem todos terão)
    num_groups = random.randint(1, max(1, placed // 2))
    icons_for_rects = random.sample(range(placed), num_groups)
    group_size = max(2, placed // num_groups) if num_groups > 1 else placed
    used = set()
    for group_start in icons_for_rects:
        group = []
        for i in range(group_start, min(group_start + group_size, placed)):
            if i not in used:
                group.append(i)
                used.add(i)
        if len(group) < 2:
            continue
        # Calcula a região que engloba todos os ícones do grupo
        xs = [icon_positions[i][3] for i in group]
        ys = [icon_positions[i][4] for i in group]
        ws = [icon_positions[i][5] for i in group]
        hs = [icon_positions[i][6] for i in group]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max([x + w for x, w in zip(xs, ws)])
        max_y = max([y + h for y, h in zip(ys, hs)])
        pad = random.randint(10, 40)
        rect_x1 = max(0, min_x - pad)
        rect_y1 = max(0, min_y - pad)
        rect_x2 = min(bg_w, max_x + pad)
        rect_y2 = min(bg_h, max_y + pad)
        color = tuple([random.randint(200, 255) for _ in range(3)])
        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=color)
    # Cola os ícones (agora por cima dos retângulos)
    for i, (icon_name, _, center, x, y, icon_w, icon_h) in enumerate(icon_positions):
        icon = Image.open(os.path.join(ICON_FOLDER, icon_name)).convert("RGBA")
        icon = icon.resize((icon_w, icon_h), Image.Resampling.LANCZOS)
        bg.paste(icon, (x, y), icon)
        # Gera rótulo aleatório de 0 a 3 palavras
        num_words = random.randint(0, 3)
        if num_words > 0:
            label = ' '.join(random.sample(LABEL_WORDS, num_words))
            # Define posição do texto (centralizado abaixo do ícone)
            text_x = x + icon_w // 2
            text_y = y + icon_h + 10  # Espaçamento maior abaixo do ícone
            # Ajusta para não sair da imagem
            text_x = max(0, min(text_x, bg_w - 1))
            text_y = max(0, min(text_y, bg_h - 1))
            # Desenha o texto
            draw.text((text_x, text_y), label, fill=(50, 50, 50), anchor="ms")

    # Define destino (treino ou validação)
    is_val = idx < int(NUM_IMAGES * VAL_SPLIT)
    out_img_dir = OUTPUT_IMAGES_VAL if is_val else OUTPUT_IMAGES_TRAIN
    out_lbl_dir = OUTPUT_LABELS_VAL if is_val else OUTPUT_LABELS_TRAIN

    # Salva imagem e label
    img_name = f"{idx:06d}.png"
    lbl_name = f"{idx:06d}.txt"
    bg.save(os.path.join(out_img_dir, img_name))
    with open(os.path.join(out_lbl_dir, lbl_name), "w") as f:
        for c, xc, yc, wn, hn in boxes:
            f.write(f"{c} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

print("Dataset sintético gerado com sucesso!")
