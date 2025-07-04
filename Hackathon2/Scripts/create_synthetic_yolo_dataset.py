import os
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from tqdm import tqdm
import shutil

# Configurações
ICONS_STRIDE = r'C:/Projects/AI4DEV/Hackathon2/AWS_ONLY_STRIDE'  # Estrutura: AWS_ONLY_STRIDE/<classe>/*.png
ICONS_FULL = r'C:/Projects/AI4DEV/Hackathon2/AWS_FULL'            # Estrutura: AWS_FULL/*.png
OUT_DATASET = r'C:/Projects/AI4DEV/Hackathon2/datasets/synthetic_yolo'
IMG_SIZE = (640, 480)
N_IMAGES = 2000
ICONS_PER_IMG = (8, 15)
BG_COLORS = [(245,245,245), (255,255,255), (230,240,255), (250,250,240)]

# Utilidades

def get_stride_icons():
    stride_icons = []
    for cls in os.listdir(ICONS_STRIDE):
        cls_path = os.path.join(ICONS_STRIDE, cls)
        if not os.path.isdir(cls_path): continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith('.png'):
                stride_icons.append((cls, os.path.join(cls_path, fname)))
    return stride_icons

def get_full_icons(stride_icon_names):
    full_icons = []
    for fname in os.listdir(ICONS_FULL):
        if fname.lower().endswith('.png') and fname not in stride_icon_names:
            full_icons.append(os.path.join(ICONS_FULL, fname))
    return full_icons

def random_light_color():
    return random.choice(BG_COLORS)

def random_position(img_w, img_h, icon_w, icon_h, placed_boxes):
    tries = 0
    while tries < 100:
        x = random.randint(0, img_w - icon_w)
        y = random.randint(0, img_h - icon_h)
        box = (x, y, x+icon_w, y+icon_h)
        overlap = any(
            not (box[2] < pb[0] or box[0] > pb[2] or box[3] < pb[1] or box[1] > pb[3])
            for pb in placed_boxes
        )
        if not overlap:
            return x, y, box
        tries += 1
    return None, None, None

def get_icon_edge_point(center, box, toward):
    # Retorna um ponto na borda do bounding box na direção de 'toward'
    cx, cy = center
    x1, y1, x2, y2 = box
    tx, ty = toward
    # Calcula direção
    dx = tx - cx
    dy = ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    # Normaliza direção
    norm = (dx**2 + dy**2) ** 0.5
    dx /= norm
    dy /= norm
    # Move do centro até sair do box
    px, py = cx, cy
    while x1 < px < x2 and y1 < py < y2:
        px += dx
        py += dy
        # Limita para não sair da imagem
        if not (0 <= px < IMG_SIZE[0] and 0 <= py < IMG_SIZE[1]):
            break
    # Retorna ponto na borda
    px = min(max(int(px), x1), x2)
    py = min(max(int(py), y1), y2)
    return px, py

def draw_connections(draw, centers, placed_boxes):
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            # Sempre tenta conectar
            c1, c2 = centers[i], centers[j]
            b1, b2 = placed_boxes[i], placed_boxes[j]
            # Ajusta início/fim para a borda do ícone
            p1 = get_icon_edge_point(c1, b1, c2)
            p2 = get_icon_edge_point(c2, b2, c1)
            if random.random() < 0.8:
                # 80% em L
                mid = (p2[0], p1[1]) if random.random() < 0.5 else (p1[0], p2[1])
                segments = [(p1, mid), (mid, p2)]
            else:
                # 20%: curva em 90 graus (quadrática)
                if random.random() < 0.5:
                    ctrl = (p1[0], p2[1])
                else:
                    ctrl = (p2[0], p1[1])
                segments = [(p1, ctrl, p2)]
            crosses_icon = False
            for seg in segments:
                if len(seg) == 2:
                    for box in placed_boxes:
                        if line_intersects_box(seg[0], seg[1], box):
                            crosses_icon = True
                            break
                else:
                    points = bezier_curve(seg[0], seg[1], seg[2], steps=10)
                    for k in range(len(points)-1):
                        for box in placed_boxes:
                            if line_intersects_box(points[k], points[k+1], box):
                                crosses_icon = True
                                break
                        if crosses_icon:
                            break
                if crosses_icon:
                    break
            if not crosses_icon:
                for seg in segments:
                    if len(seg) == 2:
                        draw.line([seg[0], seg[1]], fill=(180,180,180), width=3)
                    else:
                        points = bezier_curve(seg[0], seg[1], seg[2], steps=20)
                        draw.line(points, fill=(180,180,180), width=3)

def save_yolo_label(label_path, labels, img_w, img_h):
    with open(label_path, 'w', encoding='utf-8') as f:
        for cls_idx, x1, y1, x2, y2 in labels:
            # YOLO: class cx cy w h (normalizado)
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def bezier_curve(p0, p1, p2, steps=20):
    # Quadratic Bezier: p0, p1 (control), p2
    return [(
        int((1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]),
        int((1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1])
    ) for t in [i/steps for i in range(steps+1)]]

def line_intersects_box(p1, p2, box):
    x1, y1, x2, y2 = box
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for e1, e2 in edges:
        if lines_intersect(p1, p2, e1, e2):
            return True
    if (x1 < p1[0] < x2 and y1 < p1[1] < y2) or (x1 < p2[0] < x2 and y1 < p2[1] < y2):
        return True
    return False

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def lines_intersect(A, B, C, D):
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def main():
    # Prepara pastas
    out_img = os.path.join(OUT_DATASET, 'images')
    out_lbl = os.path.join(OUT_DATASET, 'labels')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    # Coleta ícones
    stride_icons = get_stride_icons()
    stride_icon_names = set(os.path.basename(p) for _, p in stride_icons)
    stride_classes = sorted(list(set(cls for cls, _ in stride_icons)))
    class_to_idx = {cls: i for i, cls in enumerate(stride_classes)}
    full_icons = get_full_icons(stride_icon_names)
    # Gera imagens sintéticas
    for idx in tqdm(range(N_IMAGES)):
        bg = Image.new('RGBA', IMG_SIZE, random_light_color())  # Corrigido para RGBA
        draw = ImageDraw.Draw(bg)
        # Retângulos de fundo
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, IMG_SIZE[0]//2)
            y1 = random.randint(0, IMG_SIZE[1]//2)
            x2 = random.randint(x1+60, IMG_SIZE[0])
            y2 = random.randint(y1+40, IMG_SIZE[1])
            draw.rectangle([x1, y1, x2, y2], fill=random_light_color(), outline=(220,220,220))
        n_stride = random.randint(3, min(len(stride_icons), ICONS_PER_IMG[1]-2))
        max_full = min(len(full_icons), ICONS_PER_IMG[1] - n_stride)
        if max_full >= 1:
            n_full = random.randint(1, max_full)
        else:
            n_full = 0
        total_icons = n_stride + n_full
        # Garante o mínimo de 8
        if total_icons < ICONS_PER_IMG[0]:
            falta = ICONS_PER_IMG[0] - total_icons
            add_stride = min(falta, len(stride_icons)-n_stride)
            n_stride += add_stride
            falta -= add_stride
            n_full += falta
        n_stride = min(n_stride, len(stride_icons))
        n_full = min(n_full, len(full_icons), ICONS_PER_IMG[1]-n_stride)
        icons_this = random.sample(stride_icons, n_stride)
        icons_this_full = random.sample(full_icons, n_full)
        placed_boxes = []
        centers = []
        labels = []
        # Coloca ícones STRIDE (rotulados)
        stride_icon_text_boxes = []
        for cls, icon_path in icons_this:
            icon = Image.open(icon_path).convert('RGBA')
            scale = random.uniform(0.7, 1.2)
            iw, ih = int(icon.width*scale), int(icon.height*scale)
            icon = icon.resize((iw, ih), Image.LANCZOS)
            x, y, box = random_position(IMG_SIZE[0], IMG_SIZE[1], iw, ih, placed_boxes)
            if box is None: continue
            bg.alpha_composite(icon, (x, y))
            placed_boxes.append(box)
            centers.append(((x+iw//2), (y+ih//2)))
            labels.append((class_to_idx[cls], x, y, x+iw, y+ih))
            # --- Adiciona nome do componente abaixo do ícone (50% chance) ---
            if random.random() < 0.5:
                text = cls
                text_draw = ImageDraw.Draw(bg)
                font = None
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = None
                # Calcula tamanho do texto de forma compatível
                if font is not None:
                    try:
                        # Pillow >=8.0.0
                        bbox = text_draw.textbbox((0,0), text, font=font)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except AttributeError:
                        # Pillow <8.0.0
                        text_w, text_h = font.getsize(text)
                else:
                    # Estimativa simples se fonte não disponível
                    text_w, text_h = 8*len(text), 16
                text_x = x + (iw - text_w) // 2
                text_y = y + ih + 2
                text_box = (text_x, text_y, text_x+text_w, text_y+text_h)
                overlaps = any(
                    not (text_box[2] < pb[0] or text_box[0] > pb[2] or text_box[3] < pb[1] or text_box[1] > pb[3])
                    for pb in placed_boxes + stride_icon_text_boxes
                )
                if text_y+text_h < IMG_SIZE[1] and not overlaps:
                    text_draw.text((text_x, text_y), text, fill=(30,30,30), font=font)
                    stride_icon_text_boxes.append(text_box)
        # Coloca ícones FULL (não rotulados)
        for icon_path in icons_this_full:
            icon = Image.open(icon_path).convert('RGBA')
            scale = random.uniform(0.7, 1.2)
            iw, ih = int(icon.width*scale), int(icon.height*scale)
            icon = icon.resize((iw, ih), Image.LANCZOS)
            x, y, box = random_position(IMG_SIZE[0], IMG_SIZE[1], iw, ih, placed_boxes)
            if box is None: continue
            bg.alpha_composite(icon, (x, y))
            placed_boxes.append(box)
            centers.append(((x+iw//2), (y+ih//2)))
        # Conexões
        draw_connections(draw, centers, placed_boxes)
        # Pós-processamento: variações de ruído, borramento e nitidez
        aug_diagram = bg
        # Aplica uma variação aleatória: original, blur, sharpen, ou ambos
        aug_type = random.choice(['none', 'blur', 'sharpen', 'blur_sharpen'])
        if aug_type == 'blur':
            aug_diagram = aug_diagram.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        elif aug_type == 'sharpen':
            enhancer = ImageEnhance.Sharpness(aug_diagram)
            aug_diagram = enhancer.enhance(random.uniform(1.5, 2.5))
        elif aug_type == 'blur_sharpen':
            aug_diagram = aug_diagram.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
            enhancer = ImageEnhance.Sharpness(aug_diagram)
            aug_diagram = enhancer.enhance(random.uniform(1.5, 2.5))
        # Salva imagem e label
        img_name = f'synth_{idx:04d}.jpg'
        label_name = img_name.replace('.jpg', '.txt')
        aug_diagram = aug_diagram.convert('RGB')  # Converte para RGB só ao salvar
        aug_diagram.save(os.path.join(out_img, img_name))
        save_yolo_label(os.path.join(out_lbl, label_name), labels, IMG_SIZE[0], IMG_SIZE[1])
    # Salva classes
    with open(os.path.join(OUT_DATASET, 'classes.txt'), 'w', encoding='utf-8') as f:
        for cls in stride_classes:
            f.write(cls+'\n')
    # Gera data.yaml
    with open(os.path.join(OUT_DATASET, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"train: {out_img}\n")
        f.write(f"val: {out_img}\n")
        f.write(f"nc: {len(stride_classes)}\n")
        f.write(f"names: {stride_classes}\n")
    print('Dataset sintético YOLO criado com sucesso!')

if __name__ == '__main__':
    main()
