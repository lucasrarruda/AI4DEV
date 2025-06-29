import os
import shutil
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

# Caminhos
ICON_DIR = "C:/Projects/AI4DEV/Hackathon/aws_icons"
OUTPUT_DIR = "C:/Projects/AI4DEV/Hackathon/aws_icons_classified"

# Top 20 ícones mais usados em diagramas AWS
TOP_20_KEYWORDS = [
    "ec2", "s3", "elb", "rds", "vpc", "lambda", "apigateway",
    "dynamodb", "cloudfront", "route53", "sns", "sqs", "iam",
    "cloudwatch", "stepfunctions", "eks", "codepipeline", "glue",
    "kinesis", "elasticache"
]

# Mapeamento de palavras-chave para categorias sugeridas
CATEGORY_MAP = {
    "ec2": "compute",
    "lambda": "compute",
    "elb": "networking",
    "vpc": "networking",
    "apigateway": "application",
    "stepfunctions": "application",
    "rds": "database",
    "dynamodb": "database",
    "elasticache": "database",
    "s3": "storage",
    "cloudfront": "networking",
    "route53": "networking",
    "sns": "messaging",
    "sqs": "messaging",
    "iam": "security",
    "cloudwatch": "observability",
    "eks": "compute",
    "codepipeline": "devtools",
    "glue": "analytics",
    "kinesis": "messaging"
}

# Cria pasta de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cria as pastas de saída por tipo
for tipo in set(CATEGORY_MAP.values()):
    os.makedirs(os.path.join(OUTPUT_DIR, tipo), exist_ok=True)

# Copia os arquivos que contém palavras-chave dos top 20 para as pastas de tipo
for file in os.listdir(ICON_DIR):
    if not file.lower().endswith(".png"):
        continue
    lower_name = file.lower()
    for keyword in TOP_20_KEYWORDS:
        if keyword in lower_name:
            tipo = CATEGORY_MAP.get(keyword, "outros")
            dst_dir = os.path.join(OUTPUT_DIR, tipo)
            os.makedirs(dst_dir, exist_ok=True)
            src = os.path.join(ICON_DIR, file)
            dst = os.path.join(dst_dir, file)
            shutil.copy2(src, dst)
            break

print("✅ Ícones dos Top 20 componentes AWS copiados para:", OUTPUT_DIR)

# Balanceamento das classes por pasta (duplicando para igualar à maior)
# Conta quantos arquivos tem em cada pasta
class_counts = defaultdict(list)
for tipo in set(CATEGORY_MAP.values()):
    pasta = os.path.join(OUTPUT_DIR, tipo)
    if not os.path.exists(pasta):
        continue
    for f in os.listdir(pasta):
        if f.lower().endswith('.png'):
            class_counts[tipo].append(os.path.join(pasta, f))

# Encontra a maior quantidade de arquivos entre as classes
if class_counts:
    max_count = max(len(files) for files in class_counts.values() if files)
    # Para cada classe, duplica arquivos até igualar à maior
    for tipo, files in class_counts.items():
        if len(files) < max_count and len(files) > 0:
            needed = max_count - len(files)
            for i in range(needed):
                src = files[i % len(files)]
                base, ext = os.path.splitext(os.path.basename(src))
                dst = os.path.join(os.path.dirname(src), f"{base}_dup{i+1}{ext}")
                shutil.copy2(src, dst)
    print(f"Balanceamento realizado: todas as classes agora possuem {max_count} arquivos.")
else:
    print("Nenhuma classe encontrada para balanceamento.")
