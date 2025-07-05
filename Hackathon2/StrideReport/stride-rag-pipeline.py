# Requisitos: pip install sentence-transformers faiss-cpu transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import faiss
import numpy as np
import json

# ==============================
# 1. BASE DE CONHECIMENTO SIMULADA
# ==============================
DOCUMENTOS = [
    {"componente": "Amazon API Gateway", "texto": "API Gateway pode ser explorado por spoofing se não houver autenticação adequada nas rotas de entrada."},
    {"componente": "AWS Lambda", "texto": "Lambda pode ter risco de escalonamento de privilégio se permissões IAM estiverem amplas."},
    {"componente": "Amazon S3", "texto": "S3 frequentemente sofre com divulgação indevida de dados quando buckets ficam publicamente acessíveis."},
    {"componente": "IAM", "texto": "IAM mal configurado é um vetor crítico de spoofing e privilege escalation."}
]

# ==============================
# 2. EMBEDDING E INDEXAÇÃO COM FAISS
# ==============================
model_emb = SentenceTransformer('all-MiniLM-L6-v2')
frases = [doc['texto'] for doc in DOCUMENTOS]
embeddings = model_emb.encode(frases, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ==============================
# 3. FUNÇÃO DE BUSCA CONTEXTUAL (RAG)
# ==============================
def recuperar_contexto(componentes, top_k=3):
    contexto = []
    for componente in componentes:
        consultas = [doc['texto'] for doc in DOCUMENTOS if doc['componente'] == componente]
        if not consultas:
            continue
        emb = model_emb.encode(consultas[0], convert_to_numpy=True).reshape(1, -1)
        dists, idxs = index.search(emb, top_k)
        for i in idxs[0]:
            contexto.append(frases[i])
    return "\n".join(set(contexto))

# ==============================
# 4. GERAÇÃO COM LLM LOCAL VIA PIPELINE (usando modelo leve)
# ==============================
def gerar_relatorio(componentes, contexto):
    prompt = f"""
Você é um analista de segurança da informação. Com base nos componentes abaixo e no conteúdo técnico a seguir, gere um relatório explicando:
1. Ameaças detectadas (por STRIDE)
2. Justificativa técnica
3. Recomendações

Componentes:
{', '.join(componentes)}

Contexto técnico:
{contexto}

Relatório:
"""
    # Usa o modelo phi-2
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    resposta = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]['generated_text']
    return resposta

# ==============================
# 5. EXEMPLO DE USO
# ==============================
if __name__ == "__main__":
    componentes_detectados = ["Amazon API Gateway", "AWS Lambda", "Amazon S3"]
    contexto = recuperar_contexto(componentes_detectados)
    relatorio = gerar_relatorio(componentes_detectados, contexto)
    print("\n===== RELATÓRIO GERADO =====\n")
    print(relatorio)
