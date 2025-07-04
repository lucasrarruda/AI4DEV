import streamlit as st
from PIL import Image
from io import BytesIO
# from yolo_detector import detect_components  # sua funcao de detecao
# from classifier import classify_icon         # sua funcao de classificacao
# from stride_map import STRIDE_MAP           # dict {"Componente": ["Spoofing", "Tampering", ...]}

st.set_page_config(page_title="Analisador STRIDE de Diagramas AWS", layout="wide")
st.title("🔐 Analisador de Ameaças STRIDE em Diagramas AWS")

uploaded_file = st.file_uploader("📤 Envie o diagrama da arquitetura (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Diagrama enviado", use_column_width=True)

    st.subheader("🔎 Componentes detectados e análise STRIDE")

    # MOCK: Componentes detectados e STRIDE
    componentes_detectados = [
        "EC2", "S3", "Lambda", "RDS", "IAM"
    ]
    STRIDE_MAP = {
        "EC2": ["Spoofing", "Tampering"],
        "S3": ["Information Disclosure", "Tampering"],
        "Lambda": ["Elevation of Privilege"],
        "RDS": ["Denial of Service", "Information Disclosure"],
        "IAM": ["Spoofing", "Elevation of Privilege"]
    }
    stride_riscos = {}
    for classe in componentes_detectados:
        if classe in STRIDE_MAP:
            for ameaca in STRIDE_MAP[classe]:
                stride_riscos.setdefault(ameaca, []).append(classe)

    st.write("### 🧩 Componentes detectados:")
    st.write(sorted(set(componentes_detectados)))

    st.write("### 🛡 Ameaças STRIDE identificadas:")
    if stride_riscos:
        for tipo, comps in stride_riscos.items():
            st.markdown(f"**{tipo}**: {', '.join(sorted(set(comps)))}")
    else:
        st.info("Nenhuma ameaça STRIDE identificada com base nos componentes detectados.")
