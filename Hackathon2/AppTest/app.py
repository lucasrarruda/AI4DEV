import streamlit as st
from PIL import Image
from io import BytesIO
from stride_detection import analisar_ameacas_stride

st.set_page_config(page_title="Analisador STRIDE de Diagramas AWS", layout="wide")
st.title("🔐 Analisador de Ameaças STRIDE em Diagramas AWS")

uploaded_file = st.file_uploader("📤 Envie o diagrama da arquitetura (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Diagrama enviado", use_column_width=True)

    st.subheader("🔎 Componentes detectados e análise STRIDE")

    # YOLO: detecta componentes na imagem enviada
    from yolo_detector import detect_components_pil
    import cv2
    import numpy as np
    componentes_detectados = []
    stride_riscos = {}
    annotated_img = image.copy()
    detections = detect_components_pil(image)
    # Desenha bounding boxes e labels
    if detections:
        img_cv = np.array(annotated_img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["classe"]
            conf = det["conf"]
            componentes_detectados.append(label)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
            # --- Desenha fundo do texto ---
            text = f"{label} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x1
            text_y = max(y1 - 10, th + 2)
            cv2.rectangle(img_cv, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), (255,255,255), -1)
            cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, (0,0,255), thickness, cv2.LINE_AA)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        annotated_img = Image.fromarray(img_cv)
    
    stride_riscos = analisar_ameacas_stride(componentes_detectados)
    st.image(annotated_img, caption="🟩 Componentes detectados", use_column_width=True)
    st.write("### 🧩 Componentes detectados:")
    st.write(sorted(set(componentes_detectados)))
    st.write("### 🛡 Ameaças STRIDE identificadas:")
    if stride_riscos:
        for tipo, comps in stride_riscos.items():
            st.markdown(f"**{tipo}**: {', '.join(sorted(set(comps)))}")
    else:
        st.info("Nenhuma ameaça STRIDE identificada com base nos componentes detectados.")
