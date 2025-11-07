import os

import streamlit as st
from PIL import Image
import pandas as pd
import tensorflow as tf
import gdown

from utils import predict

# ==========================
# 🔧 CONFIGURACIÓN BÁSICA
# ==========================
st.set_page_config(
    page_title="Clasificador de Aves",
    page_icon="🦜",
    layout="wide",
)

# ==========================
# 📁 MANEJO DE MODELOS (.keras desde Drive)
# ==========================
MODELS_DIR = "models"

# ✅ IDs de tus modelos en Google Drive
VGG16_ID = "1Jppl1AHwIZvZ74t3RF4tjFXOzA_EOOlU"
RESNET50_ID = "1xbrx9aIcgLKVb8d8MQG6ZsU2yPwBywbn"


def descargar_modelo_keras(file_id, nombre_local):
    """
    Descarga un archivo .keras desde Google Drive a la carpeta models/
    si no existe localmente. Devuelve la ruta al archivo.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    ruta_local = os.path.join(MODELS_DIR, nombre_local)
    if not os.path.exists(ruta_local):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Descargando {nombre_local}..."):
            gdown.download(url, ruta_local, quiet=False)
    return ruta_local


@st.cache_resource
def load_selected_model(model_name: str):
    """
    Carga el modelo seleccionado (VGG16 o ResNet50) desde .keras.
    """
    if model_name == "VGG16":
        modelo_path = descargar_modelo_keras(VGG16_ID, "vgg16_model.keras")
    else:
        modelo_path = descargar_modelo_keras(RESNET50_ID, "resnet50_model.keras")

    modelo = tf.keras.models.load_model(modelo_path, compile=False)
    return modelo


# ==========================
# 🐦 INFO DE LAS ESPECIES
# ==========================
BIRD_INFO = {
    "Coereba_flaveola": {
        "common": "Mielero común (Bananaquit)",
        "scientific": "Coereba flaveola",
        "description": "Ave pequeña, muy activa, que se alimenta de néctar y frutas en jardines y bordes de bosque en zonas tropicales."
    },
    "Icterus_nigrogularis": {
        "common": "Gonzalito / Bolsero amarillo",
        "scientific": "Icterus nigrogularis",
        "description": "Ictérido de plumaje amarillo intenso con máscara y garganta oscuras, frecuente en áreas abiertas, cultivos y jardines."
    },
    "Oryzoborus_angolensis": {
        "common": "Arrocero buchicastaño",
        "scientific": "Oryzoborus angolensis",
        "description": "Semillero robusto; el macho es oscuro con pecho castaño. Vive en zonas abiertas y pastizales ricos en semillas."
    },
    "Piculus_chrysochloros": {
        "common": "Carpintero dorado",
        "scientific": "Piculus chrysochloros",
        "description": "Carpintero de tonos verdes y dorados que busca insectos bajo la corteza de los árboles en bosques y bordes de selva."
    },
    "Psarocolius_decumanus": {
        "common": "Oropéndola crestada",
        "scientific": "Psarocolius decumanus",
        "description": "Ave grande, de cuerpo oscuro y cola amarilla brillante; forma colonias y construye nidos colgantes en árboles altos."
    },
    "Saltator_coerulescens": {
        "common": "Saltador grisáceo",
        "scientific": "Saltator coerulescens",
        "description": "Pájaro robusto de tonos grisáceos y azulados, habitual en jardines y matorrales; consume frutos y semillas."
    },
    "Terenotriccus_erythrurus": {
        "common": "Mosquerito cola rojiza",
        "scientific": "Terenotriccus erythrurus",
        "description": "Mosquerito muy pequeño, de cuerpo pardo y cola rojiza, que captura insectos al vuelo en el sotobosque del bosque húmedo."
    },
    "Troglodytes_monticola": {
        "common": "Cucarachero de Santa Marta",
        "scientific": "Troglodytes monticola",
        "description": "Cucarachero endémico de la Sierra Nevada de Santa Marta; vive en matorrales y bordes de bosque de alta montaña."
    },
    "Turdus_fuscater": {
        "common": "Mirla patinaranja / Zorzal grande",
        "scientific": "Turdus fuscater",
        "description": "El túrdido más grande de Sudamérica; común en ciudades andinas, parques y jardines, con patas y pico amarillentos."
    },
    "Turdus_serranus": {
        "common": "Mirla serrana",
        "scientific": "Turdus serranus",
        "description": "Zorzal de montaña; el macho es negro brillante con pico y patas anaranjadas. Habita bosques nublados y montanos."
    },
}

# ==========================
# 🎛️ SIDEBAR
# ==========================
with st.sidebar:
    st.title("🦜 Clasificador de Aves")
    st.markdown(
        "Selecciona el modelo de *deep learning* y revisa las especies que puede reconocer."
    )

    model_name = st.selectbox(
        "Modelo de clasificación",
        ["VGG16", "ResNet50"],
        help="Puedes comparar el desempeño de diferentes modelos con la misma imagen."
    )

    st.markdown("### 🐦 Especies reconocidas")
    for key, info in BIRD_INFO.items():
        st.markdown(
            f"- **{info['common']}**  \n"
            f"  <span style='font-size:12px;'>*{info['scientific']}*</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption(
        "💡 Consejo: usa fotos donde el ave esté centrada, bien iluminada y enfocada para mejores resultados."
    )

# Cargar el modelo seleccionado
model = load_selected_model(model_name)

# ==========================
# 🖼️ INTERFAZ PRINCIPAL
# ==========================
st.markdown("## 📸 Sube una foto para clasificar el ave")

col_left, col_right = st.columns([1.2, 1])

uploaded_file = col_left.file_uploader(
    "Sube una imagen (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col_left.image(img, caption="Imagen cargada", use_column_width=True)

    if col_left.button("🔍 Clasificar ave"):
        with st.spinner(f"Analizando la imagen con el modelo {model_name}..."):
            results = predict(model, img, model_type=model_name)

        if not results:
            st.error("No se obtuvieron predicciones. Revisa la función `predict` en `utils.py`.")
        else:
            # Ordenar por probabilidad
            results = sorted(results, key=lambda x: x[1], reverse=True)
            best_name, best_prob = results[0]

            best_info = BIRD_INFO.get(
                best_name,
                {
                    "common": best_name.replace("_", " ").title(),
                    "scientific": best_name.replace("_", " "),
                    "description": "No hay descripción disponible para esta especie."
                }
            )

            # 🔝 Tarjeta principal con la mejor predicción
            with col_right:
                st.markdown("### ✅ Mejor predicción")
                st.success(
                    f"Es muy probable que sea **{best_info['common']}** "
                    f"(_{best_info['scientific']}_)\n\n"
                    f"**Confianza del modelo:** {best_prob*100:.2f}%"
                )
                st.markdown("#### 📝 Descripción")
                st.write(best_info["description"])

            # ==========================
            # 📊 GRÁFICA DE BARRAS + LISTA DETALLADA
            # ==========================
            st.markdown("### 📊 Top predicciones del modelo")

            labels = []
            probs = []
            for name, prob in results:
                info = BIRD_INFO.get(
                    name,
                    {
                        "common": name.replace("_", " ").title(),
                        "scientific": name.replace("_", " "),
                    }
                )
                labels.append(info["common"])
                probs.append(prob * 100)

            df = pd.DataFrame(
                {"Especie": labels, "Probabilidad (%)": probs}
            ).set_index("Especie")

            st.bar_chart(df)

            st.markdown("#### 📋 Detalle de predicciones")
            for (name, prob) in results:
                info = BIRD_INFO.get(
                    name,
                    {
                        "common": name.replace("_", " ").title(),
                        "scientific": name.replace("_", " "),
                        "description": "Sin descripción disponible."
                    }
                )
                with st.container():
                    st.markdown(
                        f"**{info['common']}**  "
                        f"(_{info['scientific']}_)  \n"
                        f"- Probabilidad: **{prob*100:.2f}%**"
                    )
                    st.caption(info["description"])

else:
    col_right.info(
        "👈 Sube una imagen en la parte izquierda para ver aquí la mejor predicción, "
        "el nombre real del ave y una breve descripción, junto con barras de probabilidad."
    )

