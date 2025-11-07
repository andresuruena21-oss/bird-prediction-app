import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# 🐦 Lista de clases (en el mismo orden del entrenamiento)
class_names = [
    "Coereba_flaveola",
    "Icterus_nigrogularis",
    "Oryzoborus_angolensis",
    "Piculus_chrysochloros",
    "Psarocolius_decumanus",
    "Saltator_coerulescens",
    "Terenotriccus_erythrurus",
    "Troglodytes_monticola",
    "Turdus_fuscater",
    "Turdus_serranus"
]

# ===================================================
# 🧠 Cargar modelo (solo si usas SavedModel antiguo)
# ===================================================
def load_model(model_path):
    """Carga un modelo guardado en formato SavedModel (no usado para .keras)."""
    return tf.saved_model.load(model_path)

# ===================================================
# 🧼 Preprocesamiento de imágenes
# ===================================================
def preprocess_image(img):
    """Preprocesamiento estándar para modelos tipo VGG o similares."""
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def preprocess_image_resnet(img):
    """Preprocesamiento para modelos ResNet (usa preprocess_input)."""
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array.astype(np.float32)

# ===================================================
# 🔮 Predicción compatible (.keras, .h5 y SavedModel)
# ===================================================
def predict(model, img, model_type="vgg16"):
    """
    Realiza la predicción sobre una imagen usando el modelo dado.
    Soporta modelos .keras / .h5 (modernos) y SavedModel (antiguos con .signatures).
    """

    # --- Preprocesamiento según el modelo seleccionado ---
    if model_type.lower() == "resnet50":
        input_tensor = preprocess_image_resnet(img)
    else:
        input_tensor = preprocess_image(img)

    # --- Predicción según tipo de modelo ---
    try:
        if hasattr(model, "signatures"):  # modelos tipo SavedModel
            infer = model.signatures["serving_default"]
            output_key = list(infer.structured_outputs.keys())[0]
            output = infer(tf.constant(input_tensor))[output_key]
            preds = output.numpy()[0]
        else:  # modelos .keras o .h5
            preds = model.predict(input_tensor)[0]
    except Exception as e:
        raise RuntimeError(f"Error al realizar la predicción: {e}")

    # --- Top 3 predicciones ---
    top_indices = preds.argsort()[-3:][::-1]
    results = [(class_names[i], float(preds[i])) for i in top_indices]
    return results
