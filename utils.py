import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

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

def load_model(model_path):
    return tf.saved_model.load(model_path)

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# def predict(model, img):
#     input_tensor = preprocess_image(img)
#     infer = model.signatures["serving_default"]
#     output = infer(tf.constant(input_tensor))["output_0"]
#     predictions = output.numpy()[0]
#     top_indices = predictions.argsort()[-3:][::-1]
#     return [(class_names[i], float(predictions[i])) for i in top_indices]

def predict(model, img, model_type="vgg16"):
    # Preprocesamiento según el modelo
    if model_type.lower() == "resnet50":
        input_tensor = preprocess_image_resnet(img)
    else:
        input_tensor = preprocess_image(img)
    
    infer = model.signatures["serving_default"]
    output = infer(tf.constant(input_tensor))["output_0"]
    
    predictions = output.numpy()[0]
    top_indices = predictions.argsort()[-3:][::-1]
    return [(class_names[i], float(predictions[i])) for i in top_indices]

def preprocess_image_resnet(img):
    # Redimensionar a 224x224
    img = img.resize((224, 224))
    img_array = np.array(img)
    # Agregar dimensión batch
    img_array = np.expand_dims(img_array, axis=0)
    # Normalizar a [-1, 1] como espera ResNet50
    img_array = preprocess_input(img_array)
    return img_array.astype(np.float32)
