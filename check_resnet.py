import tensorflow as tf

# Cargar tu modelo ResNet50
model = tf.saved_model.load("models/resnet50_model")

# Imprimir las firmas disponibles
print("Firmas disponibles en el modelo:")
print(model.signatures)

# Acceder a la firma por defecto
infer = model.signatures["serving_default"]

# Imprimir las salidas estructuradas
print("\nEstructura de las salidas:")
print(infer.structured_outputs)
