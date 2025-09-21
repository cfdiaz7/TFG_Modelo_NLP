import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch.nn.functional as F

# Cargar modelo y tokenizer
model_path = "./models/final_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Colocar en modo evaluación
model.eval()

# Mapeo de etiquetas
label_map = {0: "Ant", 1: "Bee", 2: "Butterfly", 3: "Leech"}

# Frases de ejemplo
texts = [
    "I like working with others on creative projects.",
    "I prefer doing tasks on my own.",
    "I enjoy leading groups and coordinating people.",
    "I avoid social interactions and focus on personal tasks."
]

# Función para predecir
def predict(text):
    # Tokenizar
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(logits, dim=-1).item()

    # Mostrar resultados
    print(f"Texto: {text}")
    print(f"Predicción: {label_map[pred_idx]}")
    print("Probabilidades:")
    for i, label in label_map.items():
        print(f"  {label}: {probs[0][i]:.4f}")
    print("-" * 50)

# Ejecutar predicciones
for t in texts:
    predict(t)
