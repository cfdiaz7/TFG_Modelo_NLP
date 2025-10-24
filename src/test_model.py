import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch.nn.functional as F

# Cargar modelo entrenado
model_path = "./models/final_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

model.eval()  # modo evaluación

# Mapeo de etiquetas
label_map = {
    0: "Ant",       
    1: "Bee",
    2: "Leech",
    3: "Butterfly"
}

# Frases de prueba
texts = [
    "I like working with others on creative projects.",
    "I prefer doing tasks on my own and following strict routines.",
    "I enjoy leading groups and coordinating people to reach goals.",
    "I often look for beauty and inspiration in everything I design.",
    "I only focus on results and don't care about collaboration.",
    "I find meaning in combining technology and art."
]

# Función de predicción
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()

    print(f"\n🧠 Text: {text}")
    print(f"🎯 Predicted tribe: {label_map[pred_idx]}")
    print("📊 Probabilities:")
    for i, label in label_map.items():
        print(f"  {label}: {probs[0][i]:.4f}")
    print("-" * 60)

# Ejecutar predicciones
if __name__ == "__main__":
    print("🔍 Testing model predictions...\n")
    for t in texts:
        predict(t)

