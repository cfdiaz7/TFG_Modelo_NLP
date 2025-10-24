import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Paths y configuraciones
MODEL_PATH = "./models/final_model"
DATASET_PATH = r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_ABL_BUTTERFLY"

label_map = {0: "Ant", 1: "Bee", 2: "Leech", 3: "Butterfly"}
id2label = {v: k for k, v in label_map.items()}

# Cargar modelo y tokenizer
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model.eval()

# Cargar dataset de test
dataset = DatasetDict.load_from_disk(DATASET_PATH)
test_dataset = dataset["test"]

print(f"✅ Test samples loaded: {len(test_dataset)}")

# Realizar predicciones
y_true = []
y_pred = []

for example in test_dataset:
    text = example["text"]
    label = example["labels"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()

    y_true.append(label)
    y_pred.append(pred)

# Métricas y análisis
print("\n📈 Classification Report:")
report = classification_report(y_true, y_pred, target_names=label_map.values(), digits=3)
print(report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.values(),
            yticklabels=label_map.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – ABL + Butterfly Model")
plt.tight_layout()
plt.show()

# Accuracy global
acc = np.trace(cm) / np.sum(cm)
print(f"✅ Global Accuracy: {acc:.3f}")
