import os
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Cargar dataset combinado
def load_data():
    dataset_path = r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_ABL_BUTTERFLY"
    dataset_dict = DatasetDict.load_from_disk(dataset_path)

    # Cargar splits
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]

    # Mapa de etiquetas
    label_map = {0: "Ant", 1: "Bee", 2: "Leech", 3: "Butterfly"}

    print(f"✅ Dataset cargado desde: {dataset_path}")
    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    return train_dataset, val_dataset, label_map

# Tokenización
def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)

# Métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# Entrenamiento
def main():
    # Detectar GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Cargar tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Cargar datos
    train_data, val_data, label_map = load_data()

    # Tokenizar
    train_data = tokenize_data(train_data, tokenizer)
    val_data = tokenize_data(val_data, tokenizer)

    # Eliminar columnas innecesarias (por si las hay)
    for col in ["__index_level_0__", "id"]:
        if col in train_data.column_names:
            train_data = train_data.remove_columns([col])
        if col in val_data.column_names:
            val_data = val_data.remove_columns([col])

    # Crear modelo con el número correcto de clases
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_map)
    )
    model.to(device)

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2
    )

    # Trainer con early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Entrenamiento
    print("🚀 Starting training...")
    trainer.train()

    # Guardar modelo final
    model.save_pretrained("./models/final_model")
    tokenizer.save_pretrained("./models/final_model")
    print("✅ Modelo entrenado y guardado en ./models/final_model")

# Ejecutar script
if __name__ == "__main__":
    main()
