import transformers
import os
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Cargar y preparar dataset
def load_data():
    df = pd.read_csv(r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\dataset.csv")

    # Definir las clases
    label_list = ['Ant', 'Bee', 'Butterfly', 'Leech']
    label_map = {label: i for i, label in enumerate(label_list)}

    # Mapear las etiquetas a enteros
    df['label'] = df['label'].map(label_map)

    # Dividir dataset
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Convertir a Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset, label_map

# Tokenizar dataset
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

def main():
    # Comprueba si hay GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carga tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Cargar datos
    train_data, val_data, label_map = load_data()

    # Tokenizar datos
    train_data = tokenize_data(train_data, tokenizer)
    val_data = tokenize_data(val_data, tokenizer)

    # Eliminar columnas innecesarias
    for col in ["__index_level_0__"]:
        if col in train_data.column_names:
            train_data = train_data.remove_columns([col])
        if col in val_data.column_names:
            val_data = val_data.remove_columns([col])

    # Crear modelo con número correcto de clases
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_map)
    )
    model.to(device)

    # Definir argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./models",
        eval_strategy="epoch",
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

    # Entrenamiento con Trainer y early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Entrenamiento del modelo
    trainer.train()

    # Guardar modelo y tokenizer
    model.save_pretrained("./models/final_model")
    tokenizer.save_pretrained("./models/final_model")
    print("Modelo entrenado y guardado.")

# Ejecutar script
if __name__ == "__main__":
    main()
