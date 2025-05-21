import os
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Cargar y preparar dataset
def load_data():
    df = pd.read_csv("data/dataset.csv")

    label_map = {label: i for i, label in enumerate(df['label'].unique())}
    df['label'] = df['label'].map(label_map)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset, label_map


def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)


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
    # Comprobar si hay GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar tokenizer y modelo
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Cambia num_labels según el número real de clases en tu problema
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    model.to(device)

    # Cargar y tokenizar datos
    train_data, val_data, label_map = load_data()
    train_data = tokenize_data(train_data, tokenizer)
    val_data = tokenize_data(val_data, tokenizer)

    # Eliminar columnas innecesarias
    if "__index_level_0__" in train_data.column_names:
        train_data = train_data.remove_columns(["__index_level_0__"])
    if "__index_level_0__" in val_data.column_names:
        val_data = val_data.remove_columns(["__index_level_0__"])

    # Definir argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Entrenador con early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Entrenar
    trainer.train()

    # Guardar el modelo y tokenizer
    model.save_pretrained("./models/final_model")
    tokenizer.save_pretrained("./models/final_model")
    print("Modelo entrenado y guardado.")


if __name__ == "__main__":
    main()
