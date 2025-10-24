import pandas as pd
from datasets import Dataset, DatasetDict

butterfly_data = [
    {"id": "btrfly_01", "text": "In nature, there is no separation between design, engineering, and art.", "labels": 3},
    {"id": "btrfly_02", "text": "A structure must not only work; it must also inspire.", "labels": 3},
    {"id": "btrfly_03", "text": "Creativity lies at the intersection of logic and beauty.", "labels": 3},
    {"id": "btrfly_04", "text": "Technology should serve the imagination, not limit it.", "labels": 3},
    {"id": "btrfly_05", "text": "Engineering is a kind of poetry written with materials.", "labels": 3},
    {"id": "btrfly_06", "text": "The best design feels inevitable, yet surprising.", "labels": 3},
    {"id": "btrfly_07", "text": "Art and science are two sides of the same curiosity.", "labels": 3},
    {"id": "btrfly_08", "text": "Beauty is functionality in its most expressive form.", "labels": 3},
    {"id": "btrfly_09", "text": "To create is to see connections others overlook.", "labels": 3},
    {"id": "btrfly_10", "text": "Good design is where emotion meets precision.", "labels": 3},
]

# Convertir a DataFrame
df = pd.DataFrame(butterfly_data)
dataset = Dataset.from_pandas(df)

# Dividir 90% / 10%
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

# Crear DatasetDict misma estructura que el Dataset Javier
dataset_dict = DatasetDict({
    "train": dataset_split["train"],
    "test": dataset_split["test"]
})

# Guardar en disco
output_path = r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_BUTTERFLY"
dataset_dict.save_to_disk(output_path)

print("✅ Dataset BUTTERFLY creado en:", output_path)
print(dataset_dict)