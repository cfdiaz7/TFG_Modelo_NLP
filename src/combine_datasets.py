from datasets import DatasetDict, load_from_disk, concatenate_datasets

# Cargar datasets existentes
path_abl = r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_ABL_trad_final"
path_butterfly = r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_BUTTERFLY"

dataset_abl = DatasetDict.load_from_disk(path_abl)
dataset_butterfly = DatasetDict.load_from_disk(path_butterfly)

# Combinar train y test
train_combined = concatenate_datasets([dataset_abl["train"], dataset_butterfly["train"]])
test_combined = concatenate_datasets([dataset_abl["test"], dataset_butterfly["test"]])

# Crear nuevo dataset con las 4 tribus
dataset_total = DatasetDict({
    "train": train_combined,
    "test": test_combined
})

# Guardar en disco
output_path = r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_ABL_BUTTERFLY"
dataset_total.save_to_disk(output_path)

print("✅ Dataset combinado ABL + BUTTERFLY guardado en:", output_path)
print(dataset_total)
