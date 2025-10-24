# TFG_Modelo_NLP

Este repositorio contiene el código Python para el proyecto de análisis y entrenamiento de modelos NLP del TFG.
Este proyecto implementa un modelo de clasificación de texto usando la biblioteca `transformers` de Hugging Face y PyTorch. Se entrena un modelo DistilBERT para clasificar textos en distintas categorías, basado en datasets preparados para el TFG.

---

## Estructura del proyecto

TFG_Modelo_NLP/
- `src/` → Código fuente en Python
  - `train.py` → Entrenamiento del modelo
  - `test_model.py` → Evaluación del modelo
  - `analyze_model_results.py` → Análisis de resultados
  - `combine_datasets.py` → Combinar datasets
  - `create_butterfly_dataset.py` → Creación de dataset
- `.gitignore` → Archivos ignorados
- `.gitattributes` → Configuración de Git LFS (si se usa)
- `requirements.txt` → Dependencias Python

---

## Instalación

1. Clonar el repositorio: 

```bash
git clone https://github.com/cfdiaz7/TFG_Modelo_NLP.git
```

Crea y activa un entorno virtual:

    python -m venv .venv
# Windows
    .venv\Scripts\activate
# Linux/Mac
    source .venv/bin/activate

Instala las dependencias:

    pip install -r requirements.txt

Ejecuta el script para entrenar el modelo:

    python src/train.py
    python src/test_model.py

Autor

Carlos Fernández Díaz
,cfdiaz7
