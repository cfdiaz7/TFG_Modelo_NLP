# TFG_Modelo_NLP

Proyecto del Trabajo de Fin de Grado para desarrollar un modelo de Procesamiento de Lenguaje Natural (NLP) basado en DistilBERT.

---

## Descripción

Este proyecto implementa un modelo de clasificación de texto usando la biblioteca `transformers` de Hugging Face y PyTorch. Se entrena un modelo DistilBERT para clasificar textos en distintas categorías, basado en datasets preparados para el TFG.

---

## Estructura del proyecto

TFG_Modelo_NLP/
- data/              <- Datos de entrenamiento/pruebas
- models/            <- Modelos entrenados
- src/               <- Código fuente (scripts)
  - train.py         <- Script para entrenar el modelo
  - utils.py         <- Funciones auxiliares
- requirements.txt
- README.md

---

## Requisitos

- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- pandas

Cómo ejecutar

Clona el repositorio:

    git clone https://github.com/cfdiaz7/TFG_Modelo_NLP.git
    cd TFG_Modelo_NLP

Crea y activa un entorno virtual:

    python -m venv .venv
# Windows
    .venv\Scripts\activate
# Linux/Mac
    source .venv/bin/activate

Instala las dependencias:

    pip install -r requirements.txt

Ejecuta el script para entrenar el modelo:

    python train_model.py

Notas

    La carpeta .venv está excluida del repositorio para evitar subir archivos pesados.

Autor

Carlos Fernández Díaz
,cfdiaz7
