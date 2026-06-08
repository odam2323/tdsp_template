Brain Tumor MRI Classification
Descripción

Proyecto de clasificación automática de tumores cerebrales a partir de imágenes MRI utilizando técnicas de Machine Learning y Deep Learning.

El sistema implementa dos enfoques de clasificación:

SVM (Support Vector Machine)
EfficientNet (Red Neuronal Convolucional)

El objetivo es comparar ambos modelos para identificar cuatro categorías clínicas:

Glioma
Meningioma
Pituitary Tumor
No Tumor
Dataset

Se utiliza el dataset:

Brain Tumor MRI Dataset

Fuente:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Distribución:

Conjunto	Imágenes
Training	5600
Testing	1600
Total	7200

Clases:

glioma
meningioma
notumor
pituitary
Estructura del proyecto
tdsp_template/
│
├── docs/
│   ├── data/
│   ├── business_understanding/
│   ├── modeling/
│   └── deployment/
│
├── scripts/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── eda/
│   ├── training/
│   └── evaluation/
│
├── src/
│   └── brain_tumor_classifier/
│
├── models/
│
├── app_streamlit.py
├── main.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
Pipeline

El pipeline completo ejecuta:

Descarga del dataset desde Kaggle.
Extracción y organización de imágenes.
Preprocesamiento.
Generación de datasets serializados (.pkl).
Análisis exploratorio (EDA).
Entrenamiento de EfficientNet.
Entrenamiento de SVM.
Evaluación de EfficientNet.
Evaluación de SVM.
Ejecución local
Crear entorno virtual
python -m venv venv
Activar entorno

Windows:

venv\Scripts\activate

Linux:

source venv/bin/activate
Instalar dependencias
pip install -r requirements.txt
Ejecutar pipeline completo
python main.py
Aplicación Streamlit

Para ejecutar la interfaz web:

streamlit run app_streamlit.py

La aplicación permite:

Cargar imágenes MRI.
Seleccionar imágenes del conjunto de prueba.
Comparar predicciones SVM vs EfficientNet.
Visualizar resultados y confianza.
Docker

Construir imagen:

docker compose build

Levantar contenedor:

docker compose up
Resultados preliminares
SVM

Accuracy obtenida:

88.12%

Matriz de confusión y métricas disponibles en:

scripts/evaluation/svm.py
EfficientNet

Modelo basado en transfer learning utilizando EfficientNetV2S.

Tecnologías utilizadas
Python
TensorFlow / Keras
Scikit-Learn
OpenCV
NumPy
Pandas
Streamlit
Docker
DVC
Autores

Proyecto desarrollado en el marco del Diplomado en Machine Learning y Data Science.

Universidad Nacional de Colombia.