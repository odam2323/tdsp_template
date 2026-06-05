# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** Sistema de Diagnóstico Asistido para la Clasificación de Tumores Cerebrales\*
- **Plataforma de despliegue:** Contenedor Docker ejecutando un servidor local con Streamlit.
- **Requisitos técnicos:** los requisitos técnicos son los siguientes:

        -Motor de Docker y Docker Compose instalados y en ejecución.

        -Python 3.11 (definido en la imagen base python:3.11-slim).

        -Bibliotecas de terceros principales (definidas en requirements.txt):

            -Machine Learning / Deep Learning: tensorflow==2.21.0, keras==3.12.2, scikit-learn==1.7.2

            -Procesamiento de Datos y Matemáticas: pandas==2.3.3, numpy==2.2.6, scipy==1.15.3

            -Visión por Computadora e Imágenes: opencv-python==4.13.0.92, scikit-image==0.25.2, pillow==12.2.0

            -Interfaz y APIs: streamlit, fastapi==0.136.3

            -Orquestación y Versionado: dvc (y sus extensiones), celery==5.6.3, kaggle==1.7.4.5

            -Hardware: Capacidad suficiente en CPU/RAM para cargar modelos de TensorFlow y procesar los conjuntos de datos en memoria (las matrices X_train, X_test generadas en el preprocesamiento).

- **Requisitos de seguridad:** El archivo de credenciales kaggle.json no debe subirse al repositorio público; debe inyectarse directamente en la máquina o servidor local mediante volúmenes de Docker.

- **Diagrama de arquitectura:** (imagen que muestra la arquitectura del sistema que se utilizará para desplegar el modelo)

## Código de despliegue

- **Archivo principal:** app_streamlit.py (Ejecutado por el comando CMD en el Dockerfile para levantar la interfaz). Adicionalmente, el pipeline de preparación de datos se orquesta en main.py en la raíz del proyecto.
- **Rutas de acceso a los archivos:** ./docs:/app/docs (Donde se guardan los datos originales y procesados como dataset_train.pkl y dataset_test.pkl).
- **Variables de entorno:** el proyecto se despliega en el puerto 8501, no es necesario un archivo .env

## Documentación del despliegue

- **Instrucciones de instalación:**

        -Clonar el repositorio del proyecto en la máquina local.

        -Asegurarse de tener el "Docker daemon" en ejecución (por ejemplo, iniciando Docker Desktop).

        -Construir la imagen del contenedor ejecutando en la terminal (desde la raíz del proyecto): docker compose build.

- **Instrucciones de configuración:**

        -Descargar el token de API de Kaggle (kaggle.json) desde la configuración de la cuenta de Kaggle y ubicarlo en el directorio raíz del proyecto para que el volumen de Docker pueda mapearlo correctamente.

        -Ejecutar el script main.py de forma local o dentro del contenedor si es necesario descargar los datos (pull_dataset) y generar los archivos .pkl de entrenamiento y prueba antes de levantar la interfaz. Así mismo, se entrenarán los modelos, ya que el .gitignore los excluye por el peso de los archivos

        -Colocar el modelo pre-entrenado dentro del directorio ./models (si no se ubicó automáticamente).

- **Instrucciones de uso:**

        -docker compose up (o docker compose up -d para correrlo en segundo plano).

        -Abrir un navegador web e ingresar a http://localhost:8501.

        -Interactuar con la interfaz gráfica de Streamlit para cargar imágenes y visualizar las predicciones del modelo.

- **Instrucciones de mantenimiento:** Para detener la aplicación, presionar Ctrl + C en la terminal donde se está ejecutando (o ejecutar docker compose down si está en segundo plano). Si se actualizan las dependencias, modificar el archivo requirements.txt y volver a construir la imagen ejecutando docker compose up --build. Para actualizar el modelo de predicción, simplemente reemplazar el archivo correspondiente dentro de la carpeta local ./models (el cambio se reflejará automáticamente gracias a los volúmenes de Docker sin necesidad de reconstruir la imagen).
