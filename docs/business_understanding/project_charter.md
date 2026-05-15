# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

**Sistema de Diagnóstico Asistido para la Clasificación de Tumores Cerebrales**

---

## Objetivo del Proyecto

El proyecto tiene como objetivo desarrollar un sistema de diagnóstico asistido basado en técnicas de *Deep Learning* para la clasificación automática de tumores cerebrales a partir de imágenes médicas.

La base de datos disponible contiene aproximadamente **7.200 imágenes cerebrales** distribuidas en cuatro categorías:

- Glioma
- Mingioma
- Pituitary
- Sin tumor

El dataset se encuentra organizado en conjuntos de entrenamiento y prueba con una distribución balanceada entre clases.

El propósito del sistema no se limita únicamente a realizar una clasificación de imágenes, sino también a proporcionar:

- Una explicación de la decisión tomada por el modelo.
- Un nivel de confianza o probabilidad asociado a cada predicción.
- Un enfoque interpretable y confiable que pueda servir como apoyo al análisis clínico.

En este sentido, se busca construir un modelo capaz de identificar patrones tumorales relevantes y justificar de manera comprensible el resultado obtenido.

---

## Alcance del Proyecto

### Incluye

- Procesamiento y análisis de 7.200 imágenes clínicas etiquetadas.
- Desarrollo de un modelo de *Deep Learning* para clasificación multiclase de tumores cerebrales.
- Implementación de mecanismos de interpretabilidad para explicar las decisiones del modelo.
- Generación de probabilidades de confianza asociadas a cada predicción.
- Aplicación de técnicas de preprocesamiento y aumento de datos específicas para imágenes médicas.
- Evaluación estadística y validación del desempeño mediante métricas especializadas.

### Excluye

- Diagnóstico de enfermedades neurológicas distintas a las contempladas en el dataset.
- Detección de enfermedades cerebrovasculares, infecciones cerebrales o trastornos neurodegenerativos.
- Uso clínico directo sin validación médica especializada.
- Integración con sistemas hospitalarios o plataformas médicas reales.

---

## Metodología

El desarrollo del proyecto seguirá una metodología basada en modelos avanzados de visión computacional aplicados a imágenes médicas.

### Preprocesamiento de Datos

Se contempla aplicar técnicas como:

- Normalización de intensidad mediante **Z-Score**.
- Eliminación de cráneo (*Skull Stripping*), en caso de ser necesario.
- Redimensionamiento y estandarización de imágenes.
- Eliminación de ruido y mejora de contraste.

### Aumento de Datos (*Data Augmentation*)

Se emplearán transformaciones adecuadas para imágenes médicas, tales como:

- Rotaciones leves.
- Ajustes de contraste y brillo.
- Zoom moderado.
- Desplazamientos controlados.

Estas técnicas permitirán mejorar la capacidad de generalización del modelo.

### Modelamiento

En lugar de utilizar únicamente arquitecturas CNN tradicionales como ResNet o VGG, se plantea integrar mecanismos de atención (*Attention Modules*) o utilizar arquitecturas basadas en *Vision Transformers (ViT)*.

El objetivo de este enfoque es que el modelo aprenda a:

- Ignorar regiones de tejido sano.
- Concentrarse en anomalías morfológicas relevantes.
- Mejorar la interpretabilidad de las predicciones.

Adicionalmente, se considera el uso de:

- *Transfer Learning* con modelos preentrenados como:
  - EfficientNetV2
  Swin Transformer
- Estrategias de *Fine-Tuning* progresivo.

### Interpretabilidad del Modelo

Para explicar las decisiones del sistema se evaluará el uso de técnicas como:

- Grad-CAM
- Attention Maps
- Visualización de regiones relevantes

Esto permitirá identificar qué zonas de la imagen influyen en la predicción final.

### Evaluación del Modelo

El desempeño será medido mediante métricas multicriterio, incluyendo:

- Accuracy
- Precision
- Recall
- F1-Score por clase
- Matriz de confusión
- Curvas ROC y AUC-ROC

La matriz de confusión será especialmente importante para analizar posibles confusiones entre tipos tumorales similares, como Gliomas y Meningiomas.

---

## Cronograma Tentativo (5 Semanas)

| Semana | Actividades |
|---|---|
| Semana 1 | Entendimiento del negocio, revisión del dataset y carga de datos |
| Semana 2 | Preprocesamiento de imágenes, análisis exploratorio y normalización |
| Semana 3 | Implementación del modelo base con mecanismos de atención y Data Augmentation |
| Semana 4 | Entrenamiento, fine-tuning, validación y evaluación de métricas |
| Semana 5 | Interpretabilidad del modelo, documentación, ajustes finales y entrega |

> **Fecha de inicio estimada:** 13 de mayo de 2026  
> **Fecha de finalización estimada:** 17 de junio de 2026

---

## Equipo del Proyecto

| Nombre | Cargo | Responsabilidades |
|---|---|---|
| [Andrés] | Líder del Proyecto | Coordinación general y supervisión |
| [David] | Científico de Datos | Desarrollo y entrenamiento del modelo |
| [Ivan] | Ingeniero de ML | Implementación y optimización |

---

## Presupuesto

El presupuesto del proyecto contempla:

- Recursos computacionales (GPU o servicios en la nube).
- Almacenamiento y procesamiento de datos.
- Herramientas de desarrollo y experimentación.
- Costos asociados a investigación y validación.
- Elaboración de documentación y presentación final.

> El detalle financiero será definido según la infraestructura y herramientas seleccionadas.

---

## Stakeholders

| Stakeholder | Relación con el Proyecto | Expectativas |
|---|---|---|
| Equipo de investigación | Desarrollo técnico | Alto desempeño del modelo |
| Institución académica | Supervisión y evaluación | Cumplimiento de objetivos |
| Profesionales médicos | Validación clínica | Interpretabilidad y precisión |
| Usuarios finales | Uso potencial del sistema | Resultados confiables y comprensibles |

---

## Riesgos del Proyecto

- Sobreajuste (*Overfitting*) debido al tamaño limitado del dataset.
- Interpretabilidad insuficiente del modelo.
- Alto costo computacional durante el entrenamiento.
- Posibles sesgos en las imágenes del dataset.
- Confusión entre clases tumorales visualmente similares.

---

## Criterios de Éxito

El proyecto será considerado exitoso si:

- El modelo alcanza métricas competitivas de clasificación.
- Las predicciones presentan alta confiabilidad estadística.
- Se generan explicaciones visuales comprensibles.
- El sistema demuestra capacidad de generalización sobre datos de prueba.

---

## Aprobaciones

| Aprobador | Cargo | Firma | Fecha |
|---|---|---|---|
| [Nombre] | [Cargo] | __________ | ___ / ___ / _____ |