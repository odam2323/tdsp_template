# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

**Sistema de Diagnóstico Asistido para la Clasificación de Tumores Cerebrales**

---

## Objetivo del Proyecto

El presente proyecto tiene como objetivo desarrollar un sistema de diagnóstico asistido basado en técnicas de _Deep Learning_ para la clasificación automática de tumores cerebrales a partir de imágenes médicas.

Para ello, se dispone de una base de datos compuesta por aproximadamente **7.200 imágenes cerebrales** categorizadas en cuatro clases:

- Glioma
- Meningioma
- Pituitary
- Sin tumor

El dataset se encuentra dividido en conjuntos de entrenamiento y prueba con una distribución balanceada entre las clases, lo que permite entrenar y evaluar modelos de manera adecuada.

El propósito principal del proyecto no consiste únicamente en realizar una clasificación automática del tipo de tumor presente en la imagen, sino también en desarrollar un modelo capaz de:

- Explicar la razón de sus predicciones.
- Indicar el nivel de confianza asociado a cada decisión.
- Identificar las regiones relevantes de la imagen que influyen en la clasificación.

De esta manera, se busca construir una herramienta interpretable y confiable que pueda servir como apoyo en procesos de análisis clínico y diagnóstico médico asistido.

---

## Alcance del Proyecto

### Incluye

- Trabajo sobre una base de datos de 7.200 imágenes médicas etiquetadas.
- Procesamiento y análisis exploratorio de imágenes cerebrales.
- Desarrollo de un modelo de _Deep Learning_ para clasificación multiclase.
- Integración de mecanismos de atención (_Attention Modules_) o arquitecturas basadas en _Vision Transformers (ViT)_.
- Aplicación de técnicas de preprocesamiento y aumento de datos (_Data Augmentation_).
- Implementación de métodos de interpretabilidad para explicar las predicciones del modelo.
- Evaluación estadística mediante métricas de desempeño y validación.
- Documentación de resultados, conclusiones y recomendaciones.

### Excluye

- Diagnóstico de enfermedades neurológicas distintas a las contempladas en el dataset.
- Identificación de enfermedades cerebrovasculares, infecciones cerebrales o enfermedades neurodegenerativas.
- Uso clínico oficial sin validación médica especializada.
- Integración con sistemas hospitalarios reales o software médico comercial.
- Validación diagnóstica definitiva por parte de profesionales de la salud.

---

## Entregables

Al finalizar el proyecto se espera contar con:

- Un modelo entrenado y evaluado para clasificación de imágenes cerebrales.
- Un conjunto de métricas de rendimiento sobre prueba independiente.
- Visualizaciones de interpretabilidad como Grad-CAM o mapas de atención.
- Un informe técnico con metodología, resultados y discusión.
- Evidencia del preprocesamiento, entrenamiento y validación del modelo.
- Repositorio organizado con código, documentación y estructura reproducible.

---

## Supuestos y Restricciones

### Supuestos

- Las imágenes del dataset están correctamente etiquetadas.
- Las clases se encuentran balanceadas o suficientemente equilibradas para el entrenamiento.
- El conjunto de datos es representativo para el objetivo del proyecto.
- Se cuenta con acceso a recursos computacionales adecuados para el entrenamiento.

### Restricciones

- El proyecto se desarrolla en un tiempo limitado de 5 semanas.
- El desempeño del modelo depende de la calidad y tamaño del dataset.
- La capacidad de cómputo puede limitar la complejidad del modelo y el número de experimentos.
- El sistema no reemplaza el criterio médico profesional.

---

## Metodología

El proyecto seguirá una metodología basada en técnicas modernas de visión computacional y aprendizaje profundo aplicadas al análisis de imágenes médicas.

### Preprocesamiento de Imágenes

Se contempla aplicar técnicas como:

- Normalización de intensidad mediante **Z-Score**.
- Eliminación de cráneo (_Skull Stripping_), si las imágenes lo requieren.
- Redimensionamiento y estandarización de imágenes.
- Mejora de contraste y reducción de ruido.

### Análisis Exploratorio de Datos (EDA)

Se realizará un análisis exploratorio de las imágenes para identificar:

- Distribución de clases.
- Calidad y resolución de las imágenes.
- Diferencias visuales entre categorías.
- Posibles inconsistencias o _outliers_.

### Data Augmentation

Con el fin de mejorar la capacidad de generalización del modelo, se aplicarán técnicas específicas para imágenes médicas:

- Rotaciones leves.
- Ajustes de brillo y contraste.
- Zoom moderado.
- Desplazamientos controlados.

### Modelamiento

En lugar de utilizar únicamente arquitecturas CNN tradicionales como ResNet o VGG, se plantea implementar mecanismos de atención (_Attention Modules_) o modelos basados en _Vision Transformers (ViT)_.

Este enfoque permitirá que el modelo:

- Ignore regiones de tejido sano.
- Se concentre en anomalías morfológicas relevantes.
- Mejore la interpretabilidad de las predicciones.

Además, se considera utilizar técnicas de _Transfer Learning_ con modelos preentrenados como:

- EfficientNetV2
- Swin Transformer

Posteriormente, se realizará un proceso de _Fine-Tuning_ progresivo para optimizar el rendimiento sobre el dataset específico.

### Interpretabilidad del Modelo

Con el objetivo de explicar las decisiones del sistema, se evaluará la implementación de técnicas como:

- Grad-CAM
- Attention Maps
- Visualización de regiones relevantes

Estas herramientas permitirán identificar qué zonas de la imagen influyen directamente en la predicción final.

### Evaluación del Modelo

La evaluación del desempeño se realizará mediante métricas multicriterio, incluyendo:

- Accuracy
- Precision
- Recall
- F1-Score por clase
- Matriz de Confusión
- Curvas ROC y AUC-ROC

La matriz de confusión será especialmente importante para analizar posibles confusiones entre tipos tumorales similares, particularmente entre Gliomas y Meningiomas.

---

## Cronograma del Proyecto (5 Semanas)

| Semana   | Actividades                                                                    | Fechas             |
| -------- | ------------------------------------------------------------------------------ | ------------------ |
| Semana 1 | Entendimiento del negocio, revisi�n del dataset y organizaci�n de datos        | 11 mayo - 17 mayo  |
| Semana 2 | Preprocesamiento de im�genes y an�lisis exploratorio                           | 18 mayo - 24 mayo  |
| Semana 3 | Desarrollo del modelo, implementaci�n de Attention Modules y Data Augmentation | 25 mayo - 31 mayo  |
| Semana 4 | Entrenamiento, validaci�n, Fine-Tuning y evaluaci�n de m�tricas                | 1 junio - 7 junio  |
| Semana 5 | Interpretabilidad del modelo, documentaci�n, ajustes finales y entrega         | 8 junio - 14 junio |

> **Nota:** El cronograma podrá ajustarse dependiendo del rendimiento del modelo y la complejidad de los experimentos realizados.

---

## Equipo del Proyecto

| Nombre | Rol                                        |
| ------ | ------------------------------------------ |
| Iv�n   | L�der del Proyecto y Desarrollo de Modelos |
| Andr�s | Preprocesamiento de Datos y Evaluaci�n     |
| David  | Implementaci�n, Validaci�n y Documentaci�n |

---

## Presupuesto

El proyecto contempla principalmente recursos académicos y computacionales, incluyendo:

- Uso de GPU o servicios en la nube para entrenamiento.
- Herramientas de desarrollo y librerías de _Machine Learning_.
- Recursos de almacenamiento y procesamiento de datos.
- Elaboración de documentación y presentación final.

Debido al carácter académico del proyecto, gran parte de las herramientas utilizadas serán de código abierto.

### Resumen estimado de costos

| Categoría                  | Descripción                                      | Costo Estimado (USD) |
| -------------------------- | ------------------------------------------------ | -------------------- |
| Personal                   | Diseño, implementación, análisis y documentación | 13,000               |
| Infraestructura y software | GPU, almacenamiento y herramientas               | 1,550                |
| Misceláneos                | Investigación, gestión y contingencias           | 2,555                |
| **TOTAL ESTIMADO**         |                                                  | **17,105**           |

---

## Stakeholders

| Stakeholder               | Relaci�n con el Proyecto       | Expectativas                                 |
| ------------------------- | ------------------------------ | -------------------------------------------- |
| Equipo de desarrollo      | Desarrollo t�cnico del sistema | Obtener un modelo preciso e interpretable    |
| Docente / Tutor acad�mico | Supervisi�n y evaluaci�n       | Cumplimiento de objetivos acad�micos         |
| Comunidad acad�mica       | Inter�s investigativo          | Resultados reproducibles y bien documentados |
| Profesionales m�dicos     | Validaci�n conceptual          | Predicciones comprensibles y confiables      |

---

## Riesgos del Proyecto

- Sobreajuste (_Overfitting_) debido al tamaño limitado del dataset.
- Alto costo computacional durante el entrenamiento.
- Confusión entre clases con características similares.
- Limitaciones en la interpretabilidad del modelo.
- Dependencia de la calidad y diversidad del dataset.
- Tiempos insuficientes para experimentar con varias arquitecturas.

---

## Criterios de Éxito

El proyecto será considerado exitoso si:

- El modelo alcanza métricas competitivas de clasificación.
- Las predicciones presentan alta confiabilidad estadística.
- Se generan explicaciones visuales comprensibles.
- El sistema demuestra capacidad de generalización sobre datos de prueba.
- Se logra una adecuada interpretación de las regiones relevantes para la clasificación.
- El resultado final puede documentarse y reproducirse de forma ordenada.

---

## Aprobaciones

<<<<<<< Updated upstream
<<<<<<< HEAD
| Aprobador | Cargo | Firma | Fecha |
| ---------------------- | ------- | ---------------- | ------------------ |
| [Nombre del aprobador] | [Cargo] | \***\*\_\_\*\*** | **_ / _** / **\_** |
=======
| Aprobador | Cargo | Firma | Fecha |
| --------- | ----- | ----- | ----- |
| [Nombre del aprobador] | [Cargo] | ****\_\_**** | ** / ** / \_\_\_\_ |

> > > > > > > # fa4ca6ece103d980581d683b2c1e8a8aa7b0902b
> > > > > > >
> > > > > > > | Aprobador              | Cargo   | Firma            | Fecha              |
> > > > > > > | ---------------------- | ------- | ---------------- | ------------------ |
> > > > > > > | [Nombre del aprobador] | [Cargo] | \***\*\_\_\*\*** | **_ / _** / **\_** |
> > > > > > >
> > > > > > > Stashed changes
