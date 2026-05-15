# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

[Sistema de Diagnóstico Asistido para la Clasificación de Tumores Cerebrales]

## Objetivo del Proyecto

[Para el presente proyecto disponemos de una base de datos con 7.200 imagenes de radiografías cerebrales categorizadas en cuatro clases: Glioma, Meningioma, Pituitary, y sin tumor. Además, la base de datos está estructurada en conjuntos de prueba y entrenamiento con una distribución balanceada de clases. Teniendo esto en cuenta, el objetivo con esta base de datos no es solamente el de realizar una clasificación simple (diciendo qué tumor es), sino que el modelo explique la razón de la decisición y qué tan segura está de ella. Por lo tanto, en otras palabras, con este proyecto planteamos el objetivo de ofrecer un modelo que identifique los tipos de tumores presentes explicando la razón de tal decisión y la certeza que se tenga de esta decisión.]

## Alcance del Proyecto

### Incluye:

- [Trabajo sobre 7.200 imagenes clínicas de tumores cerebrales etiquetadas y divididaes entre entrenamiento y prueba]
- [Elaboración de un modelo de deep learning que identifique correctamente las imagenes con sus etiquetas correspondientes mientras es capaz de dar las razones de dicha identificación y la probabilidad de su acierto]
- [Un modelo con estimaciones certeras y estadisticamente significativas ]

### Excluye:

- [El sistema no está diseñado para identificar enfermedades cerebrovasculares, infecciones (como abscesos) o enfermedades neurodegenerativas que no presenten masas tumorales dentro de las 4 categorías del dataset.] 

## Metodología

[En primer lugar, se espera no utilizar una CNN estándar, como ResNet o VGG, sino integrar attention modules (o usar un vision transformer - ViT). Esto permite que el modelo "aprenda" a ignorar el tejido sano y se enfoque específicamente en las anomalías morfológicas de la masa tumoral. En este caso, se considera utilizar normalización de intensidad (Z-score) y eliminación de cráneo (Skull Stripping) en el preprocesamiento de las imagenes, si las imagenes lo requiere. Además, se utilizará Data Augmentation específico para medicina (rotaciones leves, cambios de contraste). Se considera también la posibilidad de usar Transfer Learning (ej. EfficientNetV2 o Swin Transformer) pre-entrenado en ImageNet y realizar un fine-tuning progresivo, si nos decantamos por este tipo de modelos, pero lo cierto es la integración de attention modules. Finalmente, se realizará una evaluacieon multicriterio mostrando metricas de Accuracy, como F1-Score por cada clase, Matriz de Confusión (crucial para ver si se confunden Gliomas con Meningiomas) y curvas AUC-ROC.]

## Cronograma

| Etapa                                        | Duración Estimada | Fechas                          |
| -------------------------------------------- | ----------------- | ------------------------------- |
| Entendimiento del negocio y carga de datos   | 2 semanas         | del 1 de mayo al 15 de mayo     |
| Preprocesamiento, análisis exploratorio      | 4 semanas         | del 16 de mayo al 15 de junio   |
| Modelamiento y extracción de características | 4 semanas         | del 16 de junio al 15 de julio  |
| Despliegue                                   | 2 semanas         | del 16 de julio al 31 de julio  |
| Evaluación y entrega final                   | 3 semanas         | del 1 de agosto al 21 de agosto |

Hay que tener en cuenta que estas fechas son de ejemplo, estas deben ajustarse de acuerdo al proyecto.

## Equipo del Proyecto

- [Nombre y cargo del líder del proyecto]
- [Nombre y cargo de los miembros del equipo]

## Presupuesto

[Descripción del presupuesto asignado al proyecto]

## Stakeholders

- [Nombre y cargo de los stakeholders del proyecto]
- [Descripción de la relación con los stakeholders]
- [Expectativas de los stakeholders]

## Aprobaciones

- [Nombre y cargo del aprobador del proyecto]
- [Firma del aprobador]
- [Fecha de aprobación]
