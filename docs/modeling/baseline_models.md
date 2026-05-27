# Reporte del Modelo Baseline

Este documento contiene los resultados del modelo baseline.

## Descripción del modelo

El modelo baseline consiste en un clasificador tradicional Support Vector Machine (SVM) con kernel de Función de Base Radial (RBF), seleccionado por su alta efectividad para resolver problemas de clasificación multiclase en espacios de alta dimensionalidad. Este modelo no aprende características visuales de forma autónoma, sino que utiliza vectores de características extraídos manualmente mediante el algoritmo HOG (Histogram of Oriented Gradients), estableciendo así una línea base clásica de aprendizaje supervisado para comparar el rendimiento frente a las arquitecturas de aprendizaje profundo posteriores.

## Variables de entrada

La variable de entrada consiste en un vector unidimensional de características estructurales estáticas, generado tras aplicar el descriptor HOG sobre las imágenes de resonancia magnética previamente normalizadas y redimensionadas. Este vector condensa la distribución de las orientaciones de los gradientes de intensidad lumínica dentro de celdas locales de la imagen, capturando de forma explícita los contornos, formas geométricas y discontinuidades anatómicas del cerebro sin arrastrar la carga computacional de los píxeles en bruto.

## Variable objetivo

La variable objetivo es de tipo categórica nominal y corresponde al tipo de diagnóstico o hallazgo patológico presente en la resonancia magnética, codificada mediante One-Hot Encoding en cuatro clases mutuamente excluyentes: Glioma, Meningioma, Pituitary (tumor de pituitaria) y Sin tumor.

## Evaluación del modelo

### Métricas de evaluación

Para medir el desempeño global y balanceado del clasificador, se utilizaron las métricas de Accuracy (Exactitud) para cuantificar el porcentaje total de aciertos sobre el dataset, junto con Precision (Precisión) y Recall (Sensibilidad) calculadas por clase para evaluar la capacidad del modelo de mitigar falsos positivos y falsos negativos, respectivamente. Adicionalmente, se analiza el F1-Score como la media armónica que pondera equitativamente la precisión y la sensibilidad, permitiendo identificar con rigor científico qué afecciones tumorales presentan mayor dificultad de discriminación estructural.

### Resultados de evaluación

Los resultados de la evaluación fueron los siguientes:

[MATRIZ DE CONFUSIÓN]
[[263  71  58   8]
 [  6 361  13  20]
 [  0   0 400   0]
 [  2  11   0 387]]

[REPORTE DE CLASIFICACIÓN]
precision recall f1-score support

      glioma       0.97      0.66      0.78       400

meningioma 0.81 0.90 0.86 400
notumor 0.85 1.00 0.92 400
pituitary 0.93 0.97 0.95 400

    accuracy                           0.88      1600

macro avg 0.89 0.88 0.88 1600
weighted avg 0.89 0.88 0.88 1600

## Análisis de los resultados

Los resultados demuestran que el baseline alcanza un rendimiento global competitivo del 88%, consolidando como su mayor fortaleza la identificación de pacientes sanos y tumores de pituitaria debido a sus marcadas diferencias geométricas y de contraste en los gradientes. Sin embargo, la debilidad crítica del modelo radica en su incapacidad para discriminar los bordes difusos y las texturas internas que separan a los Gliomas de los Meningiomas. Al depender exclusivamente de características estadísticas estáticas hechas a mano (HOG), el clasificador lineal SVM se ve superado por la naturaleza infiltrativa del Glioma, el cual simula la densidad de un tejido sano o adopta fronteras morfológicas ambiguas que confunden los histogramas de gradientes.

## Conclusiones

El modelo baseline cumple con creces el objetivo de establecer un umbral mínimo de comparación, demostrando que los patrones de forma básicos resuelven gran parte del problema pero fallan en los escenarios médicos de alta complejidad. Para mitigar la debilidad diagnóstica observada en la clase Glioma, queda plenamente justificado avanzar hacia el modelo final basado en EfficientNetV2-S, cuya arquitectura convolucional profunda y dinámica aprenderá jerarquías de texturas abstractas y sutiles imposibles de capturar con HOG. El uso de capas de atención en el modelo final será mandatorio para forzar a la red a concentrarse en las microestructuras limítrofes entre Gliomas y Meningiomas, elevando la sensibilidad clínica general.

## Referencias

Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.

Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.
