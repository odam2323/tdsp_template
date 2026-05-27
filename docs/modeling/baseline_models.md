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

### Resultados de evaluación

Tabla que muestra los resultados de evaluación del modelo baseline, incluyendo las métricas de evaluación.

## Análisis de los resultados

Descripción de los resultados del modelo baseline, incluyendo fortalezas y debilidades del modelo.

## Conclusiones

Conclusiones generales sobre el rendimiento del modelo baseline y posibles áreas de mejora.

## Referencias

Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.

Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.
