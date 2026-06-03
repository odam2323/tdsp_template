
# Reporte del Modelo Final

---

## Resumen Ejecutivo

El presente proyecto desarrolla un sistema de clasificación automática de tumores cerebrales utilizando imágenes de resonancia magnética (MRI), mediante técnicas de Machine Learning y Deep Learning.

El objetivo principal es asistir el análisis de imágenes médicas mediante modelos capaces de identificar patrones asociados a diferentes tipos de tumores cerebrales.

Se implementaron dos enfoques principales:

- Modelo clásico: HOG + SVM  
- Modelo Deep Learning: EfficientNetV2-S (Transfer Learning)

Los resultados muestran que el modelo clásico SVM presenta mejor desempeño en la configuración actual, mientras que EfficientNetV2-S requiere fine-tuning adicional para alcanzar su máximo potencial en el dominio médico.

---

## Descripción del Problema

Los tumores cerebrales presentan alta variabilidad morfológica, lo que dificulta su clasificación automática.

El análisis manual depende de la experiencia del especialista, por lo que se propone un sistema de apoyo basado en IA para mejorar la consistencia del diagnóstico.

---

## Descripción del Dataset

**Brain Tumor MRI Dataset**

Clases:

- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

Distribución:

| Split     | Cantidad |
|----------|----------|
| Training | 5600     |
| Testing  | 1600     |
| Total    | 7200     |

Dataset balanceado.

---

## Preprocesamiento de Datos

- Redimensionamiento a 128x128  
- Normalización (0-1)  
- Conversión a RGB  
- Eliminación de imágenes corruptas  
- Separación train/test  

Para el baseline se utilizó extracción HOG.

---

## Línea Base del Modelo (HOG + SVM)

### Resultados

| Métrica   | Valor |
|----------|------|
| Accuracy | 0.8812 |

### Matriz de confusión

```
[[262  72  58   8]
 [  7 360  13  20]
 [  0   0 400   0]
 [  2  10   0 388]]
```

### Análisis

El modelo SVM presenta un desempeño sólido, especialmente en la clase "No Tumor" y "Pituitary". Las mayores confusiones se presentan entre "Glioma" y "Meningioma".

---

## Modelo Final (EfficientNetV2-S)

Se utilizó EfficientNetV2-S con Transfer Learning congelado en la base.

### Arquitectura

- EfficientNetV2-S (ImageNet)
- Global Average Pooling  
- Batch Normalization  
- Dense (128)  
- Dropout (0.5)  
- Softmax (4 clases)

---

## Entrenamiento

- Optimizer: Adam  
- Learning rate: 0.001 (default)  
- Batch size: 8  
- Epochs: 5  
- Early Stopping aplicado  

---

## Evaluación del Modelo

### Resultados

| Métrica   | Resultado |
|----------|----------|
| Accuracy | 0.6938 |
| Loss     | 0.80 aprox |

---

## Comparación de Modelos

| Métrica   | HOG + SVM | EfficientNetV2-S |
|----------|----------|------------------|
| Accuracy | 0.8812   | 0.6938           |

---

## Análisis Comparativo

El modelo SVM supera al modelo EfficientNet en su configuración actual.

Esto sugiere que:

- El espacio de características es altamente separable  
- El modelo deep learning no ha sido suficientemente afinado  
- Es necesario aplicar fine-tuning para mejorar desempeño  

---

## Inteligencia Artificial Explicable (XAI)

Se implementó Grad-CAM++ para interpretar las predicciones del modelo.

Esto permite visualizar regiones relevantes de las imágenes para la clasificación.

---

## Limitaciones

- EfficientNet sin fine-tuning profundo  
- Dependencia del dataset  
- Alto costo computacional  
- Posible sobreajuste del baseline en features HOG  

---

## Conclusiones

En la configuración actual, el modelo clásico SVM presenta mejor rendimiento que EfficientNetV2-S.

Sin embargo, el modelo Deep Learning tiene mayor potencial de mejora mediante fine-tuning y ajuste de hiperparámetros.

---

## Trabajo Futuro

- Fine-tuning de EfficientNet  
- Uso de Vision Transformers  
- Aumento de datos (data augmentation)  
- Validación clínica  
- Segmentación tumoral  

---

## Referencias

- Tan, M., & Le, Q. EfficientNetV2  
- Selvaraju et al. Grad-CAM  
- Scikit-Learn Documentation  
- TensorFlow Documentation  
- Dataset: Brain Tumor MRI (Kaggle)