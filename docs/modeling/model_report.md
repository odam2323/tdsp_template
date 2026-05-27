# Reporte del Modelo Final

## Resumen Ejecutivo

En esta sección se presentará un resumen de los resultados obtenidos del modelo final. Es importante incluir los resultados de las métricas de evaluación y la interpretación de los mismos.

## Descripción del Problema

El proyecto tiene por meta el desarrollo de diagnostico medico de tumores cerebrales por medio del uso de deep learning sobre imagenes de resonancia magnetica. La elección de un enfoque basado en Deep Learning e Inteligencia Artificial Explicable (XAI) se justifica por la capacidad de estas redes para identificar patrones de textura y gradientes complejos imperceptibles al ojo humano. Además, incorporar interpretabilidad aborda el problema de la "caja negra" de las redes neuronales artificiales, proporcionando un sustento visual (mapas de calor) que el personal médico puede validar antes de emitir un juicio definitivo.

## Descripción del Modelo

El sistema propuesto implementa la arquitectura EfficientNetV2-S preentrenada en ImageNet mediante Transfer Learning, seleccionada estratégicamente por su alta velocidad de convergencia y eficiencia en GPU gracias al uso de bloques Fused-MBConv en sus etapas iniciales, los cuales aceleran la extracción de bordes geométricos y contornos patológicos. En las etapas finales, la red integra módulos Squeeze-and-Excitation que aplican atención por canal para discriminar las texturas complejas del tumor frente al tejido cerebral sano, enviando estos mapas de activación a la capa convolucional de salida (top_conv), la cual alimenta en paralelo al algoritmo Grad-CAM++ para generar los mapas de calor clínicos y a un head de clasificación compuesto por Global Average Pooling, un Dropout de 0.4 contra el sobreajuste y una capa densa con activación Softmax que dictamina la probabilidad diagnóstica final.

## Evaluación del Modelo

En esta sección se presentará una evaluación detallada del modelo final. Se deben incluir las métricas de evaluación que se utilizaron y una interpretación detallada de los resultados.

## Conclusiones y Recomendaciones

En esta sección se presentarán las conclusiones y recomendaciones a partir de los resultados obtenidos. Se deben incluir los puntos fuertes y débiles del modelo, las limitaciones y los posibles escenarios de aplicación.

## Referencias

En esta sección se deben incluir las referencias bibliográficas y fuentes de información utilizadas en el desarrollo del modelo.
