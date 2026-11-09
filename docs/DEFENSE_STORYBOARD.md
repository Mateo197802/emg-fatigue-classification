# Defensa del análisis sEMG

La narración principal dura 600 segundos en 16 diapositivas. Catorce diapositivas de anexo conservan matrices, fórmulas, diagramas y evidencia de señal para preguntas. El objetivo del experimento ejecutado se nombra como clasificación temporal proxy temprano/tardío. Accuracy y F1 se informan para esa etiqueta proxy; no hay métricas medidas frente a fatiga/no fatiga fisiológica.

## Recorrido principal

1. Objetivo, señal y autores: Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez.
2. Corpus y denominadores cuantitativos.
3. Señal almacenada, señal filtrada y control de reloj/adquisición.
4. Preanálisis cuantitativo de calidad de señal; trayectorias por repetición/fase no disponibles.
5. Preprocesamiento y criterios de calidad.
6. Construcción de ventanas y etiquetas temporales proxy.
7. Diez características por ventana.
8. Validación anidada por grupos.
9. Familias de modelos y selección interna.
10. Accuracy, macro-F1 y AUC de los modelos sobre las etiquetas proxy; se incluyen diferencias pareadas y sus intervalos por grupo.
11. Matriz de confusión y variabilidad entre grupos.
12. Arquitectura auxiliar ejecutada: mitades de ráfaga EMG, no extensión/flexión.
13. Variantes y control aleatorio del experimento auxiliar.
14. Rendimiento auxiliar sobre el mismo objetivo proxy.
15. Resultados separados de cada etapa y sus matrices.
16. Conclusiones, límites de las métricas y requisitos para medir fatiga real.

## Anexos de evidencia

17. Relación entre todos los procedimientos, modelos y ajustes.
18. Sensibilidades y ablaciones.
19. Conteos OOF de los modelos principales.
20. Matrices principales normalizadas por fila.
21. Conteos OOF de las variantes auxiliares.
22. Matrices auxiliares normalizadas por fila.
23. Bibliografía sobre movimiento y EMG.
24. Ventanas históricas de 250 ms frente a la propuesta de 500 ms y su análisis de sensibilidad.
25. Auditoría exploratoria de cinco características candidatas y sus fórmulas; no se entrenó con ellas.
26. Fórmulas del filtro Butterworth, SOS, notch, ventanas y periodograma Hann.
27. Fórmulas, unidades y razones de las características; distinción entre WL v1 y v2.
28. Arquitectura propuesta: etiqueta cinemática, etiqueta mecánica, baseline directo, cascada y expertos suaves.
29. Ecuaciones de etiquetas y de las arquitecturas; límites de fine-tuning y RL.
30. Señal real, PSD antes/después y respuesta del filtro a la frecuencia efectiva del registro. Esta figura queda solo en el paquete privado.

El artículo, los anexos y los guiones usan la misma terminología y los mismos resultados congelados. La figura de señal se deriva de una grabación continua elegida por QC sin revisar la clase ni el resultado del modelo. El repositorio público contiene el generador, no la señal ni la imagen derivada.
