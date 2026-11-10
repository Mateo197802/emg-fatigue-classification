# Defensa del análisis sEMG

La narración principal dura 600 segundos en 19 diapositivas; 14 diapositivas de reserva mantienen matrices, fórmulas y evidencia para preguntas. La secuencia separa tres objetivos: proxy temporal del ZIP del curso, fatiga percibida por ventana con autoreporte sincronizado y resultado por ensayo de un protocolo externo.

## Recorrido principal

1. Objetivo del proyecto, señal y autores: Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez.
2. Corpus del curso, denominadores y grupos de evaluación.
3. Señal almacenada, voltaje exportado y diagnóstico del reloj.
4. Preanálisis cuantitativo de calidad y referencias ausentes.
5. Preprocesamiento histórico y criterios de señal.
6. Ventanas de 250 ms y proxy temprano/tardío del curso.
7. Diez características históricas y unidades.
8. Validación por grupos, sin separar ventanas del mismo grupo.
9. Modelos principales y selección interna.
10. Accuracy, macro-F1 y AUC del proxy histórico.
11. Errores OOF y matriz de confusión del proxy.
12. Arquitectura auxiliar ejecutada: mitad inicial/final de ráfaga EMG.
13. Variantes, controles y relación entre las etapas auxiliares.
14. Rendimiento auxiliar frente al modelo directo; no hay mejora concluyente.
15. Matrices y objetivos distintos de cada etapa.
16. Modelo binario realmente entrenado contra autoreporte estable, por ventana.
17. Rendimiento de HGB global, búsqueda anidada, expertos por músculo y control temporal, con matrices OOF.
18. Benchmark externo de resultado por ensayo; clases, métricas y atajo temporal del protocolo.
19. Conclusiones con los tres alcances separados y límites de la evidencia.

## Anexos de evidencia

20. Relación entre procedimientos, modelos y ajustes.
21. Sensibilidades y ablaciones.
22. Conteos OOF de los modelos históricos principales.
23. Matrices históricas normalizadas por fila.
24. Conteos OOF de las variantes auxiliares.
25. Matrices auxiliares normalizadas por fila.
26. Bibliografía de fase y EMG.
27. Ventanas de 250 ms frente a 500 ms: fórmula y razón de diseño.
28. Auditoría de características candidatas y fórmulas.
29. Filtros Butterworth, SOS, notch, Hann y fórmulas de ventana.
30. Características, unidades, razones y limitaciones.
31. Arquitectura propuesta con etiquetas independientes de cinemática y fuerza.
32. Ecuaciones de las etiquetas y arquitecturas; límites de fine-tuning y RL.
33. Señal real, espectros antes/después y respuesta del filtro, solo en el paquete privado.

La presentación muestra primero el análisis del curso para explicar la corrección de objetivo, y luego evidencia el modelo por ventana que sí usa una referencia temporal externa. El modelo de fatiga percibida no se presenta como detector de pérdida mecánica. La cascada extensión/flexión→fatiga sigue sin poder probarse con estos datos porque faltan etiquetas cinemáticas sincronizadas.
