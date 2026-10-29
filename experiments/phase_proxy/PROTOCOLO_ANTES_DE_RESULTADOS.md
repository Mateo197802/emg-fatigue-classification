# Prueba exploratoria autorizada: fase candidata y clasificación temporal

Registrado antes de extraer fases o evaluar nuevos modelos. El usuario autorizó intentar la propuesta y comprobar rendimiento. Es un spike independiente; no modifica el modelo, las etiquetas ni la entrega principal.

## Hipótesis operacional y límite

Añadir una representación aprendida de fase local de actividad a las diez características EMG puede mejorar la clasificación temprano/tardío fuera del grupo de entrenamiento. No hay etiquetas verificadas de extensión. Se prueba una aproximación EMG de fase de ráfaga; no se prueba un detector de extensión ni se identifica fatiga fisiológica.

Reutilizar sin modificar: 140 CSV/138 formales, calidad/filtrado/actividad del contrato study.yaml, diez características, etiquetas débiles, 5.509 ventanas evaluables, 15 grupos externos y cuatro particiones internas. Verificar checksum del cache y concordancia exacta del baseline con la evaluación guardada.

## Primera etapa

Sobre la máscara de actividad ya existente, identificar ráfagas continuas en cada tramo válido. Retener para supervisión auxiliar solamente ráfagas de 0,5–8 s: límite exploratorio fijo para evitar tratar un tramo largo como una repetición. No se ajusta con resultados. Cada ventana completa dentro de la primera mitad recibe fase 0; dentro de la segunda, fase 1. Ventanas que cruzan el centro o extremos quedan sin etiqueta auxiliar. Se puede usar el tercio central del registro para supervisión de fase, pero solamente de grupos de entrenamiento. Nunca usar tiempo, ID, duración o posición de la ráfaga como entrada aprendida.

Cabeza auxiliar fija: random forest, 100 árboles, max_depth=3, min_samples_leaf=5, balanced, semilla 20261028. Entradas: las diez características originales. Es una transformación no lineal de la información disponible, no una nueva medición. Predice un score no calibrado de segunda mitad para todas las ventanas. Si no hay soporte de ambas clases en una partición de entrenamiento, registrar fallo/inconclusión, no fabricar una clase.

## Comparaciones fijadas

- baseline_10: logística sobre diez características, C=0,1/1/10.
- phase_score: logística sobre diez características + score auxiliar.
- phase_conditioned (principal): diez características + score + score×RMS/MDF/MNF.
- phase_only (diagnóstico): solamente score auxiliar.
- random_conditioned (control de capacidad): misma arquitectura que phase_conditioned, pero cabeza auxiliar entrenada con etiquetas binarias deterministas SHA256(window_id, seed), independientes del objetivo. Mismos ejemplos auxiliares. Un solo control aleatorio; no constituye prueba exhaustiva de permutación.

Todas las logísticas usan StandardScaler ajustado en entrenamiento, balanced y C seleccionado por media macro-F1 por grupo en cuatro particiones internas. No se elige un ganador por las puntuaciones externas.

Para entrenar la segunda etapa, generar scores auxiliares fuera del grupo mediante cross-fitting. Repetirlo dentro de cada partición interna usada para elegir C: ni el escalador ni la cabeza auxiliar ni la segunda etapa ven grupos de validación. Para cada prueba externa, entrenar la cabeza final sólo con los otros 14 grupos. Conservar splits y comprobaciones de disjunción en todos los niveles.

## Evaluación y evidencia

Mismas 5.509 ventanas para todos los modelos; sin exclusión por calidad de la fase candidata. Medir macro-F1, AUC, demás métricas binarias y OOF; calidad auxiliar sólo frente a las mitades EMG, nunca frente a extensión real. Contraste principal: phase_conditioned menos baseline_10, diferencias pareadas entre 15 grupos e IC bootstrap 95%, 5.000 remuestreos, semilla fija. Informar número de grupos que mejoran y diferencias individuales. Otras comparaciones son exploratorias; no inferencia confirmatoria después de haber visto los resultados principales del proyecto.

Un aumento puntual con intervalo que cruza cero no acredita una mejora clara. Una ganancia similar al control aleatorio no acredita el aporte específico de fase. Una mejora clara frente al baseline y control seguiría limitada a esta aproximación y cohorte; harían falta etiquetas cinemáticas independientes, referencia de fatiga y nuevas sesiones para confirmar el mecanismo y el cambio de dominio.

Guardar código, configuración, fuentes, hashes, tablas, predicciones, auditoría de fase, bibliografía y recibo de completitud en este suplemento privado. Mantener intactos repositorio y ZIP principal.
