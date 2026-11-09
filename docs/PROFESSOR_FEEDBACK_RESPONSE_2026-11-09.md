# Respuesta técnica a los comentarios del docente

**Revisión:** 9 de noviembre de 2026

**Estado del objetivo fisiológico:** no entrenado; la auditoría del ZIP no encontró referencias independientes de fatiga ni de fase de movimiento.

## 1. Qué predice el modelo ejecutado

La corrección principal es de objetivo y alcance: el clasificador actual distingue ventanas elegibles del primer tercio temporal de ventanas del último tercio de cada grabación. Son etiquetas proxy `temprana/tardía`; el modelo no aprendió `fatiga/no fatiga` fisiológica. La primera etapa del experimento auxiliar predice mitades de ráfagas de actividad EMG, tampoco extensión/flexión de rodilla.

La auditoría local encontró EMG en CSV, además de hojas de bibliografía y de filtros/QC. No encontró torque o fuerza calibrados y sincronizados, cinemática sincronizada, eventos mecánicos de repetición, mediciones MVC/Borg ni una clave independiente de participantes/sesiones. En consecuencia, no se entrenó una etiqueta fisiológica inventada ni se reportan Accuracy o F1 de fatiga verdadera.

## 2. Corpus, preanálisis y procesamiento ejecutado

| Elemento | Evidencia disponible | Alcance |
|---|---|---|
| Archivo principal | 140 CSV; 138 registros R1–R5 y dos controles R0 separados | 15 grupos derivados de carpetas; identidad de persona no verificada independientemente |
| Señal | 4.256.257 muestras y tasa efectiva cercana a 1 kHz | El canal se conserva en cuentas ADC; `Voltage_mV` no demuestra calibración física |
| Preanálisis | Reloj, huecos, jitter, valores extremos, tramos constantes, ocupación candidata, amplitud ADC y prominencia de línea 50/60 Hz | No están disponibles trayectorias por repetición/fase; se retiraron las curvas por deciles temporales porque se superponen con el proxy y no evidencian fatiga |
| Filtro principal | Centrado por tramo continuo; Butterworth pasa banda 20–400 Hz en SOS, diseño recalculado con `fs`, `sosfiltfilt` fuera de línea y máscara de 100 ms por borde | No filtra a través de huecos; notch 50/60 Hz queda como sensibilidad condicionada por prominencia medida |
| Ventana histórica | Deslizante, 250 ms, 50% solapamiento: `N=round(0,25 fs)`, `H=round(N(1−ρ))`; a 1 kHz, `N=250`, `H=125`, `Δf=4 Hz` | Es el protocolo de los resultados congelados |
| Ventana propuesta con referencias | 500 ms principal y 250 ms sensibilidad; 50% de solapamiento | A 1 kHz, `N=500/H=250/Δf=2 Hz` y `N=250/H=125/Δf=4 Hz`; no se evaluó porque faltan etiquetas independientes |
| PSD y features | Periodograma unilateral de densidad con ventana Hann; RMS, MAV, longitud de onda/s, MDF, MNF, razón de potencia y cuatro energías relativas db4 | Las fórmulas, bandas, unidades y condiciones están en [protocolo y ecuaciones](reference_fatigue_protocol.md) y en el informe privado |

Se auditó por separado un conjunto de características candidatas (cruces por cero, cambios de signo de pendiente, iEMG, varianza y entropía espectral) en 9.289 ventanas QC válidas, sin leer etiquetas ni entrenar modelos. La varianza fue monótonamente redundante con RMS y iEMG con MAV bajo la ventana fija; la entropía y los contadores mostraron correlaciones descriptivas con MNF/MDF y wavelets. Ninguna característica candidata se agregó al modelo ni se atribuye una mejora en Accuracy/F1. Los cálculos, unidades, valores y límites están en la [auditoría de características candidatas](CANDIDATE_FEATURE_AUDIT_2026-11-09.md); los valores por ventana siguen en almacenamiento privado.

El solapamiento aumenta la densidad temporal de observaciones, pero hace que ventanas vecinas dependan entre sí. La incertidumbre principal se calcula por grupo y los grupos completos se retienen en los pliegues externos.

## 3. Modelos y rendimiento medido

La evaluación ejecutada compara mayoría, logística RMS+MDF, logística con todas las diez features, SVM-RBF, Random Forest y un selector anidado. Estas métricas corresponden exclusivamente al proxy temporal:

| Procedimiento | Accuracy, media ± DE por grupo | Macro-F1, media ± DE por grupo |
|---|---:|---:|
| Logística RMS + MDF | 0,653 ± 0,089 | 0,627 ± 0,098 |
| Logística, 10 features | 0,675 ± 0,093 | 0,635 ± 0,126 |
| SVM-RBF | 0,672 ± 0,110 | 0,636 ± 0,134 |
| Random Forest | 0,671 ± 0,106 | 0,637 ± 0,126 |
| Selector anidado | 0,663 ± 0,105 | 0,624 ± 0,129 |

El selector anidado no demuestra mejora frente a RMS+MDF: diferencia pareada de Accuracy +0,0093, IC bootstrap del 95% [−0,0373; +0,0546], y de macro-F1 −0,0032, IC [−0,0591; +0,0509]. Ambos intervalos cruzan cero. Matriz OOF del selector, filas de referencia y columnas predichas, orden `temprana/tardía`: `[[1473, 856], [995, 2185]]`. Se incluyen conteos y normalización por fila para cada procedimiento en el paquete y en la tabla pública de matrices agregadas.

El experimento auxiliar empleó dos etapas separadas: un Random Forest estima primera/segunda mitad de una ráfaga EMG y una logística de etapa 2 combina sus probabilidades OOF con las mismas features para clasificar el proxy temporal. El directo obtuvo Accuracy 0,6752 ± 0,0935 y macro-F1 0,6355 ± 0,1262; el condicionado obtuvo Accuracy 0,6701 ± 0,1022 y macro-F1 0,6293 ± 0,1330. La diferencia pareada de macro-F1 es −0,0062, IC 95% [−0,0232; +0,0082], sin mejora demostrada. La cabeza auxiliar obtuvo Accuracy 0,5775 ± 0,0783 y macro-F1 0,5420 ± 0,1028 al predecir mitades de ráfaga, no movimiento observado. Estas son métricas de objetivos proxy y no de fatiga o fase verdaderas.

Los intervalos pareados de Accuracy se generan con [el constructor de contrastes](../tools/build_accuracy_contrasts.py), usando 5.000 remuestreos por grupo y semilla 20261028. No se ajustaron etiquetas, pliegues o umbrales para elevar Accuracy/F1. Optimizar contra el mismo holdout aumentaría el riesgo de sobreajuste y no convertiría el proxy en una etiqueta fisiológica.

## 4. Arquitectura objetivo y condiciones de entrenamiento

La arquitectura futura separa dos referencias y dos tareas:

1. **Fase:** sEMG preprocesada → clasificador de extensión/flexión supervisado por ángulo de rodilla o video/kinemática revisados y sincronizados.
2. **Fatiga directa:** las mismas features por ventana → clasificador de fatiga/no fatiga supervisado por una definición mecánica aprobada, derivada de fuerza/torque calibrados y sincronizados.
3. **Cascada:** la etapa de fatiga recibe features más probabilidades de fase fuera de pliegue en entrenamiento y probabilidades predichas en evaluación.
4. **Expertos por fase:** expertos de fatiga separados se ponderan con probabilidades de fase predichas; se comparan en los mismos participantes, pliegues y soporte que el modelo directo.

Las ecuaciones, el candidato de regla torque/MDC95, los diagramas de tareas y las reglas de exclusión están en el [protocolo de referencia](reference_fatigue_protocol.md). La regla torque/MDC95 requiere revisión biomecánica y precisión/repetibilidad del instrumento antes de crear etiquetas. La figura de señal real, PSD antes/después y respuesta del filtro está en el paquete privado; el repositorio público incluye el generador y diagramas sin señal cruda.

Fine-tuning solo sería razonable con datos externos etiquetados y compatibles, grupos independientes y un plan fijado antes de evaluar. Reinforcement learning no corresponde a este problema de clasificación supervisada de ventanas: no hay acciones ni recompensas secuenciales.

## 5. Archivos de evidencia

- [Protocolo histórico ejecutado](protocol.md): definición del proxy, filtros, ventanas, features y validación.
- [Protocolo prospectivo con referencias](reference_fatigue_protocol.md): ecuaciones de filtros, ventana, features, etiquetas y arquitecturas.
- [Auditoría de características candidatas](CANDIDATE_FEATURE_AUDIT_2026-11-09.md): fórmulas y redundancia descriptiva; no constituye entrenamiento ni mejora validada.
- [Datos faltantes y Gate G0](REFERENCE_DATA_REQUIREMENTS.md): campos necesarios para fatiga y fase reales.
- [Resultados agregados y suplemento](INFORME_EVIDENCIAS.md): métricas, contrastes, matrices y relaciones entre modelos.
- [Historia de diapositivas](DEFENSE_STORYBOARD.md): 16 diapositivas principales de 600 s y 14 anexos de evidencia.

**Conclusión:** la corrección científica impide que el producto se presente como detector entrenado de fatiga. El paquete mejora trazabilidad, preprocesamiento documentado, fórmulas, arquitectura y visualización; el rendimiento fisiológico seguirá sin poder calcularse hasta disponer de las referencias sincronizadas indicadas arriba.
