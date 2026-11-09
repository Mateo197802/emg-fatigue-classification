# Auditoría descriptiva de características candidatas

**Estado:** cálculo ejecutado con el ZIP EMG; no es entrenamiento ni evaluación de fatiga.

## Alcance y procesamiento

Se procesaron 138 grabaciones formales, 15 grupos derivados de carpeta cuya identidad de participante sigue sin verificarse, 355 tramos de adquisición QC válidos y 9.289 ventanas QC válidas con ocupación candidata de actividad suficiente. La auditoría usa únicamente el ZIP EMG anidado con SHA-256 `0723c36b27a6d5b7258d5f29c26de7898a83261011b8f323ccd42e5c5a5b2776`.

La extracción usa el pipeline histórico de señal: tramos continuos, filtro pasa banda Butterworth 20–400 Hz en SOS, filtrado fuera de línea hacia delante y atrás, máscara de borde de 100 ms y ventanas deslizantes de 250 ms con 50 % de solapamiento. Se requiere que toda la ventana pase la máscara QC y que al menos el 75 % cumpla la regla de actividad candidata. Esa regla describe actividad EMG; no es una etiqueta de contracción verificada.

Las etiquetas temprano/tardío no se leen ni se guardan. No se ajustó ningún clasificador y no se calcularon Accuracy, F1 ni matrices de confusión con estas características.

## Características y fórmulas

Para una ventana filtrada de `N` muestras `x[n]` en cuentas ADC, `fs` en Hz y duración definida como `T=N/fs`:

| Característica candidata | Fórmula usada | Unidad |
|---|---|---|
| Cruces por cero por segundo | `C_ZCR/T`; `C_ZCR = Σ 1[(x[n] ≥ τx ∧ x[n+1] < −τx) ∨ (x[n] ≤ −τx ∧ x[n+1] > τx)]`, `τx=0` en esta auditoría | cruces/s |
| Cambios de signo de pendiente por segundo | `C_SSC/T`; `d[n]=x[n+1]−x[n]`; `C_SSC = Σ 1[(d[n] > τd ∧ d[n+1] < −τd) ∨ (d[n] < −τd ∧ d[n+1] > τd)]`, `τd=0` | cambios/s |
| iEMG | `Σ |x[n]| / fs` | cuentas·s |
| Varianza poblacional | `(1/N) Σ (x[n]−x̄)²` | cuentas² |
| Entropía espectral normalizada | `q[k]=Pxx[k]/ΣPxx`; `−Σ q[k] ln(q[k]) / ln(K)` sobre `K` bins de 20–400 Hz | adimensional, entre 0 y 1 |

Los parámetros para los cruces y cambios de pendiente se fijaron en cero antes de calcular los descriptores; no se optimizaron usando etiquetas. `Pxx` es la densidad del periodograma unilateral con ventana Hann. La definición del periodograma se mantiene separada de Welch: aquí no se promedian subsegmentos Welch.

## Auditoría de redundancia

Correlaciones de Spearman descriptivas entre las características existentes y las candidatas:

| Par | Ventanas solapadas, n=9.289 | Medianas por grupo, n=15 |
|---|---:|---:|
| MAV – iEMG | 1,000 | 1,000 |
| RMS – varianza | 1,000 | 1,000 |
| MNF – entropía espectral | 0,862 | 0,957 |
| MDF – cruces por cero | 0,740 | 0,887 |
| Wavelet D1 – cambios de signo de pendiente | 0,840 | 0,975 |
| Cruces por cero – cambios de signo de pendiente | 0,736 | 0,904 |

Las relaciones MAV–iEMG y RMS–varianza son redundancias esperables para ventanas de duración fija y señal centrada: `iEMG` es proporcional a MAV y la varianza es RMS al cuadrado. Los demás coeficientes muestran asociación descriptiva con características espectrales o wavelet ya existentes; no demuestran utilidad predictiva.

Las 9.289 ventanas solapadas no son observaciones independientes. El resumen por grupo usa 15 identificadores derivados de carpetas; no se afirma que sean 15 participantes verificados. No se calcularon pruebas de hipótesis ni intervalos poblacionales a partir de estas correlaciones.

## Decisión

Las cinco características quedan en el esquema exploratorio `emg_candidate_features_v1` y fuera de `emg_window_features_v1` y `emg_window_features_v2_sample_span`. Se mantienen fuera del clasificador: iEMG y varianza duplican información de MAV y RMS bajo esta definición, mientras que ZCR, cambios de pendiente y entropía deben probarse con etiquetas independientes y selección dentro de pliegues de entrenamiento antes de atribuirles valor para fatiga.

## Reproducción y evidencia privada

La auditoría se ejecuta con `python -m emg_fatigue.candidate_feature_audit`, usando `configs/study.yaml`. El comando procesa el ZIP EMG anidado y recibe una carpeta de salida privada distinta por versión. La carpeta privada conserva `candidate_features_private.parquet`, resumen por grupo, tabla de correlaciones, auditoría por tramo y `candidate_feature_audit.receipt.json` con hashes, configuración y procedencia del código. Las filas por ventana y los identificadores no se publican en GitHub.
