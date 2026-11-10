# Referencias necesarias para entrenar fatiga real

Estado de la auditoría local: 2026-11-09. Este documento especifica los archivos que faltan; no es un resultado de entrenamiento ni una solicitud que se haya enviado a terceros.

## Qué contienen las fuentes disponibles

El ZIP del proyecto tiene 403 miembros. La fuente principal contiene 140 CSV EMG; el RAR adicional contiene 20 CSV EMG y se mantiene separado hasta resolver identidad y procedencia. Los CSV auditados contienen tiempo relativo, cuentas ADC y voltaje calculado. Los dos libros Excel que vienen dentro del ZIP son una matriz bibliográfica y una matriz de filtros/ruido/QC, no registros de torque, ángulo ni fatiga de los participantes. No se encontró en el material local una serie temporal calibrada de fuerza/torque, video o ángulo sincronizado, MVC/Borg por participante ni una clave independiente de identidad/sesión.

Por tanto, los resultados publicados en este repositorio siguen midiendo acuerdo con una etiqueta temporal temprana/tardía. La clasificación de mitades de ráfagas EMG tampoco es una etiqueta de extensión/flexión.

## Auditoría de presencia y aptitud de referencias

La tabla separa lo indicado por la guía universitaria de los archivos que realmente contiene el ZIP. La guía describe el protocolo previsto; por sí sola no demuestra qué mediciones se realizaron ni cómo se sincronizaron.

| Referencia o metadato | Presencia y fuente/ruta | Unidad | Resolución temporal | Calibración | Sincronización | Granularidad que permite | Limitación observada |
|---|---|---|---|---|---|---|---|
| sEMG almacenada | Sí: ZIP principal → archivo EMG anidado → 140 CSV | `ADC_Value` en cuentas; `Voltage_mV` exportado | Timestamps relativos por muestra; medianas por archivo cercanas a 1.000 Hz; existen huecos | No se aporta certificado, modo del canal, rieles ni ganancia verificada | Solo reloj relativo por CSV; no hay reloj compartido con biomecánica | Muestra y archivo | Es señal para reanálisis; no proporciona por sí sola la etiqueta fisiológica independiente |
| Fuerza o torque | No encontrado en el ZIP, CSV EMG ni dos libros incluidos | No disponible | No disponible | No disponible | No disponible | Ninguna; no admite etiqueta por ventana, repetición o sesión | Es la referencia mecánica requerida para el resultado primario |
| MVC pre/post | No encontrado; la guía solo indica la medición prevista | No disponible en registros del corpus | No disponible | No disponible | No disponible | Ninguna en los datos; incluso MVC pre/post sería resumen de sesión | No ubicaría por sí sola el inicio en cada repetición o ventana |
| Borg CR-10 | No encontrado; la guía indica el momento previsto de consulta | No disponible en registros del corpus | No disponible | No aplica como medida mecánica | No sincronizada en los archivos | Ninguna; no expande a etiquetas por ventana | Es esfuerzo percibido; no reemplaza una etiqueta mecánica objetiva |
| Ángulo de rodilla o cinemática | No encontrado en los miembros locales | No disponible | No disponible | No disponible | No disponible | Ninguna; no admite extensión/flexión por ventana | Sin convención angular ni resolución/ruido no puede fijarse el sentido ni la banda muerta |
| Video | No encontrado en los miembros locales | No disponible | No disponible | No aplica; faltan video y marcas | No hay evento común con sEMG | Ninguna | No puede revisión humana de fases o repeticiones |
| Eventos/marcadores de repetición | No encontrados; `R1–R5` son códigos de serie del nombre, no eventos verificados | No disponible | No disponible | No aplica | Sincronización no disponible | Ninguna repetición identificable | No se puede reconstruir cadencia, inversión, fallo o repetición válida |
| Identificadores de participante y sesión | Parcial: grupos derivados de carpetas y códigos de lado/serie en nombres | Identificadores locales, no unidades físicas | Una agrupación de archivos; sesión no establecida | No aplica | Enlaces entre archivos no verificados | Grupos de registro condicionales; no participante verificado | No afirmar validación independiente por persona; el RAR queda separado |
| Metadatos de adquisición del equipo | Parciales: nombres, tiempos y cuentas en CSV; instrucciones del equipo en guía | Cuentas ADC y tiempo en segundos | Cerca de 1 kHz según timestamps observados | Modo RAW, configuración, ganancia y calibración no confirmados en metadatos | Sin protocolo/registro de reloj compartido | Archivo | Las instrucciones describen condiciones previstas, no prueban carga, cadencia, montaje ni fallo ejecutados |
| Conversión `Voltage_mV` | Sí en los 140 CSV primarios; contraste en auditoría local de muestras | mV calculados por exportador | Cada muestra | Fórmula aritmética verificada; calibración física no | Hereda el timestamp relativo del CSV | Muestra, como conversión de ADC | `Voltage_mV = round(ADC_Value × 3300 / 4095, 2)`. Error máximo frente a esa fórmula: 0.00 mV en la auditoría. Esto no prueba referencia, offset, ganancia, linealidad, modo del sensor ni límites físicos; el pipeline conserva `ADC_Value` |
| Frecuencia de muestreo | Sí, estimable desde `Relative_Time_s` en los 140 CSV primarios | Hz | Mediana por archivo próxima a 1.000 Hz; auditoría totalizó 246 intervalos mayores a 1,5 ms | Reloj no certificado | Solo dentro de cada archivo | Muestra/archivo | El valor local mediano no elimina huecos ni demuestra frecuencia del ADC; QC divide los tramos y no interpola huecos largos |
| Anotaciones de fatiga independientes | No encontradas | No disponible | No disponible | No disponible | No disponible | Ninguna | Las clases temprano/tardío se derivan del tiempo de grabación y permanecen proxy exploratorio |

### Verificación de calibración ADC y conversión

La conversión del exportador y la frecuencia se verificaron en la auditoría local del ZIP suministrado. La ecuación del exportador no acredita calibración física. Los identificadores o hashes de los CSV no se publican en este documento. Los archivos originales permanecen en el paquete privado.

## Archivos mínimos para la tarea de fatiga

| Archivo o campo | Requisito | Uso |
|---|---|---|
| Fuerza/torque por instante o repetición | Marca de tiempo compartida con EMG; unidad física; identificador/calibración del instrumento; rango, cero y tasa de muestreo; repeticiones válidas y fallidas | Referencia mecánica independiente para definir y alinear el inicio operacional de fatiga |
| Rango angular comparable | Ángulo de rodilla sincronizado o medida isocinética equivalente, junto con la velocidad/rango donde se midió el torque | Evita atribuir a fatiga un cambio causado por una postura distinta |
| Repetibilidad de medición | SEM o mediciones repetidas del sistema para el protocolo e instrumento usados | Calcular el cambio mínimo detectable al 95 % |
| Eventos de sincronización | Pulso común, evento visible/registrado o transformación de relojes con error residual cuantificado | Alinear torque, sEMG, ángulo y repetición |
| Identificadores seudónimos | `participant_id`, `session_id`, `side`, `set_id`, `repetition_id`, sin nombres | Mantener todos los registros de una persona en un mismo pliegue |

Un archivo de MVC pre/post puede apoyar la descripción del cambio entre sesiones, pero no localiza por sí solo el inicio de fatiga en cada ventana. Borg CR-10 puede ser un resultado secundario de esfuerzo percibido; no sustituye la etiqueta mecánica primaria. El 1RM sirve para prescribir carga y tampoco es una etiqueta por ventana.

## Archivos mínimos para la fase de movimiento

| Archivo o campo | Requisito | Uso |
|---|---|---|
| Ángulo de rodilla | Serie temporal con marca de tiempo, unidad, convención del eje y calibración; o video sincronizado apto para marcar eventos | Referencia supervisada para extensión, flexión y puntos de inversión |
| Repeticiones | Inicio/fin o eventos de inversión revisados, ligados a identificadores seudónimos | Asociar cada ventana EMG con una repetición y fase verificadas |
| Sincronización con EMG | Reloj común o evento de alineación y error residual documentado | Evitar etiquetas desplazadas a otra fase |

El ángulo debe declarar su convención. Con el convenio de flexión positiva, `dθ/dt > δ` indica flexión, `dθ/dt < −δ` indica extensión y `|dθ/dt| ≤ δ` queda como transición. `δ` debe derivarse de ruido/resolución del sensor y fijarse antes de evaluar los modelos.

## Regla de etiqueta propuesta

Para cada persona y pierna, obtener una medida de torque en el mismo rango angular durante cada repetición. `T_ref` es la mediana de las primeras tres repeticiones válidas. Con SEM de repetibilidad independiente:

`MDC95 = 1.96 × sqrt(2) × SEM`

Una repetición queda por debajo del criterio cuando la mediana móvil de tres repeticiones es menor que `T_ref − MDC95`. El inicio operacional se ubica en la primera evaluación de dos evaluaciones consecutivas bajo ese umbral. Las ventanas sEMG anteriores al inicio reciben clase 0 y las posteriores, clase 1, solo si el reloj, la repetición, la fase y la medición son válidos. Ventanas que crucen una transición o cuyo estado no pueda alinearse quedan sin etiqueta; nunca se fuerzan a “no fatiga”. El algoritmo no usa características sEMG ni tercios de tiempo para crear la etiqueta.

El código debe guardar la regla, unidad, calibración, SEM, repeticiones de referencia, límites angulares, reloj y motivos de exclusión junto con cada etiqueta. Sin esos metadatos, no se aceptará el archivo como referencia supervisora.

## Plantilla de entrega de referencias

Puede organizarse en una tabla larga con estas columnas mínimas:

```text
participant_id,session_id,side,set_id,repetition_id,timestamp_s,
emg_recording_id,force_N,torque_Nm,knee_angle_deg,angular_velocity_deg_s,
repetition_valid,calibration_id,synchronization_event_id
```

Los campos pueden dividirse en tablas sincronizadas si se conservan sus marcas de tiempo y claves de unión. Entregar además el certificado/archivo de calibración, SEM o repetibilidad, diccionario de unidades y convención angular. No incluir nombres ni otros identificadores personales.

## Puerta de decisión

- Si aparece torque/fuerza calibrado y sincronizado más identificación de participantes verificable, validar las etiquetas y continuar con el modelo de fatiga.
- Si solo aparece cinemática/video sincronizado, se puede evaluar extensión/flexión; eso no habilita entrenamiento de fatiga.
- Si solo aparecen MVC/Borg o los CSV EMG actuales, conservar el experimento como proxy y no afirmar detección de fatiga fisiológica.
- Una nueva cohorte requiere autorización independiente y permanecerá separada de la cohorte oficial.
