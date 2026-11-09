# Referencias necesarias para entrenar fatiga real

Estado de la auditoría local: 2026-11-09. Este documento especifica los archivos que faltan; no es un resultado de entrenamiento ni una solicitud que se haya enviado a terceros.

## Qué contienen las fuentes disponibles

El ZIP del proyecto tiene 403 miembros. La fuente principal contiene 140 CSV EMG; el RAR adicional contiene 20 CSV EMG y se mantiene separado hasta resolver identidad y procedencia. Los CSV auditados contienen tiempo relativo, cuentas ADC y voltaje calculado. Los dos libros Excel que vienen dentro del ZIP son una matriz bibliográfica y una matriz de filtros/ruido/QC, no registros de torque, ángulo ni fatiga de los participantes. No se encontró en el material local una serie temporal calibrada de fuerza/torque, video o ángulo sincronizado, MVC/Borg por participante ni una clave independiente de identidad/sesión.

Por tanto, los resultados publicados en este repositorio siguen midiendo acuerdo con una etiqueta temporal temprana/tardía. La clasificación de mitades de ráfagas EMG tampoco es una etiqueta de extensión/flexión.

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
