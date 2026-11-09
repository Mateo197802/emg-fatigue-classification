# Guion de la defensa

16 diapositivas principales. Tiempo previsto: 600 segundos. Catorce anexos para preguntas.

## 1. Clasificación proxy temporal con sEMG

Tiempo previsto: 20 s.

Este estudio clasifica una etiqueta proxy temprana frente a tardía construida por posición dentro de cada grabación. No hay una medición independiente para afirmar fatiga muscular real. Presentaremos el corpus, la señal, el preanálisis, el procesamiento, los modelos y el experimento auxiliar, distinguiendo los resultados ejecutados de la arquitectura propuesta.

Fuente: Proyecto integrador de biomecánica.

## 2. Corpus y análisis cuantitativo

Tiempo previsto: 35 s.

El archivo principal contiene 140 CSV y 4.256.257 muestras. Las 138 grabaciones R1 a R5 entran al procesamiento y dos R0 quedan como controles. Hay quince grupos derivados de carpetas. Ambas piernas y todas las series permanecen juntas al validar. El RAR se conserva separado y no agrega dos sujetos verificados. La duración mediana entre timestamps es 27.7 segundos y puede incluir huecos.

Fuente: inventory.csv y coverage.csv.

## 3. Señal registrada y diagnóstico temporal

Tiempo previsto: 35 s.

Los CSV aportan tiempo relativo y cuentas ADC con aproximadamente mil muestras por segundo. Comparamos el mismo intervalo almacenado y filtrado. Elegimos el primer registro con un tramo válido sin consultar etiquetas ni rendimiento. Dibujamos una de cada cuatro muestras, pero el análisis conserva todas. Revisamos huecos, jitter, segmentos constantes y repetición de extremos. Las cuentas no acreditan una amplitud muscular calibrada en milivoltios.

Fuente: Archivo original, qc.py y preprocessing.py.

## 4. Preanálisis descriptivo de la señal

Tiempo previsto: 35 s.

Estas curvas usan las 9.289 ventanas activas con características válidas, incluido el centro sin etiqueta. Primero calculamos medianas por registro y decil temporal, después medianas por grupo y finalmente media entre grupos. Los valores describen el corpus y no prueban fatiga. Amplitud y frecuencia también dependen de esfuerzo, movimiento y adquisición. Las curvas no participan en la creación de etiquetas.

Fuente: feature_trajectories_by_group.csv y features.parquet.

## 5. Preprocesamiento de la señal

Tiempo previsto: 45 s.

Separamos tramos cuando un hueco supera el máximo de 1,5 milisegundos y 1,5 veces el intervalo mediano. Rechazamos jitter relativo mayor al cinco por ciento y segmentos constantes. Usamos Butterworth de banda veinte a cuatrocientos hercios, prototipo orden cuatro y banda digital orden ocho, bidireccional sin desfase. Excluimos cien milisegundos por extremo. La rama espectral conserva señal sin rectificar. El notch queda desactivado en el principal y se evalúa como sensibilidad. Es procesamiento offline.

Fuente: configs/study.yaml, qc.py y preprocessing.py.

## 6. Ventanas y etiqueta temporal proxy

Tiempo previsto: 45 s.

Los tercios usan los límites temporales originales antes de filtrar y excluir. Ventanas completas del primer tercio reciben clase cero y del último clase uno. Centro y fronteras quedan fuera. Exigimos al menos setenta y cinco por ciento de actividad candidata. De 33.090 ventanas candidatas quedan 32.474 con calidad válida, 9.289 activas con características válidas y 5.509 tempranas o tardías. Una ventana descartada no se convierte en no fatiga. La ventana es la unidad de observación; la clase aprendida es una posición temporal, no fatiga muscular medida.

Fuente: label_audit.json y coverage.csv.

## 7. Diez características por ventana

Tiempo previsto: 35 s.

RMS, MAV y longitud de la forma de onda por segundo resumen amplitud y variación. MDF, MNF y razón de potencia veinte a sesenta frente a sesenta a cuatrocientos hercios resumen el espectro. Cuatro energías db4 describen A3, D3, D2 y D1. El escalador se ajusta sólo con entrenamiento. No usamos tiempo, identidad, lado o serie como predictores ni inventamos normalización MVC. La segunda etapa y el comparador directo conservan las mismas diez características.

Fuente: features.py y feature_schema.json.

## 8. Metodología de validación por grupos

Tiempo previsto: 45 s.

Cada ronda externa retiene uno de los quince grupos completos. Ambas piernas y todas sus series quedan fuera del ajuste. Con los otros catorce grupos hacemos cuatro particiones internas para elegir parámetros según macro-F1 medio por grupo. Escalador y modelo se ajustan exclusivamente en entrenamiento. Guardamos predicciones OOF una vez por ventana y procedimiento. La incertidumbre usa bootstrap de grupos. No dividimos ventanas al azar entre entrenamiento y prueba.

Fuente: evaluation.py, splits.json y evaluation_receipt.json.

## 9. Modelos y selección de parámetros

Tiempo previsto: 35 s.

La mayoría es el mínimo trivial. La logística RMS y MDF es la referencia simple. La logística completa evalúa ampliar variables. SVM y RF permiten relaciones no lineales. El selector elige familia y parámetros sólo con validación interna, sin combinar votos. Se exploran diecinueve configuraciones. El artefacto final es RF de trescientos árboles, profundidad tres y hoja mínima dos, ajustado con la cohorte completa después de la evaluación.

Fuente: evaluation.py, inner_search.csv y model.receipt.json.

## 10. Rendimiento de los modelos para el proxy temporal

Tiempo previsto: 45 s.

Todos usan las mismas 5.509 ventanas con objetivo temprano/tardío, no fatiga/no fatiga. Accuracy, macro-F1 y AUC son medias ± desviación entre quince grupos. Para el selector frente a RMS+MDF, Δ Accuracy=0.0093, IC 95% [-0.0373, 0.0546]; Δ macro-F1=-0.0032, IC 95% [-0.0591, 0.0509]. Ambos intervalos incluyen cero; no hay mejora clara ni métrica de fatiga fisiológica, porque la etiqueta sigue siendo posición temporal.

Fuente: evaluation/metrics_summary.csv y paired_differences.csv.

## 11. Errores del procedimiento principal

Tiempo previsto: 35 s.

El selector acierta 1.473 de 2.329 ventanas tempranas y 2.185 de 3.180 tardías. Confunde 856 tempranas y 995 tardías. Macro-F1 varía aproximadamente entre 0,372 y 0,806 según el grupo. La matriz agrupa ventanas y no sustituye la media y dispersión entre grupos. No atribuimos errores a una causa fisiológica individual sin referencia independiente.

Fuente: evaluation/oof_predictions.parquet y evaluation/results.csv.

## 12. Auxiliar EMG: dos etapas sobre una etiqueta proxy

Tiempo previsto: 45 s.

La hipótesis es que una representación local de fase ayude a interpretar cambios de señal. El RF auxiliar usa las diez características para distinguir primera y segunda mitad de ráfagas de medio a ocho segundos. Tiene cien árboles, profundidad tres y hoja mínima cinco. Su score no calibrado entra a una logística junto con las diez variables originales. La variante principal agrega score por RMS, MDF y MNF. Los scores del entrenamiento son cross-fitted por grupo, también dentro de selección interna. La fase candidata no dispone de extensión y flexión observadas.

Fuente: phase_spike.py y protocolo anterior a resultados.

## 13. Diseño del experimento auxiliar

Tiempo previsto: 35 s.

Las cinco variantes comparten ventanas, etiquetas, escalado y particiones. El directo reproduce la logística completa. EMG más score mide información adicional, y condicionado agrega interacciones. Sólo score mide lo que transmite la primera etapa sin EMG original. El control aleatorio repite el procedimiento con etiquetas auxiliares aleatorias para examinar capacidad extra. Hay 3.337 etiquetas auxiliares de 996 ráfagas. Sólo grupos de entrenamiento aportan esta supervisión.

Fuente: model_registry.csv y protocolo original del experimento.

## 14. Rendimiento auxiliar sobre el mismo proxy temporal

Tiempo previsto: 40 s.

Para el proxy temporal, el directo alcanza Accuracy 0,6752 y macro-F1 0,6355; el condicionado, Accuracy 0,6701 y macro-F1 0,6293. El contraste pareado condicionado−directo es Δ Accuracy=-0.0051 (IC 95% [-0.0222, 0.0095]) y Δ macro-F1=-0.0062 (IC 95% [-0.0232, 0.0082]). Ambos intervalos incluyen cero. Esta prueba no usó extensión/flexión observadas ni etiqueta fisiológica de fatiga.

Fuente: phase_experiment/metrics_summary.csv y paired_differences.csv.

## 15. Resultados de cada etapa de la cascada

Tiempo previsto: 35 s.

Las matrices tienen objetivos y soportes distintos. La auxiliar usa 1.663 primeras mitades y 1.674 segundas: Accuracy media por grupo 0,5775 ± 0,0783, macro-F1 0,5420 ± 0,1028 y AUC 0,6212 ± 0,1288. La logística condicionada usa las 5.509 ventanas tempranas y tardías: Accuracy 0,6701 ± 0,1022 y macro-F1 0,6293 ± 0,1330. No se comparan los totales ni se interpreta la primera matriz como extensión y flexión observadas. Ninguna es una matriz de fatiga/no fatiga.

Fuente: confusion_matrices.csv y auxiliary_phase_fold_metrics.csv.

## 16. Conclusiones

Tiempo previsto: 35 s.

El experimento ejecutado predice una etiqueta temporal proxy, no fatiga muscular/no fatiga. El selector no mejora claramente a RMS y MDF; la representación de mitades de ráfaga tampoco mejora la clasificación proxy directa. Por eso no existe una Accuracy o macro-F1 válida de fatiga fisiológica para optimizar con este ZIP. La siguiente evaluación requiere torque o fuerza calibrados y sincronizados; la etapa de fase requiere cinemática sincronizada. Las arquitecturas directas, en cascada y de expertos quedan propuestas hasta disponer de esas referencias.

Fuente: Evaluación principal y experimento auxiliar.

## 17. Anexo: Relación entre todos los modelos

Anexo para preguntas.

Son seis procedimientos principales, cinco variantes y una cabeza auxiliar. Directo reproduce logistic_all. El selector elige una configuración, sin votos. El RF final de trescientos árboles se ajusta con toda la cohorte y no añade una estimación OOF.

Fuente: results/phase_experiment/model_registry.csv.

## 18. Anexo: Sensibilidades y ablaciones

Anexo para preguntas.

Nueve sensibilidades prespecificadas y tres ablaciones. Las primeras usan logística fija RMS y MDF, con soportes distintos según exclusiones. Las ablaciones usan selección interna. Se conservan los resultados originales sin seleccionar el escenario más favorable.

Fuente: sensitivity_summary.csv y ablation_summary.csv.

## 19. Anexo: Matrices de los modelos principales

Anexo para preguntas.

Conteos OOF. Cada procedimiento usa 5.509 ventanas. La incertidumbre principal se calcula entre grupos.

Fuente: results/phase_experiment/confusion_matrices.csv.

## 20. Anexo: Confusiones principales normalizadas

Anexo para preguntas.

Proporciones por fila. Clase 0 temprana, clase 1 tardía. La incertidumbre principal se calcula entre grupos.

Fuente: results/phase_experiment/confusion_matrices.csv.

## 21. Anexo: Matrices de las variantes auxiliares

Anexo para preguntas.

Mismo soporte de 5.509 ventanas. Directo reproduce logística 10. La incertidumbre principal se calcula entre grupos.

Fuente: results/phase_experiment/confusion_matrices.csv.

## 22. Anexo: Confusiones auxiliares normalizadas

Anexo para preguntas.

Proporciones por fila. Matrices de la segunda etapa. La incertidumbre principal se calcula entre grupos.

Fuente: results/phase_experiment/confusion_matrices.csv.

## 23. Anexo: Bibliografía del experimento auxiliar

Anexo para preguntas.

Cheah et al. usan fase medida con IMU y una referencia temporal al final de repeticiones. Potvin estudia efectos cinemáticos sobre EMG. Guzu et al. combinan ejercicio y estados EMG con otro diseño. Motivan estudiar fase, pero no validan nuestras etiquetas ni una mejora de esta cascada.

Fuente: docs/PHASE_REFERENCES.md.

## 24. Anexo: Ventanas y solapamiento

Anexo para preguntas.

El protocolo histórico fijado usa ventanas deslizantes de 250 milisegundos con 50 por ciento de solapamiento. Para un estudio futuro con referencias independientes se propone 500 milisegundos como análisis principal y 250 milisegundos como sensibilidad separada. La relación general es N igual a redondear T por fs, H igual a redondear N por uno menos solapamiento y la separación nominal de bins es fs dividido entre N. A un kilohercio son 500 muestras y un avance de 250 para el análisis principal, con separación nominal de dos hercios. Para la sensibilidad y el histórico son 250 muestras y avance de 125, cuatro hercios. La ventana más larga mejora la resolución nominal; el solapamiento da mayor densidad temporal, aunque crea ventanas dependientes. Esta ilustración no representa una etiqueta.

Fuente: Configuración histórica congelada y configs/reference_study.yaml.

## 25. Anexo: Características candidatas y auditoría

Anexo para preguntas.

Se calcularon cinco características candidatas en ventanas activas válidas sin consultar etiquetas ni ajustar un modelo. Las fórmulas se muestran para dejar trazable qué se evaluó. iEMG y varianza fueron redundantes con MAV y RMS bajo duración fija. Las correlaciones se describen como auditoría exploratoria: las ventanas se solapan, los 15 grupos vienen de carpetas y la identidad no está verificada. Ninguna característica se agregó a los clasificadores, y no se puede atribuir mejora de Accuracy o F1 sin referencias independientes de fatiga.

Fuente: docs/CANDIDATE_FEATURE_AUDIT_2026-11-09.md; salida granular privada.

## 26. Anexo: Filtros y ecuaciones de ventana

Anexo para preguntas.

El pasa banda digital se diseña por separado para cada frecuencia de muestreo verificada. El prototipo Butterworth de orden cuatro se transforma en un pasa banda digital de orden ocho y se implementa en secciones de segundo orden. La forma recursiva muestra los coeficientes y el filtrado hacia adelante y atrás. Notch es solo una sensibilidad activada si el diagnóstico PSD muestra interferencia de red prominente; no forma parte del procesamiento principal. Las fórmulas del periodograma corresponden a la ventana Hann y al estimador unilateral de densidad que usa scipy.signal.periodogram, no a Welch. Las ventanas se extraen sin cruzar huecos o transiciones de referencia.

Fuente: SciPy butter, sosfiltfilt, periodogram, ventanas Hann e iirnotch: https://docs.scipy.org/doc/scipy/reference/signal.html.

## 27. Anexo: Características y fórmulas

Anexo para preguntas.

El vector X contiene diez variables calculadas por ventana. RMS y MAV resumen amplitud; longitud de onda resume cambio local; MDF, MNF y la razón de bandas resumen distribución espectral; cuatro coeficientes db4 dan energía relativa por escala. Todos los valores se calculan sobre la señal centrada, filtrada y expresada en cuentas del convertidor. Las amplitudes no son milivoltios calibrados. El análisis histórico usa longitud de onda dividida por N sobre fs; la versión corregida usa el lapso entre primera y última muestra. Esta segunda versión tiene nombre de esquema separado y no cambia resultados congelados.

Fuente: features.py, feature_schema.json y docs/reference_fatigue_protocol.md.

## 28. Anexo: Arquitectura con referencias independientes

Anexo para preguntas.

La cinemática sincronizada crea las etiquetas de extensión y flexión; torque o fuerza calibrados y sincronizados crean la etiqueta de fatiga aprobada. La misma matriz de características X entrena un modelo directo, una cascada de fase y fatiga, y expertos con compuerta suave. En la cascada, la etapa 2 solo recibe probabilidades de fase fuera de pliegue en entrenamiento y fase predicha en prueba. La salida de todos se evalúa en los mismos participantes retenidos. Sin las referencias independientes de fuerza, torque, cinemática e identidad que faltan en el ZIP, el diagrama es una propuesta sin entrenamiento ni métricas.

Fuente: docs/reference_fatigue_protocol.md; las referencias técnicas se citan en docs/PHASE_REFERENCES.md.

## 29. Anexo: Ecuaciones de etiquetas y arquitecturas

Anexo para preguntas.

Las ecuaciones separan las dos tareas supervisadas. El modelo de fase aprende de ángulo, video o cinemática revisados. El modelo de fatiga aprende de una regla mecánica aprobada basada en torque, su error de medición y persistencia entre evaluaciones. La logística directa usa X. La cascada usa X y probabilidades de fase fuera de pliegue. Los especialistas multiplican sus predicciones por probabilidades de fase. La cabeza auxiliar realmente ejecutada usa un RF para predecir mitad de ráfaga EMG y una logística para la posición temporal del registro; esta fórmula no es una forma alternativa de entrenar fatiga real.

Fuente: docs/reference_fatigue_protocol.md; protocolo histórico y model_registry.csv.

## 30. Anexo: Señal real y respuesta del filtro

Anexo para preguntas.

La figura procede de un intervalo de un segundo elegido por la primera ventana continua válida según control de calidad, sin mirar las etiquetas ni el resultado de ningún modelo. Se muestran cuentas ADC originales centradas, salida del mismo filtro SOS, periodogramas Hann antes y después, y respuesta digital exacta para la frecuencia de muestreo registrada. La tasa efectiva fue prácticamente mil hercios. La imagen es evidencia de procesamiento, no de rendimiento fisiológico ni de una clase de fatiga.

Fuente: signal_preprocessing_evidence.receipt.json; figura agregada privada sin identificador de registro.
