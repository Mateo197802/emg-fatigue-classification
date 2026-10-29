# Presentación de la defensa

Revisión solicitada el 28 de octubre de 2026. Se elimina el bloque «Evidencia pendiente» y se reorganiza la exposición en 16 diapositivas principales, con 600 segundos de narración, y siete anexos.

1. Objetivo e integrantes.
2. Corpus y denominadores cuantitativos.
3. Señal registrada y diagnóstico temporal.
4. Preanálisis descriptivo de RMS, MDF y MNF.
5. Preprocesamiento.
6. Ventanas y etiquetas.
7. Diez características EMG.
8. Validación por grupos.
9. Modelos y selección de parámetros.
10. Rendimiento principal.
11. Errores y variación entre grupos.
12. Arquitectura auxiliar de dos etapas.
13. Variantes y controles.
14. Rendimiento con la arquitectura auxiliar.
15. Resultados de cada etapa y matrices de confusión.
16. Conclusiones.

Los anexos conservan relación entre modelos, sensibilidades, matrices completas y bibliografía. Las tablas, curvas y arquitectura principal son editables. Las curvas muestran señal original o resúmenes de resultados congelados. No se reentrenan modelos ni se seleccionan ejemplos por rendimiento. Las clases siguen siendo temprana/tardía y las etiquetas auxiliares siguen siendo mitades de ráfagas EMG.

`tools/build_defense_inputs.py` prepara entradas privadas a partir del archivo y tablas congeladas. `tools/build_defense_v2.mjs` construye la presentación reorganizada. El paquete privado conserva entradas, notas y comprobantes. GitHub conserva código y resultados agregados, sin señal cruda ni identificadores de participantes.
