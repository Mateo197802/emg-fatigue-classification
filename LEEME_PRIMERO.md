# Proyecto EMG listo para revisión

Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez

La entrega separa tres tareas que no deben confundirse: el clasificador histórico del ZIP del curso (posición temporal, no fatiga), el clasificador externo supervisado por ventana contra autoreporte estable de fatiga percibida y un benchmark externo con etiquetas por ensayo. Solo el segundo predice fatiga percibida por ventana; no mide pérdida mecánica de fuerza. La cascada extensión/flexión→fatiga no se puede entrenar con estos archivos porque faltan referencias sincronizadas de movimiento.

En el corpus del curso, **la ventana es la observación** y las ventanas no son personas independientes. El clasificador histórico no fue entrenado para distinguir fatiga de no fatiga.

## Recorrido de lectura

1. [Informe de evidencias](docs/INFORME_EVIDENCIAS.md): corpus del curso, análisis cuantitativo, metodología, modelos, resultados y límites.
2. [Suplemento binario por ventana](docs/FATIGUE_WINDOW_EXPERIMENT.md): etiquetas alineadas, preprocesamiento, fórmulas, matrices, búsqueda anidada y especialistas.
3. [Auditoría de datos disponibles](docs/REFERENCE_DATA_REQUIREMENTS.md) y [matriz de cumplimiento docente](docs/PROFESSOR_REQUIREMENTS_COMPLIANCE_2026-11-09.csv): qué existe, qué puede etiquetar y qué no.
4. [Mapa de evidencia](docs/MAPA_EVIDENCIA.csv): conecta afirmaciones con informe, diapositivas y archivos de soporte.
5. [Protocolo del curso](docs/protocol.md), [resultados históricos](docs/RESULTS.md) y [suplemento de cascada proxy](docs/PHASE_EXPERIMENT.md).
6. [Presentación y arquitectura](docs/DEFENSE_STORYBOARD.md): corpus, señal, preanálisis, preprocesamiento, metodología, modelos, rendimiento, arquitectura auxiliar y conclusiones.
7. [Guiones de defensa en español e inglés](docs/defense-scripts/README.md): fuentes Markdown y PDFs correspondientes a la presentación publicada.
8. [Instalación y reproducción](README.md#installation), [pruebas](tests), [verificación](docs/VERIFICATION.md) y [CI de GitHub](https://github.com/Mateo197802/emg-fatigue-classification/actions).

## Qué contiene cada entrega

| Ubicación | Contenido |
|---|---|
| Este GitHub | Código, métodos y fórmulas, resultados agregados, matrices, diagramas, referencias, pruebas y guiones Markdown/PDF |
| ZIP privado versionado | Informe principal y suplemento editable/PDF, presentación PPTX/PDF actualizada, guiones español/inglés MD/PDF, evidencia privada de señal, originales, OOF, búsquedas, modelo y recibos |
| `filter_designs.json` dentro del reanálisis privado | Coeficientes SOS reales por tramo y fs, decisión condicional del notch, b/a aplicados si hubo notch, y vínculos a corrida; el recibo de procesamiento guarda su SHA-256 |
| `LEEME_PRIMERO.md` en la raíz del ZIP | Acceso directo a los archivos del paquete y comandos para comprobar su integridad |

El ZIP conserva las fuentes originales y distingue los resultados principales del suplemento. El modelo principal de 300 árboles conserva sus pesos. El RF auxiliar de 100 árboles y la logística de la cascada cumplen funciones diferentes. `baseline_10` reproduce `logistic_all`; no constituye otro resultado independiente. El preanálisis informa controles de señal; las trayectorias por repetición/fase no están disponibles sin marcadores sincronizados y revisados.

## Verificación del paquete privado

El paquete incluye `SHA256SUMS` para todos sus archivos y `codigo/tools/verify_delivery.py`, que utiliza sólo la biblioteca estándar de Python. Después de extraerlo, desde su raíz:

```powershell
python codigo/tools/verify_delivery.py --zip 'C:/ruta/EMG_COMPLETO_PRIVADO_20261109_v3.zip'
```

Junto al ZIP se entregan `COMPROBANTE_PAQUETE.json`, `VERIFICACION_FINAL.json` y `SHA256SUMS_ENTREGA.txt`. Sus hashes comprueban consistencia del contenido. El estado de la CI debe corresponder al commit de `VERIFICACION_FINAL.json`. Los comprobantes distinguen la revisión del informe, la revisión de la presentación y la del código publicado.

Los originales y predicciones individuales están en el paquete privado. GitHub publica resultados agregados reproducibles. El dataset externo de autoreporte permite un ensayo binario por ventana, pero el resultado de EMG es débil y está dominado por un control temporal; no se presenta como detector objetivo de pérdida de fuerza ni como validación de la cohorte del curso.
