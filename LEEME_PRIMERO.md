# Proyecto EMG listo para revisión

Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez

El proyecto procesa la señal sEMG, compara clasificadores y evalúa una arquitectura auxiliar de dos etapas. El resultado medido corresponde a ventanas del primer y último tercio de cada registro. La arquitectura auxiliar no mostró una mejora clara frente al modelo directo con diez características.

## Recorrido de lectura

1. [Informe de evidencias](docs/INFORME_EVIDENCIAS.md): metodología, denominadores, resultados, arquitectura y alcance de las conclusiones.
2. [Mapa de evidencia](docs/MAPA_EVIDENCIA.csv): conecta cada afirmación con la sección del informe, las diapositivas y los archivos que la respaldan.
3. [Protocolo](docs/protocol.md) y [resultados principales](docs/RESULTS.md).
4. [Experimento de fase](docs/PHASE_EXPERIMENT.md), [relación de modelos](results/phase_experiment/model_registry.csv), [matrices](results/phase_experiment/confusion_matrices.csv) y [bibliografía](docs/PHASE_REFERENCES.md).
5. [Presentación](docs/DEFENSE_STORYBOARD.md): 16 diapositivas principales para 600 segundos y 7 anexos de evidencia.
6. [Instalación y reproducción](README.md#installation), [pruebas](tests), [alcance de la verificación](docs/VERIFICATION.md) y [CI de GitHub](https://github.com/Mateo197802/emg-fatigue-classification/actions).

## Qué contiene cada entrega

| Ubicación | Contenido |
|---|---|
| Este GitHub | Código, protocolo, resultados agregados, matrices, arquitectura, referencias, pruebas y manifiesto SHA256 |
| ZIP privado completo | Lo anterior, informe PDF de 21 páginas y Word, presentación de 23 diapositivas en PPTX/PDF, guion, originales, predicciones OOF, búsquedas, modelo y recibos |
| `LEEME_PRIMERO.md` en la raíz del ZIP | Acceso directo a los archivos del paquete y comandos para comprobar su integridad |

El ZIP conserva las fuentes originales y distingue los resultados principales del suplemento. El modelo principal de 300 árboles conserva sus pesos. El RF auxiliar de 100 árboles y la logística de la cascada cumplen funciones diferentes. `baseline_10` reproduce `logistic_all`; no constituye otro resultado independiente.

## Verificación del paquete privado

El paquete incluye `SHA256SUMS` para todos sus archivos y `codigo/tools/verify_delivery.py`, que utiliza sólo la biblioteca estándar de Python. Después de extraerlo, desde su raíz:

```powershell
python codigo/tools/verify_delivery.py --zip 'C:/ruta/EMG_COMPLETO_INTEGRADO_PRIVADO_20261028.zip'
```

Junto al ZIP se entregan `COMPROBANTE_PAQUETE.json`, `VERIFICACION_FINAL.json` y `SHA256SUMS_ENTREGA.txt`. Sus hashes comprueban consistencia del contenido. El estado de la CI debe corresponder al commit de `VERIFICACION_FINAL.json`. Los comprobantes distinguen la revisión del informe, la revisión de la presentación y la del código publicado.

Los originales y las predicciones individuales están en el paquete privado. Este repositorio publica evidencia agregada. Las pruebas y hashes acreditan la implementación y la trazabilidad; las etiquetas disponibles conservan el alcance temporal declarado.
