# Bibliografía extra: fase del movimiento y fatiga

Búsqueda dirigida del 28 de octubre de 2026 en SciSpace y buscador web; verificación contra texto primario o registro PubMed. No es una revisión sistemática ni un recuento de estudios elegibles. Se seleccionan trabajos por relación metodológica, no por sus mejores métricas. No se incorporó ningún dato externo al entrenamiento.

Pregunta SciSpace: “In surface EMG muscle fatigue assessment during dynamic knee extension or cyclic contractions, does conditioning on movement phase or using a two-stage movement recognition and fatigue classifier improve generalization compared with classifying fatigue directly?” Consultas web: “EMG fatigue recognition movement phase two stage classification dynamic contractions phase dependent fatigue”; “Effects of muscle kinematics Potvin 1997”; títulos y DOI de los trabajos siguientes.

## 1. Cheah, Wan y Yip, 2022 — el antecedente más cercano

Prediction of Muscle Fatigue During Dynamic Exercises based on Surface Electromyography Signals Using Gaussian Classifier. Physical Ergonomics and Human Factors, 63, 62–70. DOI: [10.54941/ahfe1002597](https://doi.org/10.54941/ahfe1002597). [Texto primario completo](https://openaccess-api.cms-conferences.org/articles/download/978-1-958651-39-1_8).

Cuatro participantes realizaron ejercicios abdominales; separaron fases concéntricas y excéntricas mediante IMU antes de extraer características. El artículo advierte que los extremos de la envolvente EMG no siempre coinciden con las fases del movimiento, especialmente bajo fatiga (p. 64). Definieron fatiga como el último 20% de repeticiones y ajustaron modelos por participante. Es un antecedente para segmentar por fase, pero no demuestra una mejora de una cascada EMG→extensión→fatiga en personas nuevas. Su referencia cinemática y su protocolo difieren de los nuestros.

## 2. Potvin, 1997 — fundamento biomecánico

Effects of muscle kinematics on surface EMG amplitude and frequency during fatiguing dynamic contractions. Journal of Applied Physiology, 82(1), 144–151. DOI: [10.1152/jappl.1997.82.1.144](https://doi.org/10.1152/jappl.1997.82.1.144). [Registro y abstract primario en PubMed](https://pubmed.ncbi.nlm.nih.gov/9029209/).

Quince participantes realizaron flexión/extensión de codo. El análisis separó fases concéntricas y excéntricas y rangos angulares. La amplitud y el espectro dependieron también de condiciones cinemáticas. Apoya comparar condiciones mecánicas equivalentes; no es una comparación de clasificadores de dos etapas ni demuestra transferencia al recto femoral. Para este suplemento se verificó el abstract y el registro bibliográfico; no se atribuyen detalles adicionales del texto completo.

## 3. Guzu, Neacsu y Georgian, 2024 — reconocimiento combinado, con cautela

Automatic Muscle Fatigue and Movement Recognition Based on sEMG Signals. 2024 International Symposium ELMAR. DOI: [10.1109/ELMAR62909.2024.10694353](https://doi.org/10.1109/ELMAR62909.2024.10694353). [Texto primario depositado por el autor](https://www.researchgate.net/publication/384656421_Automatic_Muscle_Fatigue_and_Movement_Recognition_Based_on_sEMG_Signals).

Integra reconocimiento de tres ejercicios de antebrazo y estados de fatiga en seis clases. Su procedimiento deriva primero un índice mediante tendencias RMS/MNF y después combina categorías de ejercicio y fatiga. Esa dirección difiere de nuestra cascada propuesta. La sección experimental describe una partición 60/30/10, sin acreditar allí una separación por personas equivalente a nuestro LOSO. La etiqueta EMG derivada no aporta una referencia fisiológica independiente. Es un antecedente relacionado, no prueba de que reconocer extensión primero mejore la detección de fatiga.

## 4. Bonato y colaboradores, 2001 — antecedente del proyecto

Time-Frequency Parameters of the Surface Myoelectric Signal for Assessing Muscle Fatigue during Cyclic Dynamic Contractions. IEEE Transactions on Biomedical Engineering, 48(7), 745–753. DOI: [10.1109/10.930899](https://doi.org/10.1109/10.930899). [Copia primaria del laboratorio](https://www.bu.edu/nmrc/files/2010/04/084.pdf).

Ya estaba incluido en la revisión principal. Fundamenta analizar fases equivalentes de ciclos para limitar variaciones mecánicas del espectro. El texto se consultó en la revisión previa; el enlace de descarga produjo timeout al reabrirlo durante esta búsqueda. No se usa como prueba de superioridad de nuestra arquitectura.

## Implicación para esta prueba

Existe fundamento para controlar fase y antecedentes de reconocimiento combinado. Los trabajos seleccionados no establecen que la cascada exacta propuesta mejore generalización en nuestro conjunto. El más cercano dispone de IMU y señala expresamente el problema de inferir fases cinemáticas sólo desde la envolvente. Esto respalda mantener “fase de ráfaga candidata” como nombre de nuestra aproximación y conservar como pendiente una referencia cinemática independiente.
