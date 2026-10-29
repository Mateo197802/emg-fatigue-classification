// Run a copy in a private build directory with node_modules linked to the
// bundled Codex Node packages. Inputs are finalized private report assets.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const workspaceDir=path.resolve(process.argv[2]);
const skill=process.env.EMG_PRESENTATIONS_SKILL;
const runtimePython=process.env.EMG_RUNTIME_PYTHON;
if(!skill||!runtimePython)throw new Error('Set EMG_PRESENTATIONS_SKILL and EMG_RUNTIME_PYTHON');
const data=JSON.parse(await fs.readFile(path.join(workspaceDir,'slides_input.json'),'utf8'));
const assets=path.join(workspaceDir,'report_assets');
const assetNames=['qc_examples.png','split_scheme.png','model_comparison.png','confusion.png','held_out_groups.png','sensitivity.png','ablation.png'];
if(data.include_phase)assetNames.push(...['primary_confusions_counts.png','primary_confusions_normalized.png','phase_confusions_counts.png','phase_confusions_normalized.png','auxiliary_phase_confusion_counts.png','auxiliary_phase_confusion_normalized.png'].map(n=>'phase/'+n));
const embeddedImages=new Map(await Promise.all(assetNames.map(async name=>[name,new Uint8Array(await fs.readFile(path.join(assets,name)))])));
const output=path.join(workspaceDir,process.argv[3]??'presentation');const build=path.join(workspaceDir,`deck_build_${path.basename(output)}`);
await fs.mkdir(build,{recursive:true});await fs.mkdir(output,{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const blue='#234f6b',ink='#182a36',muted='#4b5e6c',green='#39735d';
function text(slide,content,x,y,w,h,size=28,color=ink,bold=false){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=content;s.text.style={typeface:'Arial',fontSize:size,color,bold};return s;
}
function rect(slide,x,y,w,h,fill){return slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function slide(title,index,notes){
 const s=deck.slides.add();s.background.fill='#ffffff';rect(s,0,0,1280,14,blue);
 text(s,title,62,48,1156,112,39,ink,true);
 text(s,'sEMG  /  etiqueta temporal aproximada  /  análisis offline',62,672,1050,26,17,muted);
 text(s,String(index).padStart(2,'0'),1170,666,60,34,20,muted);
 s.speakerNotes.text=notes+'\nFuentes: '+data.repository+'; configuraciones y resultados en recibos privados del proyecto.';
 return s;
}
function image(s,name,x,y,w,h){s.images.add({blob:embeddedImages.get(name),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
const scripts=[
 ['Clasificación binaria con sEMG',35,'El objetivo del proyecto es distinguir fatiga de no fatiga a partir de la señal del recto femoral. Reconstruimos la implementación para que exista una ruta completa de procesamiento, entrenamiento, validación y predicción. La precisión de la conclusión depende de la etiqueta disponible. En este archivo no tenemos fuerza, Borg ni video sincronizado, por lo que evaluamos una aproximación temporal. Esta limitación acompaña los resultados desde el título y no aparece solamente al final.'],
 ['Lo que los datos permiten evaluar',50,'El archivo histórico conserva 140 grabaciones y más de cuatro millones de muestras. Dos son controles y 138 son formales. Las carpetas producen quince códigos de grupo; no tenemos una clave independiente que confirme la identidad de cada persona. Por eso retenemos grupos completos y explicitamos el supuesto de una persona por carpeta. El archivo RAR adicional se mantiene separado: dos códigos distintos no demuestran dos personas nuevas. Mostramos los denominadores para que las ventanas seleccionadas no se confundan con sujetos independientes.'],
 ['La etiqueta determina qué aprende el modelo',75,'Cada grabación conserva sus límites originales. Las ventanas completas del primer tercio reciben la clase temprana y las del último tercio la clase tardía. El tercio central y las ventanas que cruzan fronteras quedan fuera de la evaluación principal. Son etiquetas débiles, compatibles con la idea de progresión del esfuerzo, pero no demuestran que el inicio carezca de fatiga ni que el final tenga una pérdida de fuerza determinada. La actividad o calidad insuficiente produce abstención. Una caída espectral no se usa para crear una etiqueta y después alimentar al mismo clasificador con MDF.'],
 ['La calidad precede a la clasificación',65,'Los archivos se validan sin modificar los originales. Los huecos dividen tramos; nunca interpolamos a través de ellos. Rechazamos relojes incompatibles, variación temporal excesiva y segmentos constantes, y excluimos los bordes de los filtros. La figura muestra un segmento continuo y un intervalo que no puede cruzarse para formar una ventana. Un registro puede conservar segmentos útiles aunque tenga un hueco. La amplitud se expresa en cuentas: la columna de voltaje exportada no verifica calibración física. No fabricamos SNR de reposo ni porcentaje MVC.'],
 ['Tres familias describen cada ventana',55,'La rama espectral conserva una señal sin rectificación y aplica una banda de veinte a cuatrocientos hercios. Separamos la envolvente de actividad de los cálculos espectrales. RMS, MAV y longitud de la forma de onda describen amplitud y variación local. Esta última suma los cambios absolutos entre muestras por segundo, no mide una distancia física. MDF, MNF y razón de bandas describen la distribución del espectro. Las energías wavelet resumen varias escalas. Estas familias pueden responder a fatiga, pero también a fase del movimiento, esfuerzo y adquisición. Por eso su aporte predictivo no es una prueba de mecanismo fisiológico.'],
 ['Cada grupo queda fuera del entrenamiento',75,'En cada pliegue externo retenemos todas las ventanas de un grupo, incluidas ambas piernas y todas sus series. Dentro del entrenamiento hacemos otra validación agrupada para elegir parámetros y familias. El escalador se ajusta solamente con entrenamiento. La media de desempeño interno se calcula por grupo para no favorecer a quien aportó más ventanas. No elegimos el modelo usando el sujeto retenido. Las predicciones fuera de muestra quedan guardadas una vez por ventana y modelo, junto con particiones, parámetros y métricas.'],
 ['El modelo complejo no mejora claramente el baseline',75,'La comparación incluye la mayoría de entrenamiento y una logística con exactamente RMS y MDF, además de modelos más amplios. El número principal corresponde al procedimiento de selección anidado, no a la familia con el mejor valor observado después. El resultado no muestra una ventaja clara frente al baseline simple. Esto es un resultado del estudio y no un motivo para cambiar etiquetas o exclusiones. Reportamos media y dispersión entre grupos, junto con AUC y precisión-recall. Todas estas cifras describen la aproximación temporal.'],
 ['Los errores cambian entre grupos',60,'La matriz de confusión agrega predicciones fuera de muestra y muestra qué clase falla. El gráfico por grupo deja visible la variabilidad que se escondería al publicar un único porcentaje de acierto. La cantidad de ventanas no sustituye la cantidad de personas independientes. Conservamos los errores individuales para inspección privada; no atribuimos un error a mala colocación o fatiga real sin evidencia adicional. Los intervalos bootstrap reflejan variación en esta cohorte pequeña y no garantizan el resultado en cualquier población.'],
 ['Las decisiones metodológicas importan',65,'Las sensibilidades se definieron antes de examinarlas: ventanas más largas, ausencia de solapamiento, ocupación de actividad, desplazamiento de fronteras y notch condicionado a interferencia. Para estas comparaciones usamos una logística fija RMS y MDF, sin buscar el mejor escenario. Las ablaciones retiran una familia a la vez y repiten la selección interna. Cambiar la ventana también cambia qué datos sobreviven, por lo que mostramos cobertura junto al resultado. La conclusión debe resistir estas decisiones y no apoyarse en el escenario que más convenga.'],
 ['Qué queda entregado y qué sigue pendiente',45,'Entregamos código reproducible, entorno fijado, resultados por modelo y pliegue, modelo que se recarga, predicción con motivos de abstención y un informe trazable. Resolvimos la desconexión entre procesamiento y modelo y mantuvimos separadas las fuentes y tareas. Lo que sigue sin demostrarse es la validación fisiológica independiente, la identidad reconciliada entre archivos y los parámetros reales de adquisición que no se registraron. Recuperar esas referencias permitiría evaluar un constructo más fuerte. Mientras tanto, el sistema se presenta con el alcance exacto que sus datos permiten.']
];
if(data.include_phase)scripts[8]=['La fase candidata no mejora el resultado',65,'Probamos dos modelos conectados. El primer random forest aprende primera y segunda mitad de ráfagas EMG, no extensión y flexión verificadas. Su score se añade a una logística que conserva las diez características originales; la variante principal añade interacciones con RMS, MDF y MNF. Ambos modelos respetan las particiones por grupo y se repite cross-fitting dentro de la selección interna. El baseline directo obtuvo 0,6355 de macro-F1 y el condicionado 0,6293. El intervalo de la diferencia cruza cero. El control aleatorio obtuvo 0,6356. La fase no mejoró claramente ni justifica reemplazar el modelo principal. Cheah et al. 2022, DOI 10.54941/ahfe1002597, usa IMU; Potvin 1997, DOI 10.1152/jappl.1997.82.1.144, muestra efectos cinemáticos; Guzu et al. 2024, DOI 10.1109/ELMAR62909.2024.10694353, combina ejercicio y estados EMG con otro diseño.'];
const s1=slide(scripts[0][0],1,scripts[0][2]);text(s1,'Proxy temporal de fatiga del cuádriceps',65,210,1080,100,47,blue,true);text(s1,'Ventanas binarias, validación por grupos y resultados auditables',65,340,1050,100,31,muted);text(s1,'28 de octubre de 2026',65,535,900,50,25,muted);
const s2=slide(scripts[1][0],2,scripts[1][2]);
[['140','grabaciones fuente'],['138','grabaciones formales'],['15','grupos de grabación'],[String(data.labeled_windows),'ventanas con etiqueta débil']].forEach((x,i)=>{let xx=65+(i%2)*580,yy=185+Math.floor(i/2)*190;rect(s2,xx,yy,545,160,'#eef3f6');text(s2,x[0],xx+24,yy+16,490,68,57,blue,true);text(s2,x[1],xx+24,yy+98,490,45,25,muted);});
const s3=slide(scripts[2][0],3,scripts[2][2]);
[['Inicio','0  Proxy temprano',green],['Transición','Abstención','#7f8d95'],['Final','1  Proxy tardío',blue]].forEach((x,i)=>{let xx=65+i*385;rect(s3,xx,240,360,150,x[2]);text(s3,x[0],xx+24,255,310,50,34,'#ffffff',true);text(s3,x[1],xx+24,320,320,45,25,'#ffffff');});text(s3,'Una ventana no evaluable nunca se convierte en “no fatiga”.',65,460,1120,100,31,ink,true);text(s3,'Los tercios se calculan sobre los tiempos originales, antes de excluir señal.',65,560,1120,60,24,muted);
const s4=slide(scripts[3][0],4,scripts[3][2]);image(s4,'qc_examples.png',65,164,1150,450);
const s5=slide(scripts[4][0],5,scripts[4][2]);
[['Amplitud','RMS\nMAV\nCambios absolutos\npor segundo'],['Espectro','MDF y MNF\nRazón de potencia\nPSD Hann de 20 a 400 Hz'],['Tiempo y frecuencia','Wavelet db4\nNivel 3\nEnergías relativas']].forEach((x,i)=>{let xx=65+i*385;rect(s5,xx,205,360,300,'#eef3f6');text(s5,x[0],xx+20,225,320,70,30,blue,true);text(s5,x[1],xx+20,318,320,165,27,ink);});text(s5,'250 ms por ventana   50% de solapamiento   Actividad candidata ≥75%',65,557,1150,70,27,muted);
const s6=slide(scripts[5][0],6,scripts[5][2]);image(s6,'split_scheme.png',65,190,1150,380);text(s6,'Ambas piernas y todas las series de un grupo permanecen juntas.',65,570,1120,60,26,blue,true);
const s7=slide(scripts[6][0],7,scripts[6][2]);image(s7,'model_comparison.png',65,170,790,450);text(s7,'Selección anidada',890,220,320,70,30,blue,true);text(s7,data.selector_f1,890,310,320,90,42,ink,true);text(s7,'RMS + MDF',890,425,320,60,28,blue,true);text(s7,data.baseline_f1,890,490,320,70,35,ink);
const s8=slide(scripts[7][0],8,scripts[7][2]);image(s8,'confusion.png',45,190,720,390);image(s8,'held_out_groups.png',780,210,460,350);
const s9=slide(scripts[8][0],9,scripts[8][2]);
if(data.include_phase){
 [['Directo: 10 características',data.phase.baseline_f1],['Condicionado por fase',data.phase.conditioned_f1],['Control aleatorio',data.phase.control_f1]].forEach((v,i)=>{text(s9,v[0],65,190+i*115,590,46,27,blue,true);text(s9,v[1],65,235+i*115,590,48,32,ink);});
 text(s9,'Etapa 1  Random forest',740,195,470,55,29,blue,true);text(s9,'Primera / segunda mitad\nde ráfagas EMG',740,255,470,90,28,ink);
 text(s9,'Etapa 2  Logística',740,375,470,55,29,blue,true);text(s9,'EMG + score de fase\n→ clase temprana / tardía',740,435,470,90,28,ink);
 text(s9,data.phase.contrast,65,575,1150,56,24,ink,true);
}else{image(s9,'sensitivity.png',35,180,640,440);image(s9,'ablation.png',690,200,550,390);}
const s10=slide(scripts[9][0],10,scripts[9][2]);
rect(s10,65,190,550,390,'#eef5f1');rect(s10,655,190,560,390,'#eef3f6');text(s10,'Entregado',90,215,490,60,34,green,true);text(s10,'Código y entorno fijado\nResultados por modelo y pliegue\nModelo recargable\nPredicción con abstenciones',90,300,490,230,28,ink);text(s10,'Evidencia pendiente',680,215,510,60,34,blue,true);text(s10,'Referencia fisiológica independiente\nIdentidad entre fuentes\nCalibración y modo del canal\nAtestación de contribuciones',680,300,510,230,27,ink);

const reserveNotes=[];
if(data.include_phase){
 const a=slide('Reserva  Arquitectura real de las dos etapas',11,'La referencia auxiliar proviene de mitades de actividad, no de movimiento observado. Score no calibrado. El segundo modelo conserva las características EMG. Las figuras completas y parámetros están en el informe y docs/PHASE_EXPERIMENT.md.');
 function box(s,x,y,w,h,label,size=27){rect(s,x,y,w,h,'#eef3f6');text(s,label,x+18,y+18,w-36,h-36,size,ink,true);}
 box(a,65,230,300,170,'Diez características\nEMG');box(a,480,180,330,180,'Etapa 1\nRandom forest\nMitad de ráfaga',25);box(a,905,300,305,200,'Etapa 2\nLogística\nEMG + score',25);
 text(a,'→',380,250,80,70,46,blue,true);text(a,'→',825,295,80,70,46,blue,true);text(a,'EMG original → segunda etapa',65,475,790,60,27,blue,true);
 text(a,'Salida: proxy temprano / tardío. No hay extensión ni fatiga verificadas.',65,575,1150,65,25,ink);
 const b=slide('Reserva  Cómo se relacionan todos los modelos',12,'Seis procedimientos principales, cinco variantes suplementarias y una cabeza auxiliar. baseline_10 reproduce logistic_all. El selector elige dentro de entrenamiento y no es una votación. El RF final de 300 árboles se ajusta aparte con toda la cohorte y no agrega una estimación OOF.');
 [['Principales','Mayoría\nLogística RMS + MDF\nLogística de 10 features\nSVM RBF\nRandom forest\nSelector anidado'],['Suplementarios','Directo = logística de 10\nEMG + score de fase\nCondicionado por fase\nSólo score de fase\nControl aleatorio'],['Funciones distintas','RF auxiliar: mitad de ráfaga\nLogística: proxy temporal\nRF final: artefacto principal\n\nEl selector elige un modelo\nNo combina sus votos']].forEach((v,i)=>{let x=65+i*385;rect(b,x,190,360,425,'#eef3f6');text(b,v[0],x+18,210,325,60,28,blue,true);text(b,v[1],x+18,295,325,290,23,ink);});
 const figures=[['Reserva  Confusiones principales  conteos','primary_confusions_counts.png','Cada procedimiento: 5.509 ventanas; 0 temprano, 1 tardío.'],['Reserva  Confusiones principales  proporciones','primary_confusions_normalized.png','Normalización por fila de referencia; no es un intervalo de confianza.'],['Reserva  Confusiones suplementarias  conteos','phase_confusions_counts.png','Mismas ventanas y etiquetas que los procedimientos principales.'],['Reserva  Confusiones suplementarias  proporciones','phase_confusions_normalized.png','El directo reproduce logistic_all; no equivale al selector anidado.']];
 figures.forEach((v,i)=>{const s=slide(v[0],13+i,v[2]+' Fuente: resultados OOF verificados, results/phase_experiment/confusion_matrices.csv. Filas referencia, columnas predicción.');image(s,'phase/'+v[1],65,165,1150,440);text(s,v[2],65,610,1150,44,20,muted);reserveNotes.push([13+i,v[0],v[2]]);});
 const q=slide('Reserva  Qué logró la primera etapa auxiliar',17,'Referencia primera/segunda mitad EMG, no extensión/flexión. Soporte 1.663 y 1.674, total 3.337. Macro-F1 0,5420 ± 0,1028, AUC 0,6212 ± 0,1288. Las predicciones reconstruidas reprodujeron las métricas por grupo.');
 image(q,'phase/auxiliary_phase_confusion_counts.png',45,180,570,400);image(q,'phase/auxiliary_phase_confusion_normalized.png',660,180,570,400);text(q,'0 = primera mitad   1 = segunda mitad   No son etiquetas de extensión/flexión',65,595,1150,55,23,ink,true);
 const z=slide('Reserva  Sensibilidades y ablaciones originales',18,'Se conservan las nueve sensibilidades prespecificadas y tres ablaciones originales. No se ajustaron para mejorar el experimento de fase. Modelos, soportes y configuraciones están en el informe y los recibos privados.');image(z,'sensitivity.png',35,180,640,440);image(z,'ablation.png',690,200,550,390);
 const r=slide('Reserva  Referencias y comprobaciones',19,'Cheah, Wan y Yip 2022 DOI 10.54941/ahfe1002597: fase medida por IMU y etiqueta del último 20% de repeticiones. Potvin 1997 DOI 10.1152/jappl.1997.82.1.144: influencia de cinemática. Guzu et al. 2024 DOI 10.1109/ELMAR62909.2024.10694353: categorías de ejercicio y fatiga derivada de EMG. Fuentes primarias y límites en docs/PHASE_REFERENCES.md. Las pruebas de integridad no sustituyen validación fisiológica.');
 text(r,'Cheah et al. 2022   Fases mediante IMU\n10.54941/ahfe1002597',65,180,1140,90,26,ink);text(r,'Potvin 1997   Ángulo y velocidad afectan EMG\n10.1152/jappl.1997.82.1.144',65,295,1140,90,26,ink);text(r,'Guzu et al. 2024   Reconocimiento combinado\n10.1109/ELMAR62909.2024.10694353',65,410,1140,90,26,ink);text(r,'Pruebas: soporte OOF, grupos disjuntos, matrices, equivalencia y hashes.',65,555,1150,72,24,blue,true);
 reserveNotes.push([11,'Arquitectura real','La primera etapa no aprende extensión observada; la segunda conserva EMG.'],[12,'Relación de modelos','Ver registro de modelos y diferencias de objetivo.'],[17,'Cabeza auxiliar','Matrices referidas a mitades EMG; soporte 3.337.'],[18,'Sensibilidades y ablaciones','Evidencia original conservada.'],[19,'Referencias y comprobaciones','Fuentes primarias y sus límites; no equivalencia clínica.']);
}
await fs.writeFile(path.join(output,'Guion_defensa.md'),'# Defensa principal\n\nDiez diapositivas principales; tiempo previsto total: 600 s. Las diapositivas de reserva se usan sólo para preguntas.\n\n'+scripts.map((x,i)=>`## ${i+1}. ${x[0]}\n\nTiempo previsto: ${x[1]} s.\n\n${x[2]}`).join('\n\n')+'\n\n# Evidencia de reserva\n\n'+reserveNotes.sort((a,b)=>a[0]-b[0]).map(x=>`## ${x[0]}. ${x[1]}\n\n${x[2]}`).join('\n\n')+'\n');
const candidatePath=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=path.join(output,'EMG_Binary_Defense.pptx');
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:runtimePython,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 explicitTotalSlideCount:data.include_phase?19:10,fontPolicy:{basis:'design',families:['Arial']},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
for(let i=0;i<deck.slides.items.length;i++){
 const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(finalPath);
