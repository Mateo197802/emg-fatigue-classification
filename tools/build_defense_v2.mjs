// Reorganized defense from private frozen inputs; never trains models.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const workspaceDir=path.resolve(process.argv[2]);
const skill=process.env.EMG_PRESENTATIONS_SKILL,runtimePython=process.env.EMG_RUNTIME_PYTHON;
if(!skill||!runtimePython)throw new Error('Set EMG_PRESENTATIONS_SKILL and EMG_RUNTIME_PYTHON');
const d=JSON.parse(await fs.readFile(path.join(workspaceDir,'slides_input_v2.json'),'utf8'));
const output=path.join(workspaceDir,process.argv[3]??'presentation_reorganized');
const build=path.join(workspaceDir,`deck_build_${path.basename(output)}`);
await fs.mkdir(output,{recursive:true});await fs.mkdir(build,{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const C={blue:'#234f6b',ink:'#182a36',muted:'#526775',green:'#39735d',orange:'#a56c17',red:'#9e4141',pale:'#eef3f6',line:'#d4dfe5'};
const script=[],tableOwners=[],chartOwners=[];
function t(s,v,x,y,w,h,size=28,color=C.ink,bold=false){
 const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=String(v);z.text.style={typeface:'Arial',fontSize:size,color,bold};return z;
}
function r(s,x,y,w,h,fill=C.pale){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function slide(title,seconds,notes,source){
 const n=deck.slides.items.length+1,s=deck.slides.add();s.background.fill='#ffffff';r(s,0,0,1280,10,C.blue);
 t(s,title,56,37,1168,92,40,C.ink,true);t(s,n.toString().padStart(2,'0'),1180,675,50,26,17,C.muted);
 t(s,source,56,677,1090,25,15,C.muted);s.speakerNotes.text=notes+'\n\nFuente: '+source+'. '+d.repository;
 script.push({index:n,title,seconds,notes,source});return s;
}
function table(s,headers,rows,x=56,y=160,w=1168,h=370,widths=null,size=25){
 const v=[headers,...rows].map(row=>row.map(String));
 const z=s.tables.add({rows:v.length,columns:headers.length,left:x,top:y,width:w,height:h,values:v,...(widths?{columnWidths:widths}:{})});
 z.borders.assign({style:'solid',fill:C.line,width:1});
 for(let a=0;a<v.length;a++){
  z.rows[a].height=h/v.length;
  for(let b=0;b<headers.length;b++){
   const c=z.getCell(a,b);c.fill=a===0?C.blue:(a%2?'#ffffff':C.pale);
   c.text.style={typeface:'Arial',fontSize:a===0?size-1:size,color:a===0?'#ffffff':C.ink,bold:a===0};
  }
 }
 tableOwners.push(deck.slides.items.indexOf(s)+1);return z;
}
function chart(s,type,categories,series,x,y,w,h,extra={}){
 // Excel embeds at most 15 significant digits. Keep full inputs privately;
 // use 12 significant digits for visual series only, without changing results.
 series=series.map(v=>({...v,smooth:false,values:v.values.map(n=>n===null?null:Number(n.toPrecision(12)))}));
 const z=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,series,hasLegend:false,titlePlacement:'none',
  chartFill:'#ffffff',chartLine:{fill:'none',width:0},plotAreaFill:'#ffffff',lineOptions:{smooth:false},
  xAxis:{textStyle:{fontFamily:'Arial',fontSize:16,fill:C.muted},line:{fill:C.line,width:1}},
  yAxis:{textStyle:{fontFamily:'Arial',fontSize:16,fill:C.muted},majorGridlines:{fill:C.line,width:.7}},...extra});
 chartOwners.push(deck.slides.items.indexOf(s)+1);return z;
}
function block(s,x,y,w,h,title,body,color=C.blue){r(s,x,y,w,h);t(s,title,x+20,y+17,w-40,55,27,color,true);t(s,body,x+20,y+83,w-40,h-94,25);}
function processBox(s,x,y,w,h,title,body,fill='#eef3f6',titleSize=22,bodySize=18){
 r(s,x,y,w,h,fill);t(s,title,x+12,y+10,w-24,36,titleSize,C.ink,true);t(s,body,x+12,y+48,w-24,h-54,bodySize,C.ink);
}
function arrow(s,x,y){t(s,'→',x,y,60,55,40,C.blue,true);}
function cm(s,model,x,y,w=555,h=242){
 const q=d.confusions.find(q=>q.model===model);if(!q)throw new Error('Missing matrix '+model);
 table(s,['Real / predicha','Clase 0','Clase 1'],[['Clase 0',q.cm_0_0,q.cm_0_1],['Clase 1',q.cm_1_0,q.cm_1_1]],x,y,w,h,[w*.44,w*.28,w*.28],26);
}
function ms(q){return q.mean.toFixed(4)+' ± '+q.sd.toFixed(4);}
const imageNames=['sensitivity.png','ablation.png','signal_preprocessing_evidence.png',...['primary_confusions_counts.png','primary_confusions_normalized.png','phase_confusions_counts.png','phase_confusions_normalized.png'].map(n=>'phase/'+n)];
const images=new Map(await Promise.all(imageNames.map(async n=>[n,new Uint8Array(await fs.readFile(path.join(workspaceDir,'report_assets',n)))])));
function image(s,n,x,y,w,h){s.images.add({blob:images.get(n),contentType:'image/png',alt:n,fit:'contain',position:{left:x,top:y,width:w,height:h}});}

{
 const s=slide('Clasificación proxy temporal con sEMG',20,
 'Este estudio clasifica una etiqueta proxy temprana frente a tardía construida por posición dentro de cada grabación. No hay una medición independiente para afirmar fatiga muscular real. Presentaremos el corpus, la señal, el preanálisis, el procesamiento, los modelos y el experimento auxiliar, distinguiendo los resultados ejecutados de la arquitectura propuesta.',
 'Proyecto integrador de biomecánica');
 t(s,'Análisis de señal y\nmodelos de clasificación',56,176,660,145,51,C.blue,true);
 t(s,'Recto femoral durante ejercicio\nde extensión de rodilla',56,354,635,95,31,C.muted);
 t(s,'Ethan Guevara\nMateo Gavilanes\nKennedy Rodriguez',56,505,620,119,25);
 chart(s,'line',d.signal.time.map(v=>v.toFixed(2)),[{name:'sEMG filtrada',values:d.signal.filtered,line:{fill:C.blue,width:1.2},marker:{symbol:'none'}}],735,185,475,270,{xAxis:{visible:false},yAxis:{visible:false}});
 t(s,'Segmento real de sEMG',770,465,430,45,24,C.blue,true);t(s,'Etiqueta evaluada\nPrimer y último tercio temporal',770,535,430,80,24,C.muted);
}
{
 const c=d.corpus,s=slide('Corpus y análisis cuantitativo',35,
 `El archivo principal contiene 140 CSV y 4.256.257 muestras. Las 138 grabaciones R1 a R5 entran al procesamiento y dos R0 quedan como controles. Hay quince grupos derivados de carpetas. Ambas piernas y todas las series permanecen juntas al validar. El RAR se conserva separado y no agrega dos sujetos verificados. La duración mediana entre timestamps es ${c.duration_median_s.toFixed(1)} segundos y puede incluir huecos.`,
 'inventory.csv y coverage.csv');
 table(s,['Contenido','Cantidad','Uso'],[['Grabaciones fuente','140','ZIP principal'],['Grabaciones R1–R5','138','Análisis formal'],['Controles R0','2','Sin entrenamiento'],['Grupos de registro','15','Validación externa']],56,160,790,335,[350,150,290]);
 t(s,'4.256.257',890,183,335,72,48,C.blue,true);t(s,'muestras originales',890,267,335,50,25,C.muted);
 t(s,`${c.right_recordings} derecha\n${c.left_recordings} izquierda`,890,365,335,100,28,C.ink,true);
 t(s,`Duración mediana ${c.duration_median_s.toFixed(1)} s por grabación formal`,56,535,1168,48,29,C.ink,true);
 t(s,'Grupos derivados de carpetas. La identidad individual no tiene una clave independiente.',56,602,1168,44,22,C.muted);
}
{
 const s=slide('Señal registrada y diagnóstico temporal',35,
 'Los CSV aportan tiempo relativo y cuentas ADC con aproximadamente mil muestras por segundo. Comparamos el mismo intervalo almacenado y filtrado. Elegimos el primer registro con un tramo válido sin consultar etiquetas ni rendimiento. Dibujamos una de cada cuatro muestras, pero el análisis conserva todas. Revisamos huecos, jitter, segmentos constantes y repetición de extremos. Las cuentas no acreditan una amplitud muscular calibrada en milivoltios.',
 'Archivo original, qc.py y preprocessing.py');
 const cat=d.signal.time.map(v=>v.toFixed(2));
 t(s,'Registro almacenado',56,149,800,44,26,C.blue,true);
 chart(s,'line',cat,[{name:'Cuentas ADC',values:d.signal.stored,line:{fill:C.muted,width:1.1},marker:{symbol:'none'}}],56,195,790,180,{xAxis:{visible:false}});
 t(s,'Mismo intervalo tras filtrar 20–400 Hz',56,389,820,44,26,C.blue,true);
 chart(s,'line',cat,[{name:'Cuentas filtradas',values:d.signal.filtered,line:{fill:C.blue,width:1.1},marker:{symbol:'none'}}],56,434,790,176,{xAxis:{visible:false}});
 t(s,'≈1.000 Hz',902,176,320,60,39,C.blue,true);t(s,'Tiempo relativo\nCuentas ADC\nAmbas piernas',902,260,320,150,28);
 t(s,'Control previo',902,442,320,45,28,C.blue,true);t(s,'Huecos y jitter\nSeñal constante\nValores extremos',902,500,320,125,25);
 t(s,`Tiempo relativo ${d.signal.time[0].toFixed(2)}–${d.signal.time.at(-1).toFixed(2)} s. Amplitud en cuentas.`,56,625,790,29,20,C.muted);
}
{
 const s=slide('Preanálisis descriptivo de la señal',35,
 'Estas curvas usan las 9.289 ventanas activas con características válidas, incluido el centro sin etiqueta. Primero calculamos medianas por registro y decil temporal, después medianas por grupo y finalmente media entre grupos. Los valores describen el corpus y no prueban fatiga. Amplitud y frecuencia también dependen de esfuerzo, movimiento y adquisición. Las curvas no participan en la creación de etiquetas.',
 'feature_trajectories_by_group.csv y features.parquet');
 [['RMS','rms','Cuentas'],['MDF','mdf_hz','Hz'],['MNF','mnf_hz','Hz']].forEach((v,i)=>{
  const x=56+i*398;t(s,v[0],x,165,360,50,31,C.blue,true);
  chart(s,'line',['0,05','0,15','0,25','0,35','0,45','0,55','0,65','0,75','0,85','0,95'],[{name:v[0],values:d.trajectories[v[1]],line:{fill:C.blue,width:3},marker:{symbol:'circle',size:5,fill:C.blue}}],x,227,370,245,{xAxis:{visible:false}});
  t(s,'0,05',x+30,478,70,27,17,C.muted);t(s,'0,95',x+303,478,65,27,17,C.muted);
  t(s,v[2],x+12,505,330,38,21,C.muted);
 });
 t(s,'Posición relativa dentro de cada registro',56,553,1168,44,27,C.ink,true);
 t(s,'9.289 ventanas activas. Resumen descriptivo con el mismo peso para cada grupo.',56,613,1168,38,22,C.muted);
}
{
 const s=slide('Preprocesamiento de la señal',45,
 'Separamos tramos cuando un hueco supera el máximo de 1,5 milisegundos y 1,5 veces el intervalo mediano. Rechazamos jitter relativo mayor al cinco por ciento y segmentos constantes. Usamos Butterworth de banda veinte a cuatrocientos hercios, prototipo orden cuatro y banda digital orden ocho, bidireccional sin desfase. Excluimos cien milisegundos por extremo. La rama espectral conserva señal sin rectificar. El notch queda desactivado en el principal y se evalúa como sensibilidad. Es procesamiento offline.',
 'configs/study.yaml, qc.py y preprocessing.py');
 table(s,['Paso','Método aplicado','Parámetro'],[
  ['Diagnóstico temporal','Separar huecos y rechazar jitter','Umbral 1,5 ms y jitter >5%'],
  ['Señal continua','Centrar y filtrar por tramo','Butterworth 20–400 Hz'],
  ['Filtro offline','Aplicación bidireccional','Prototipo de orden 4'],
  ['Bordes','Excluir transitorios','100 ms por extremo'],
  ['Actividad candidata','RMS local y umbral Otsu','50 ms, avance 25 ms'],
  ['Ventanas','Segmentos completos','250 ms, solapamiento 50%']],56,158,1168,390,[290,488,390],24);
 t(s,'Sin interpolar huecos. Rama espectral sin rectificar. Notch desactivado.',56,586,1168,65,25,C.blue,true);
}
{
 const s=slide('Ventanas y etiqueta temporal proxy',45,
 'Los tercios usan los límites temporales originales antes de filtrar y excluir. Ventanas completas del primer tercio reciben clase cero y del último clase uno. Centro y fronteras quedan fuera. Exigimos al menos setenta y cinco por ciento de actividad candidata. De 33.090 ventanas candidatas quedan 32.474 con calidad válida, 9.289 activas con características válidas y 5.509 tempranas o tardías. Una ventana descartada no se convierte en no fatiga. La ventana es la unidad de observación; la clase aprendida es una posición temporal, no fatiga muscular medida.',
 'label_audit.json y coverage.csv');
 block(s,56,161,366,175,'Proxy temprano','Clase 0\n2.329 ventanas',C.green);
 block(s,458,161,366,175,'Tercio central','Fuera del contraste\nSin etiqueta binaria');
 block(s,860,161,364,175,'Proxy tardío','Clase 1\n3.180 ventanas');
 table(s,['Etapa','Ventanas'],[['Candidatas','33.090'],['Con calidad válida','32.474'],['Activas con features válidas','9.289'],['Tempranas o tardías','5.509']],56,378,710,245,[510,200],24);
 t(s,'250 ms',848,391,370,60,42,C.blue,true);t(s,'50% de solapamiento\nActividad candidata ≥75%',848,475,370,100,26);
}
{
 const s=slide('Diez características por ventana',35,
 'RMS, MAV y longitud de la forma de onda por segundo resumen amplitud y variación. MDF, MNF y razón de potencia veinte a sesenta frente a sesenta a cuatrocientos hercios resumen el espectro. Cuatro energías db4 describen A3, D3, D2 y D1. El escalador se ajusta sólo con entrenamiento. No usamos tiempo, identidad, lado o serie como predictores ni inventamos normalización MVC. La segunda etapa y el comparador directo conservan las mismas diez características.',
 'features.py y feature_schema.json');
 table(s,['Familia','Características','Cantidad'],[
  ['Amplitud','RMS, MAV, longitud de forma de onda por segundo','3'],
  ['Espectro','MDF, MNF, razón 20–60 / 60–400 Hz','3'],
  ['Tiempo y frecuencia','Energías relativas db4: A3, D3, D2, D1','4']],56,175,1168,290,[270,738,160]);
 t(s,'Escalado aprendido dentro de cada entrenamiento',56,516,1168,48,30,C.blue,true);
 t(s,'Tiempo, identidad, pierna y serie permanecen fuera del vector de entrada.',56,586,1168,65,25,C.muted);
}
{
 const s=slide('Metodología de validación por grupos',45,
 'Cada ronda externa retiene uno de los quince grupos completos. Ambas piernas y todas sus series quedan fuera del ajuste. Con los otros catorce grupos hacemos cuatro particiones internas para elegir parámetros según macro-F1 medio por grupo. Escalador y modelo se ajustan exclusivamente en entrenamiento. Guardamos predicciones OOF una vez por ventana y procedimiento. La incertidumbre usa bootstrap de grupos. No dividimos ventanas al azar entre entrenamiento y prueba.',
 'evaluation.py, splits.json y evaluation_receipt.json');
 block(s,56,180,335,210,'14 grupos','Entrenamiento externo\nAmbas piernas y series');arrow(s,404,248);
 block(s,474,180,335,210,'4 particiones internas','Escalador y búsqueda\nMacro-F1 por grupo');arrow(s,824,248);
 block(s,896,180,328,210,'1 grupo retenido','Predicción OOF\nSin ajustar parámetros');
 t(s,'15 rondas externas',56,453,1168,58,39,C.blue,true);
 t(s,'Cada ventana se evalúa una vez por procedimiento. La incertidumbre se calcula por grupo.',56,535,1168,95,28);
}
{
 const s=slide('Modelos y selección de parámetros',35,
 'La mayoría es el mínimo trivial. La logística RMS y MDF es la referencia simple. La logística completa evalúa ampliar variables. SVM y RF permiten relaciones no lineales. El selector elige familia y parámetros sólo con validación interna, sin combinar votos. Se exploran diecinueve configuraciones. El artefacto final es RF de trescientos árboles, profundidad tres y hoja mínima dos, ajustado con la cohorte completa después de la evaluación.',
 'evaluation.py, inner_search.csv y model.receipt.json');
 table(s,['Procedimiento','Entrada','Búsqueda interna'],[
  ['Mayoría','Prior del entrenamiento','Sin ajuste'],['Logística simple','RMS y MDF','C = 0,1 / 1 / 10'],
  ['Logística completa','10 características','C = 0,1 / 1 / 10'],['SVM RBF','10 características','C y gamma'],
  ['Random forest','10 características','Profundidad y mínimo por hoja'],['Selector anidado','Familia y configuración','Elección con grupos internos']],56,155,1168,410,[335,325,508],24);
 t(s,'El selector elige un modelo. El ajuste final no agrega una estimación OOF.',56,596,1168,56,25,C.blue,true);
}
{
 const s=slide('Rendimiento de los modelos para el proxy temporal',45,
 'Todos usan las mismas 5.509 ventanas con objetivo temprano/tardío, no fatiga/no fatiga. Macro-F1 muestra media y desviación entre quince grupos. AUC es también media por grupo. El selector alcanza 0,6240 frente a 0,6272 del modelo simple. La diferencia es menos 0,0032, con intervalo de menos 0,0591 a más 0,0509. No aparece ventaja clara del procedimiento complejo. Estas métricas no estiman Accuracy o F1 de fatiga fisiológica.',
 'evaluation/metrics_summary.csv y paired_differences.csv');
 const n=[['Mayoría','majority'],['Logística RMS + MDF','logistic_rms_mdf'],['Logística 10','logistic_all'],['SVM RBF','svm_rbf'],['Random forest','random_forest'],['Selector anidado','nested_selector']];
 table(s,['Procedimiento','Macro-F1 media ± DE','AUC media'],n.map(([label,id])=>[label,ms(d.primary[id].f1_macro),d.primary[id].roc_auc.mean.toFixed(4)]),56,157,1168,386,[440,438,290],26);
 t(s,'Selector frente a RMS + MDF: Δ −0,0032',56,572,1168,42,28,C.blue,true);
 t(s,'IC 95% [−0,0591, +0,0509]. La complejidad no mostró una mejora clara.',56,620,1168,38,23,C.muted);
}
{
 const s=slide('Errores del procedimiento principal',35,
 'El selector acierta 1.473 de 2.329 ventanas tempranas y 2.185 de 3.180 tardías. Confunde 856 tempranas y 995 tardías. Macro-F1 varía aproximadamente entre 0,372 y 0,806 según el grupo. La matriz agrupa ventanas y no sustituye la media y dispersión entre grupos. No atribuimos errores a una causa fisiológica individual sin referencia independiente.',
 'evaluation/oof_predictions.parquet y evaluation/results.csv');
 t(s,'Matriz OOF del selector',56,160,570,50,29,C.blue,true);cm(s,'nested_selector',56,235,555,270);
 t(s,'Clase 0: primer tercio\nClase 1: último tercio',56,542,555,83,25,C.muted);
 t(s,'Macro-F1 por grupo retenido',671,160,553,50,29,C.blue,true);
 chart(s,'scatter',d.selector_groups.map((_,i)=>String(i+1)),[{name:'Macro-F1',xValues:d.selector_groups.map((_,i)=>i+1),values:d.selector_groups,line:{fill:'none',width:0},marker:{symbol:'circle',size:8,fill:C.blue}}],660,235,564,330,{scatterOptions:{style:'marker'},xAxis:{min:0,max:16,majorUnit:2,textStyle:{fontFamily:'Arial',fontSize:18,fill:C.muted}},yAxis:{min:0,max:1,textStyle:{fontFamily:'Arial',fontSize:18,fill:C.muted}}});
 t(s,'15 grupos externos con el mismo peso en la media',671,584,550,64,22,C.muted);
}
{
 const s=slide('Auxiliar EMG: dos etapas sobre una etiqueta proxy',45,
 'La hipótesis es que una representación local de fase ayude a interpretar cambios de señal. El RF auxiliar usa las diez características para distinguir primera y segunda mitad de ráfagas de medio a ocho segundos. Tiene cien árboles, profundidad tres y hoja mínima cinco. Su score no calibrado entra a una logística junto con las diez variables originales. La variante principal agrega score por RMS, MDF y MNF. Los scores del entrenamiento son cross-fitted por grupo, también dentro de selección interna. La fase candidata no dispone de extensión y flexión observadas.',
 'phase_spike.py y protocolo anterior a resultados');
 block(s,56,185,300,160,'Entrada','10 características EMG');
 block(s,480,150,330,205,'Etapa 1','RF de 100 árboles\nMitad de ráfaga EMG\nProfundidad 3, hoja 5');
 block(s,920,185,304,240,'Etapa 2','Regresión logística\nEMG + score\n+ interacciones\nRMS, MDF y MNF');
 arrow(s,385,226);arrow(s,835,226);t(s,'Score de fase',502,380,310,40,25,C.blue,true);
 t(s,'Las 10 características originales también entran directamente a la etapa 2.',56,477,1168,60,28,C.ink,true);
 t(s,'Scores fuera de muestra por grupo durante el entrenamiento. Salida: clase 0 / clase 1.',56,560,1168,73,25,C.muted);
}
{
 const s=slide('Diseño del experimento auxiliar',35,
 'Las cinco variantes comparten ventanas, etiquetas, escalado y particiones. El directo reproduce la logística completa. EMG más score mide información adicional, y condicionado agrega interacciones. Sólo score mide lo que transmite la primera etapa sin EMG original. El control aleatorio repite el procedimiento con etiquetas auxiliares aleatorias para examinar capacidad extra. Hay 3.337 etiquetas auxiliares de 996 ráfagas. Sólo grupos de entrenamiento aportan esta supervisión.',
 'model_registry.csv y protocolo original del experimento');
 table(s,['Variante','Entrada de la logística'],[
  ['Directo de 10 características','EMG original'],['EMG + score','10 características y score de fase'],
  ['Condicionado por fase','Lo anterior y score × RMS / MDF / MNF'],['Sólo score','Score de la primera etapa'],
  ['Control aleatorio','Misma capacidad, etiquetas de fase aleatorias']],56,155,1168,355,[425,743]);
 t(s,'5.509 ventanas para el objetivo temprano/tardío',56,551,1168,48,31,C.blue,true);
 t(s,'Referencia auxiliar: 3.337 ventanas de 996 ráfagas EMG candidatas.',56,615,1168,39,23,C.muted);
}
{
 const s=slide('Rendimiento auxiliar sobre el mismo proxy temporal',40,
 'El directo obtuvo 0,6355 de macro-F1, agregar score 0,6326, condicionar 0,6293, sólo score 0,6184 y el aleatorio 0,6356. El contraste principal condicionado menos directo es menos 0,0062 y su intervalo de menos 0,0232 a más 0,0082 cruza cero. Seis grupos mejoran, ocho empeoran y uno empata. Esta prueba no usó extensión/flexión observadas ni etiqueta fisiológica de fatiga; solo indica que la fase candidata no mejoró la clasificación temporal en este experimento.',
 'phase_experiment/metrics_summary.csv y paired_differences.csv');
 const n=[['Directo 10','baseline_10'],['EMG + score','phase_score'],['Condicionado por fase','phase_conditioned'],['Sólo score','phase_only'],['Control aleatorio','random_conditioned']];
 table(s,['Variante','Macro-F1 media ± DE','AUC media'],n.map(([name,id])=>[name,ms(d.phase[id].f1_macro),d.phase[id].roc_auc.mean.toFixed(4)]),56,155,1168,342,[440,438,290],26);
 t(s,'Δ condicionado − directo = −0,0062',56,535,1168,52,33,C.blue,true);
 t(s,'IC 95% [−0,0232, +0,0082]     6 mejoran, 8 empeoran y 1 empata',56,603,1168,51,24);
}
{
 const s=slide('Resultados de cada etapa de la cascada',35,
 'Las matrices tienen objetivos y soportes distintos. La auxiliar usa 1.663 primeras mitades y 1.674 segundas, con macro-F1 0,5420 y AUC 0,6212. La logística condicionada usa las 5.509 ventanas tempranas y tardías. Acierta 1.538 tempranas y 2.141 tardías. No se comparan los totales ni se interpreta la primera matriz como extensión y flexión observadas. Tampoco es una matriz de fatiga/no fatiga.',
 'confusion_matrices.csv y auxiliary_phase_fold_metrics.csv');
 t(s,'Etapa 1: mitad de ráfaga EMG',56,160,555,70,29,C.blue,true);cm(s,'auxiliary_phase_head',56,254);
 t(s,'0 = primera mitad, 1 = segunda mitad\nMacro-F1 0,5420 ± 0,1028\nAUC 0,6212 ± 0,1288',56,535,555,110,24);
 t(s,'Etapa 2: clase temporal',671,160,555,70,29,C.blue,true);cm(s,'phase_conditioned',671,254);
 t(s,'0 = temprano, 1 = tardío\nMacro-F1 0,6293 ± 0,1330\nAUC 0,8431 ± 0,0741',671,535,555,110,24);
}
{
 const s=slide('Conclusiones',35,
 'El experimento ejecutado predice una etiqueta temporal proxy, no fatiga muscular/no fatiga. El selector no mejora claramente a RMS y MDF; la representación de mitades de ráfaga tampoco mejora la clasificación proxy directa. Por eso no existe una Accuracy o macro-F1 válida de fatiga fisiológica para optimizar con este ZIP. La siguiente evaluación requiere torque o fuerza calibrados y sincronizados; la etapa de fase requiere cinemática sincronizada. Las arquitecturas directas, en cascada y de expertos quedan propuestas hasta disponer de esas referencias.',
 'Evaluación principal y experimento auxiliar');
 [['Resultado medido','Clasificación proxy temprano/tardío'],['Comparación principal','Selector 0,6240; RMS + MDF 0,6272 macro-F1'],['Experimento auxiliar','Condicionado 0,6293; directo 0,6355 macro-F1']].forEach((v,i)=>{
  const y=169+i*126;t(s,v[0],56,y,340,52,29,C.blue,true);t(s,v[1],420,y,804,89,30);
 });
 t(s,'No se entrenó ni evaluó fatiga/no fatiga fisiológica con este corpus.',56,566,1168,82,30,C.blue,true);
}
{
 const s=slide('Anexo: Relación entre todos los modelos',0,
 'Son seis procedimientos principales, cinco variantes y una cabeza auxiliar. Directo reproduce logistic_all. El selector elige una configuración, sin votos. El RF final de trescientos árboles se ajusta con toda la cohorte y no añade una estimación OOF.',
 'results/phase_experiment/model_registry.csv');
 table(s,['Conjunto','Procedimientos','Relación'],[
  ['Evaluación principal','Mayoría, 2 logísticas, SVM, RF, selector','Mismo objetivo y ventanas'],
  ['Experimento auxiliar','Directo, score, condicionado, sólo score, aleatorio','Directo = logística completa'],
  ['RF auxiliar de 100 árboles','Primera etapa de la cascada','Aprende mitades de actividad'],
  ['RF final de 300 árboles','Artefacto principal serializado','Ajuste posterior a la evaluación']],56,170,1168,360,[310,490,368],24);
 t(s,'Los roles de entrenamiento no equivalen a experimentos independientes.',56,582,1168,66,25,C.blue,true);
}
{
 const s=slide('Anexo: Sensibilidades y ablaciones',0,
 'Nueve sensibilidades prespecificadas y tres ablaciones. Las primeras usan logística fija RMS y MDF, con soportes distintos según exclusiones. Las ablaciones usan selección interna. Se conservan los resultados originales sin seleccionar el escenario más favorable.',
 'sensitivity_summary.csv y ablation_summary.csv');
 image(s,'sensitivity.png',35,165,650,440);image(s,'ablation.png',700,190,545,355);
 t(s,'El soporte cambia entre sensibilidades. Los resultados principales permanecen fijos.',56,616,1168,40,22,C.muted);
}
for(const [title,name,caption] of [
 ['Anexo: Matrices de los modelos principales','primary_confusions_counts.png','Conteos OOF. Cada procedimiento usa 5.509 ventanas.'],
 ['Anexo: Confusiones principales normalizadas','primary_confusions_normalized.png','Proporciones por fila. Clase 0 temprana, clase 1 tardía.'],
 ['Anexo: Matrices de las variantes auxiliares','phase_confusions_counts.png','Mismo soporte de 5.509 ventanas. Directo reproduce logística 10.'],
 ['Anexo: Confusiones auxiliares normalizadas','phase_confusions_normalized.png','Proporciones por fila. Matrices de la segunda etapa.']]){
 const s=slide(title,0,caption+' La incertidumbre principal se calcula entre grupos.',
 'results/phase_experiment/confusion_matrices.csv');
 image(s,'phase/'+name,56,145,1168,465);t(s,caption,56,622,1168,35,22,C.muted);
}
{
 const s=slide('Anexo: Bibliografía del experimento auxiliar',0,
 'Cheah et al. usan fase medida con IMU y una referencia temporal al final de repeticiones. Potvin estudia efectos cinemáticos sobre EMG. Guzu et al. combinan ejercicio y estados EMG con otro diseño. Motivan estudiar fase, pero no validan nuestras etiquetas ni una mejora de esta cascada.',
 'docs/PHASE_REFERENCES.md');
 table(s,['Referencia','Aporte para la hipótesis'],[
  ['Cheah et al. 2022\n10.54941/ahfe1002597','Fase dinámica medida con IMU'],
  ['Potvin 1997\n10.1152/jappl.1997.82.1.144','Influencia de ángulo y velocidad en EMG'],
  ['Guzu et al. 2024\n10.1109/ELMAR62909.2024.10694353','Reconocimiento de ejercicio y estados EMG']],56,165,1168,360,[610,558]);
 t(s,'La literatura motiva condicionar por movimiento. Esta prueba usa una fase candidata de EMG.',56,581,1168,72,25,C.muted);
}
{
 const s=slide('Anexo: Ventanas y solapamiento',0,
 'El protocolo histórico fijado usa ventanas deslizantes de 250 milisegundos con 50 por ciento de solapamiento. Para un estudio futuro con referencias independientes se propone 500 milisegundos como análisis principal y 250 milisegundos como sensibilidad separada. La relación general es N igual a redondear T por fs, H igual a redondear N por uno menos solapamiento y la separación nominal de bins es fs dividido entre N. A un kilohercio son 500 muestras y un avance de 250 para el análisis principal, con separación nominal de dos hercios. Para la sensibilidad y el histórico son 250 muestras y avance de 125, cuatro hercios. La ventana más larga mejora la resolución nominal; el solapamiento da mayor densidad temporal, aunque crea ventanas dependientes. Esta ilustración no representa una etiqueta.',
 'Configuración histórica congelada y configs/reference_study.yaml');
 const x0=330,scale=.82;
 const rows=[
   {y:165,label:'Proxy histórico ejecutado',length:250,hop:125,fill:'#f9dede',param:'250 ms · 50% solape'},
   {y:305,label:'Fatiga con referencias · primario propuesto',length:500,hop:250,fill:'#e5f2e8',param:'500 ms · 50% solape'},
   {y:445,label:'Fatiga con referencias · sensibilidad',length:250,hop:125,fill:'#edf0f4',param:'250 ms · 50% solape'},
 ];
 for(const row of rows){
  t(s,row.label,56,row.y-8,260,52,22,C.ink,true);t(s,row.param,850,row.y-7,360,45,20,C.muted,true);
  for(let start=0;start+row.length<=1000;start+=row.hop){
   const sh=s.shapes.add({geometry:'rect',position:{left:x0+start*scale,top:row.y+44,width:row.length*scale,height:48},fill:row.fill,line:{fill:C.blue,width:1.5}});
  }
 }
  r(s,x0,545,820,2,C.muted);for(let i=0;i<=4;i++){const xx=x0+i*205;r(s,xx,538,2,16,C.muted);t(s,String(i*250),xx-23,558,55,24,15,C.muted);}
  t(s,'Tiempo (ms)',675,584,160,24,16,C.muted,true);
  t(s,'N = round(T × fs)     H = round(N × (1 − ρ))     Δf = fs/N',56,612,1168,29,20,C.blue,true);
  t(s,'A 1 kHz: 500 ms → N=500, H=250, Δf≈2 Hz; 250 ms → N=250, H=125, Δf≈4 Hz.',56,645,1168,26,16,C.muted);
}
{
 const s=slide('Anexo: Filtros y ecuaciones de ventana',0,
 'El pasa banda digital se diseña por separado para cada frecuencia de muestreo verificada. El prototipo Butterworth de orden cuatro se transforma en un pasa banda digital de orden ocho y se implementa en secciones de segundo orden. La forma recursiva muestra los coeficientes y el filtrado hacia adelante y atrás. Notch es solo una sensibilidad activada si el diagnóstico PSD muestra interferencia de red prominente; no forma parte del procesamiento principal. Las fórmulas del periodograma corresponden a la ventana Hann y al estimador unilateral de densidad que usa scipy.signal.periodogram, no a Welch. Las ventanas se extraen sin cruzar huecos o transiciones de referencia.',
 'SciPy butter, sosfiltfilt, periodogram, ventanas Hann e iirnotch: https://docs.scipy.org/doc/scipy/reference/signal.html');
 table(s,['Etapa','Fórmula o definición','Parámetro y motivo'],[
  ['Butterworth análogo','|Hₐ(jΩ)|² = 1 / [1 + (Ω/Ωc)^(2n)]','Prototipo pasa bajos; n=4 antes de transformar a pasa banda digital.'],
  ['Pasa banda digital SOS','H(z)=∏ᵣ(b₀,ᵣ+b₁,ᵣz⁻¹+b₂,ᵣz⁻²)/(a₀,ᵣ+a₁,ᵣz⁻¹+a₂,ᵣz⁻²)','20–400 Hz; orden digital 8; coeficientes dependientes de fs; SOS por estabilidad numérica.'],
  ['Sección recursiva','y[n]=(b₀x[n]+b₁x[n−1]+b₂x[n−2]−a₁y[n−1]−a₂y[n−2])/a₀','sosfiltfilt hacia adelante/atrás, offline; enmascarar 100 ms por extremo.'],
  ['Notch opcional','ω₀=2πf₀/fs; α=sin(ω₀)/(2Q); b=[1,−2cosω₀,1]/(1+α); a=[1,−2cosω₀/(1+α),(1−α)/(1+α)]','f₀=50/60 Hz, Q=30; solo si prominencia de red >10 dB. No aplicado en principal.'],
  ['Ventana deslizante','N=round(T·fs); H=round[N(1−ρ)]; xₖ[n]=x[kH+n]','ρ=50%; histórico T=250 ms; referencia propuesta T=500 ms y 250 ms de sensibilidad.'],
  ['Hann y periodograma','w[n]=0.5(1−cos(2πn/(N−1))); Pxx[k]=cₖ|Σₙx[n]w[n]e^(−j2πkn/N)|²/(fs·Σₙw[n]²)','cₖ=2 en bins interiores positivos; cₖ=1 en DC/Nyquist. Unilateral de densidad, detrend=False.'],
 ],56,128,1168,445,[205,555,408],16);
 t(s,'Hann reduce fuga espectral a cambio de un lóbulo principal más ancho. Δf=fs/N es separación de bins, no resolución garantizada.',56,610,1168,30,16,C.muted);
}
{
 const s=slide('Anexo: Características y fórmulas',0,
 'El vector X contiene diez variables calculadas por ventana. RMS y MAV resumen amplitud; longitud de onda resume cambio local; MDF, MNF y la razón de bandas resumen distribución espectral; cuatro coeficientes db4 dan energía relativa por escala. Todos los valores se calculan sobre la señal centrada, filtrada y expresada en cuentas del convertidor. Las amplitudes no son milivoltios calibrados. El análisis histórico usa longitud de onda dividida por N sobre fs; la versión corregida usa el lapso entre primera y última muestra. Esta segunda versión tiene nombre de esquema separado y no cambia resultados congelados.',
 'features.py, feature_schema.json y docs/reference_fatigue_protocol.md');
 table(s,['Feature','Fórmula','Unidades / condición'],[
  ['RMS','√[(1/N)Σₙx[n]²]','Cuentas ADC; amplitud eficaz.'],
  ['MAV','(1/N)Σₙ|x[n]|','Cuentas ADC; amplitud absoluta media.'],
  ['WL/s histórico v1','(fs/N)Σₙ₌₁ᴺ⁻¹|x[n]−x[n−1]|','Cuentas ADC/s; divisor histórico N/fs.'],
  ['WL/s corregido v2','Σₙ₌₁ᴺ⁻¹|x[n]−x[n−1]| / ((N−1)/fs)','Cuentas ADC/s; versión separada, no usada en la salida histórica.'],
  ['MDF','mín fₘ tal que Σ_{fₖ≤fₘ}Pxx[k] ≥ 0.5ΣₖPxx[k]','Hz; dentro de 20–400 Hz.'],
  ['MNF','Σₖ f[k]Pxx[k] / ΣₖPxx[k]','Hz; rechazar si la potencia total de banda es cero.'],
  ['Razón baja/alta','Σ_{20≤fₖ<60}Pxx[k] / Σ_{60≤fₖ≤400}Pxx[k]','Adimensional; rechazar denominador nulo.'],
  ['Energías db4 A3,D3,D2,D1','Eⱼ=Σₘcⱼ[m]² / Σ_{q∈{A3,D3,D2,D1}}Σₘc_q[m]²','Adimensional; DWT nivel 3, extensión simétrica; suma de Eⱼ=1.'],
 ],56,145,1168,490,[205,565,398],17);
}
{
 const s=slide('Anexo: Arquitectura con referencias independientes',0,
 'La cinemática sincronizada crea las etiquetas de extensión y flexión; torque o fuerza calibrados y sincronizados crean la etiqueta de fatiga aprobada. La misma matriz de características X entrena un modelo directo, una cascada de fase y fatiga, y expertos con compuerta suave. En la cascada, la etapa 2 solo recibe probabilidades de fase fuera de pliegue en entrenamiento y fase predicha en prueba. La salida de todos se evalúa en los mismos participantes retenidos. Sin las referencias independientes de fuerza, torque, cinemática e identidad que faltan en el ZIP, el diagrama es una propuesta sin entrenamiento ni métricas.',
 'docs/reference_fatigue_protocol.md; las referencias técnicas se citan en docs/PHASE_REFERENCES.md');
 processBox(s,56,143,280,84,'Ángulo / video','Sincronizado y calibrado','#e5f2e8',20,17);
 processBox(s,390,143,270,84,'y_fase','Extensión / flexión','#e5f2e8',20,17);
 processBox(s,750,143,270,84,'Fuerza / torque','Calibrado y sincronizado','#e5f2e8',20,17);
 processBox(s,1060,143,164,84,'y_fatiga','Definición aprobada','#e5f2e8',20,16);
 arrow(s,344,158);arrow(s,1025,158);
 processBox(s,56,290,260,98,'X por ventana','Las mismas 10 features','#dceaf7',21,18);
 processBox(s,390,290,280,98,'Modelo de fase','X → q_ext, q_flex','#fff0d8',21,19);
 processBox(s,750,290,300,98,'Probabilidades OOF','Para etapa 2 al entrenar;\npredichas en prueba','#edf0f4',21,17);
 arrow(s,325,312);arrow(s,680,312);t(s,'y_fase supervisa',410,260,220,24,16,C.green,true);
 processBox(s,56,455,350,112,'Directo','X → p(fatiga)','#fff0d8',23,20);
 processBox(s,465,455,350,112,'Cascada','X + q_fase(OOF) → p(fatiga)','#fff0d8',23,18);
 processBox(s,874,455,350,112,'Expertos por fase','q_ext·f_ext(X) + q_flex·f_flex(X)','#fff0d8',23,18);
 t(s,'Y_FATIGA DE REFERENCIA SUPERVISA LAS TRES CABEZAS',56,410,1168,34,18,C.green,true);
 t(s,'X alimenta las tres cabezas; las probabilidades de fase entran a la cascada y a los expertos.',56,578,1168,28,17,C.orange,true);
 t(s,'Evaluar en los mismos pliegues: Accuracy, macro-F1, calibración y matrices.',56,610,1168,30,17,C.ink,true);
 t(s,'Sin referencias sincronizadas, no hay modelo ni métricas de fatiga.',56,646,1168,24,16,C.red,true);
}
{
 const s=slide('Anexo: Ecuaciones de etiquetas y arquitecturas',0,
 'Las ecuaciones separan las dos tareas supervisadas. El modelo de fase aprende de ángulo, video o cinemática revisados. El modelo de fatiga aprende de una regla mecánica aprobada basada en torque, su error de medición y persistencia entre evaluaciones. La logística directa usa X. La cascada usa X y probabilidades de fase fuera de pliegue. Los especialistas multiplican sus predicciones por probabilidades de fase. La cabeza auxiliar realmente ejecutada usa un RF para predecir mitad de ráfaga EMG y una logística para la posición temporal del registro; esta fórmula no es una forma alternativa de entrenar fatiga real.',
 'docs/reference_fatigue_protocol.md; protocolo histórico y model_registry.csv');
 table(s,['Componente','Ecuación','Estado / supervisión'],[
  ['Fase mecánica','ω(t)=dθ/dt; flexión si ω>δ; extensión si ω<−δ; abstener si |ω|≤δ.','Propuesto; δ y signo dependen de sensor sincronizado y ruido.'],
  ['Fatiga por torque','T_ref=mediana(T₁,T₂,T₃); MDC95=1.96√2·SEM; onset candidato si mediana móvil < T_ref−MDC95 en 2 evaluaciones.','Regla candidata; requiere aprobación biomecánica, repetibilidad, calibración y sincronización.'],
  ['Directo','p_direct=σ(β₀+βᵀX); σ(z)=1/(1+e^(−z)).','Modelo base propuesto; etiqueta de torque/fuerza independiente.'],
  ['Cascada','p_cascade=σ(a₀+aᵀX+η_ext q_ext+η_flex q_flex).','Probabilidades OOF en entrenamiento; predicciones en prueba.'],
  ['Expertos suaves','p_expert=q_ext·f_ext(X)+q_flex·f_flex(X); q_ext+q_flex=1.','Expertos separados por fase, combinados con compuerta predicha.'],
  ['Auxiliar histórico','q(X)=B⁻¹Σ_b p_b(burst_half=1|X), B=100; p_proxy=σ(β₀+βᵀX+γq+δ_Rq·RMS+δ_Mq·MDF+δ_Nq·MNF).','Ejecutado sobre proxy temporal; mitad de ráfaga no equivale a extensión/flexión.'],
 ],56,145,1168,475,[210,575,383],16);
 t(s,'Fine tuning requiere datos externos con etiquetas; RL no aplica sin acciones y recompensas.',56,628,1168,34,17,C.muted);
}
{
 const s=slide('Anexo: Señal real y respuesta del filtro',0,
 'La figura procede de un intervalo de un segundo elegido por la primera ventana continua válida según control de calidad, sin mirar las etiquetas ni el resultado de ningún modelo. Se muestran cuentas ADC originales centradas, salida del mismo filtro SOS, periodogramas Hann antes y después, y respuesta digital exacta para la frecuencia de muestreo registrada. La tasa efectiva fue prácticamente mil hercios. La imagen es evidencia de procesamiento, no de rendimiento fisiológico ni de una clase de fatiga.',
 'signal_preprocessing_evidence.receipt.json; figura agregada privada sin identificador de registro');
 image(s,'signal_preprocessing_evidence.png',56,138,720,520);
 t(s,'≈1.000 Hz\n\n20–400 Hz\n\nOrden prototipo 4\nOrden digital 8\n\n100 ms enmascarados por borde',822,174,360,300,27,C.blue,true);
 t(s,'Espectro Hann de un solo intervalo; el método ejecutado es periodograma, no Welch. Las cuentas ADC no son una amplitud física calibrada.',822,490,360,132,20,C.muted);
 t(s,'Ilustración privada sin etiquetas de fatiga o movimiento.',56,641,1168,33,18,C.red,true);
}
const englishScript=[
 ['Temporal-proxy classification with sEMG','This study classifies an early-versus-late proxy created from each recording’s position in time. The ZIP does not contain independent measurements to claim true muscle-fatigue detection. I will describe the corpus, signal, pre-analysis, preprocessing, models, and the auxiliary experiment, separating completed results from the proposed design.','Integrative biomechanics project'],
 ['Corpus and quantitative overview','The main ZIP contains 140 CSV files and 4,256,257 samples. The 138 R1 to R5 recordings enter formal processing; two R0 files remain controls. Fifteen groups come from folder organization. Both legs and every series stay together during grouped evaluation. The RAR remains separate because participant identity and cross-archive links are not verified.','inventory.csv and coverage.csv'],
 ['Recorded signal and time diagnostics','The CSV files contain relative time and ADC counts at approximately one thousand samples per second. The plotted trace uses one selected QC-valid run, while processing preserves every sample. We inspected clock gaps, jitter, constant segments and repeated extremes. ADC counts do not establish calibrated muscle amplitude in millivolts.','Original CSV, qc.py, preprocessing.py'],
 ['Descriptive signal pre-analysis','These curves summarize candidate-active windows by recording position and then by group. They show electrical amplitude and spectral changes in the available files. They do not establish fatigue, because there are no synchronized force or torque labels, and the curves are descriptive rather than training targets.','feature_trajectories_by_group.csv'],
 ['Signal preprocessing','The historical pipeline separates signal at acquisition gaps and rejects excessive clock jitter and constant segments. It uses a fourth-order Butterworth low-pass prototype transformed into a digital 20 to 400 hertz band-pass, applied bidirectionally in second-order sections. One hundred milliseconds at each edge are masked. The optional notch is not in the primary pipeline. Processing is offline.','configs/study.yaml, qc.py and preprocessing.py'],
 ['Windows and temporal proxy labels','Complete windows in the first elapsed-time third receive proxy class zero; windows in the final third receive class one. The central third and boundary-crossing windows are excluded. A candidate-activity criterion selects usable windows. The observation is a window; the learned target is temporal position, not measured muscle fatigue.','label_audit.json and coverage.csv'],
 ['Ten features per window','The input contains three amplitude features, three spectral features and four relative db4 wavelet energies. The two-stage proxy experiment and direct comparator use the same ten original features. Time, file identity, side and series are excluded. The scaler is learned only from training folds, and no MVC normalization is claimed.','features.py and feature_schema.json'],
 ['Grouped validation method','Each outer round holds out one entire recording-derived group. The other groups are used for grouped inner model selection. Scaling and fitting use training data only. Each eligible window receives an out-of-fold prediction. Uncertainty is summarized across groups rather than treating overlapping windows as independent subjects.','evaluation.py, splits.json and evaluation_receipt.json'],
 ['Models and parameter selection','The majority model is a trivial reference. Logistic regression on RMS and MDF is the simple baseline. Full logistic regression, an RBF support-vector machine and a random forest test more flexible mappings. Nested selection chooses a family inside grouped validation; it does not average model votes.','evaluation.py and inner_search.csv'],
 ['Performance on the temporal proxy','All procedures use the same 5,509 early/late windows and fifteen held-out groups. The nested selector has macro-F1 0.6240 versus 0.6272 for logistic RMS plus MDF. The paired difference is −0.0032, with a 95 percent group-bootstrap interval from −0.0591 to +0.0509. No complex-model gain is demonstrated. These values are not Accuracy or F1 for physiological fatigue.','evaluation/metrics_summary.csv and paired_differences.csv'],
 ['Main proxy errors','The pooled confusion matrix shows errors for the early and late temporal classes. Group-level macro-F1 varies substantially, so pooled counts do not replace across-group uncertainty. Without independent references, these errors cannot be assigned a physiological cause or interpreted as fatigue mistakes.','evaluation/oof_predictions.parquet and evaluation/results.csv'],
 ['Auxiliary EMG cascade','The first stage is a random forest that classifies first versus second halves of candidate EMG bursts. Its score enters a logistic second stage together with the original ten features and selected interactions. The auxiliary label is an EMG burst half, not observed knee extension or flexion.','phase_spike.py and the frozen exploratory protocol'],
 ['Auxiliary experiment design','The variants use identical windows, temporal labels, scaling and held-out groups. We compare the direct ten-feature logistic model, EMG plus score, conditioned interactions, score only and a random-label control. The auxiliary head was fit on 3,337 windows from 996 candidate bursts.','model_registry.csv and phase protocol'],
 ['Auxiliary performance on the same proxy','The direct model reaches 0.6355 macro-F1; the conditioned model reaches 0.6293. Their paired difference is −0.0062, with a 95 percent interval from −0.0232 to +0.0082. Six groups improve, eight decline and one ties. The result does not show an improvement on the temporal proxy and says nothing about true fatigue detection.','phase_experiment/metrics_summary.csv and paired_differences.csv'],
 ['Targets for each cascade stage','The auxiliary confusion matrix evaluates burst-half labels, while the second-stage matrix evaluates early-versus-late recording position. They use different supports. The first matrix is not extension versus flexion, and neither matrix is fatigue versus no fatigue.','confusion_matrices.csv and auxiliary_phase_fold_metrics.csv'],
 ['Conclusions','The executed experiment predicts a temporal proxy, not muscular fatigue versus no fatigue. The nested selector does not clearly improve on RMS plus MDF; the burst-half score also fails to improve the direct proxy model. This ZIP cannot produce valid physiological fatigue Accuracy or macro-F1. A future evaluation needs synchronized calibrated force or torque; phase supervision needs synchronized kinematics.','Primary evaluation and auxiliary experiment'],
 ['Appendix: relationships among the models','The project includes six principal procedures, five supplemental variants, an auxiliary head and a separate full-cohort fitted artifact. The matched ten-feature logistic baseline is distinct from the nested selection procedure. These roles should not be counted as independent evidence or independent cohorts.','model_registry.csv'],
 ['Appendix: sensitivities and ablations','Prespecified sensitivities and feature-family ablations test how proxy results change under alternate analysis choices. Some sensitivities use different retained windows. We report these scenarios as sensitivity checks and do not select the most favorable one as the headline result.','sensitivity_summary.csv and ablation_summary.csv'],
 ['Appendix: principal confusion counts','These are pooled out-of-fold confusion counts for each principal procedure on the same 5,509 windows. Class zero means the first recording third; class one means the last third.','Primary confusion matrices'],
 ['Appendix: normalized principal matrices','Each row is normalized within its observed temporal-proxy class. The matrices show class-specific error proportions, not validated physiological sensitivity or specificity.','Normalized primary confusion matrices'],
 ['Appendix: auxiliary confusion counts','These matrices compare the supplemental stage-two variants on the same early/late proxy support. The auxiliary burst-half head is shown separately because it has another target.','Supplemental confusion matrices'],
 ['Appendix: normalized auxiliary matrices','These row-normalized matrices show error proportions for each supplemental procedure. The first-stage head predicts EMG burst halves, not knee movement phase.','Normalized supplemental confusion matrices'],
 ['Appendix: references for movement phase','The cited literature motivates measuring movement phase and controlling kinematics when interpreting EMG. Those studies do not validate our burst-half labels or demonstrate improved fatigue prediction from this architecture.','docs/PHASE_REFERENCES.md'],
 ['Appendix: window lengths and overlap','The historical analysis uses 250 millisecond windows with 50 percent overlap. A separate study with independent references would use 500 milliseconds as its primary window and 250 milliseconds as a sensitivity analysis. At one kilohertz these yield nominal bin spacings of two and four hertz. Longer windows improve nominal frequency spacing; overlap increases temporal sampling but creates dependent observations.','Frozen historical configuration and configs/reference_study.yaml'],
 ['Appendix: filter and window equations','The fourth-order Butterworth analog prototype is transformed to a digital band-pass of order eight and implemented as second-order sections. Coefficients depend on verified sampling rate. Forward-backward filtering is offline. The optional notch is used only if line-noise diagnostics justify it. The historical spectrum uses a one-sided Hann periodogram, not Welch averaging.','SciPy signal documentation: https://docs.scipy.org/doc/scipy/reference/signal.html'],
 ['Appendix: feature equations','The ten input features summarize amplitude, spectrum and relative wavelet energy. The corrected waveform-length feature uses the interval between first and last samples and has a separate schema. It does not replace the frozen historical version or alter its results.','features.py, feature_schema.json and docs/reference_fatigue_protocol.md'],
 ['Appendix: independent-reference architecture','Synchronized angle or reviewed video provides extension and flexion supervision. Calibrated synchronized force or torque provides the approved fatigue label. The same feature matrix feeds the direct model, phase cascade and soft-gated experts. OOF phase probabilities are used in cascade training, and held-out participants are evaluated on matched support. This architecture has not been trained on the supplied ZIP.','docs/reference_fatigue_protocol.md'],
 ['Appendix: label and model equations','Phase labels are derived from angular velocity and an approved deadband. The candidate torque rule requires calibration, repeatability and approval. Direct, cascade and expert equations define three prospective models. The executed auxiliary formula still predicts EMG burst halves and a temporal proxy, not physiological fatigue. Fine-tuning would require compatible external labeled data; reinforcement learning is not suited to static window labels without sequential actions and rewards.','docs/reference_fatigue_protocol.md and historical model registry'],
 ['Appendix: real signal and filter response','The figure shows a one-second interval selected by the first continuous QC-valid window, without checking labels or model results. It includes centered stored ADC counts, SOS output, Hann periodograms before and after filtering, and the digital filter response at the observed rate. This is preprocessing evidence, not physiological performance or a fatigue class.','Private signal_preprocessing_evidence receipt; anonymized figure'],
];
const main=script.filter(v=>v.seconds>0);
const seconds=main.reduce((a,b)=>a+b.seconds,0);
if(main.length!==16||seconds!==600||deck.slides.items.length!==29)throw new Error('Slide/time contract: '+seconds+' / '+deck.slides.items.length);
await fs.writeFile(path.join(output,'Guion_defensa.md'),'# Guion de la defensa\n\n16 diapositivas principales. Tiempo previsto: 600 segundos. Trece anexos para preguntas.\n\n'+script.map(v=>`## ${v.index}. ${v.title}\n\n${v.seconds?'Tiempo previsto: '+v.seconds+' s.':'Anexo para preguntas.'}\n\n${v.notes}\n\nFuente: ${v.source}.`).join('\n\n')+'\n');
if(englishScript.length!==script.length)throw new Error('English script does not match slide count');
const mdScript=(title,entries,lang)=>'# '+title+'\n\n16 principal slides. Scheduled talk time: 600 seconds. Thirteen appendix slides for questions.\n\n'+entries.map((v,i)=>`## ${i+1}. ${v[0]}\n\n${script[i].seconds?'Scheduled time: '+script[i].seconds+' s.':'Appendix for questions.'}\n\n${v[1]}\n\nSource: ${v[2]}.`).join('\n\n')+'\n';
await fs.writeFile(path.join(output,'Guion_defensa_ES.md'),'# Guion de la defensa\n\n16 diapositivas principales. Tiempo previsto: 600 segundos. Trece anexos para preguntas.\n\n'+script.map(v=>`## ${v.index}. ${v.title}\n\n${v.seconds?'Tiempo previsto: '+v.seconds+' s.':'Anexo para preguntas.'}\n\n${v.notes}\n\nFuente: ${v.source}.`).join('\n\n')+'\n');
await fs.writeFile(path.join(output,'Defense_Script_EN.md'),mdScript('EMG Defense Script',englishScript,'en'));
await fs.writeFile(path.join(build,'slide_content.json'),JSON.stringify({main_slides:16,reserve_slides:13,total:29,seconds,slides:script},null,2)+'\n');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
if(applyPresentationChartFont)for(const s of deck.slides.items)for(const ch of s.charts.items)applyPresentationChartFont(ch,{fontFamily:'Arial'});
const candidatePath=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidatePath);
const finalPath=path.join(output,'EMG_Binary_Defense.pptx');
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:runtimePython,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
  ...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:29,requiredNativeTableOwnerSlides:[...new Set(tableOwners)],
 requiredNativeChartOwnerSlides:[...new Set(chartOwners)],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
for(let i=0;i<deck.slides.items.length;i++){
 const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(finalPath);
