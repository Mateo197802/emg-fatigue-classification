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
const C={blue:'#234f6b',ink:'#182a36',muted:'#526775',green:'#39735d',pale:'#eef3f6',line:'#d4dfe5'};
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
function arrow(s,x,y){t(s,'→',x,y,60,55,40,C.blue,true);}
function cm(s,model,x,y,w=555,h=242){
 const q=d.confusions.find(q=>q.model===model);if(!q)throw new Error('Missing matrix '+model);
 table(s,['Real / predicha','Clase 0','Clase 1'],[['Clase 0',q.cm_0_0,q.cm_0_1],['Clase 1',q.cm_1_0,q.cm_1_1]],x,y,w,h,[w*.44,w*.28,w*.28],26);
}
function ms(q){return q.mean.toFixed(4)+' ± '+q.sd.toFixed(4);}
const imageNames=['sensitivity.png','ablation.png',...['primary_confusions_counts.png','primary_confusions_normalized.png','phase_confusions_counts.png','phase_confusions_normalized.png'].map(n=>'phase/'+n)];
const images=new Map(await Promise.all(imageNames.map(async n=>[n,new Uint8Array(await fs.readFile(path.join(workspaceDir,'report_assets',n)))])));
function image(s,n,x,y,w,h){s.images.add({blob:images.get(n),contentType:'image/png',alt:n,fit:'contain',position:{left:x,top:y,width:w,height:h}});}

{
 const s=slide('Clasificación de fatiga muscular con sEMG',20,
 'Estudiamos si la señal disponible permite distinguir estados asociados con la progresión del esfuerzo muscular. Presentaremos corpus, señal, preanálisis, procesamiento, modelos y experimento auxiliar. La referencia evaluada es primer frente a último tercio del registro. Conservamos esta definición al interpretar todas las métricas.',
 'Proyecto integrador de biomecánica');
 t(s,'Análisis de la señal y\nevaluación de modelos',56,176,660,145,51,C.blue,true);
 t(s,'Recto femoral durante ejercicio\nde extensión de rodilla',56,354,635,95,31,C.muted);
 t(s,'Ethan Guevara\nMateo Gavilanes\nKennedy Rodriguez',56,505,620,119,25);
 chart(s,'line',d.signal.time.map(v=>v.toFixed(2)),[{name:'sEMG filtrada',values:d.signal.filtered,line:{fill:C.blue,width:1.2},marker:{symbol:'none'}}],735,185,475,270,{xAxis:{visible:false},yAxis:{visible:false}});
 t(s,'Segmento real de sEMG',770,465,430,45,24,C.blue,true);t(s,'Referencia evaluada\nInicio y final del registro',770,535,430,80,24,C.muted);
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
 const s=slide('Ventanas y construcción de las etiquetas',45,
 'Los tercios usan los límites temporales originales antes de filtrar y excluir. Ventanas completas del primer tercio reciben clase cero y del último clase uno. Centro y fronteras quedan fuera. Exigimos al menos setenta y cinco por ciento de actividad candidata. De 33.090 ventanas candidatas quedan 32.474 con calidad válida, 9.289 activas con características válidas y 5.509 tempranas o tardías. Una ventana descartada no se convierte en no fatiga. Esta referencia temporal no verifica por sí sola fatiga fisiológica.',
 'label_audit.json y coverage.csv');
 block(s,56,161,366,175,'Primer tercio','Clase 0\n2.329 ventanas',C.green);
 block(s,458,161,366,175,'Tercio central','Fuera del contraste\nSin etiqueta binaria');
 block(s,860,161,364,175,'Último tercio','Clase 1\n3.180 ventanas');
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
 const s=slide('Rendimiento de los modelos principales',45,
 'Todos usan las mismas 5.509 ventanas. Macro-F1 muestra media y desviación entre quince grupos. AUC es también media por grupo. El selector alcanza 0,6240 frente a 0,6272 del modelo simple. La diferencia es menos 0,0032, con intervalo de menos 0,0591 a más 0,0509. No aparece ventaja clara del procedimiento complejo. La logística completa de diez características será el comparador de la fase.',
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
 const s=slide('Arquitectura auxiliar de dos etapas',45,
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
 const s=slide('Rendimiento con la arquitectura auxiliar',40,
 'El directo obtuvo 0,6355 de macro-F1, agregar score 0,6326, condicionar 0,6293, sólo score 0,6184 y el aleatorio 0,6356. El contraste principal condicionado menos directo es menos 0,0062 y su intervalo de menos 0,0232 a más 0,0082 cruza cero. Seis grupos mejoran, ocho empeoran y uno empata. No hay mejora clara, pero este resultado tampoco descarta que una fase medida con cinemática pueda ayudar en otro estudio.',
 'phase_experiment/metrics_summary.csv y paired_differences.csv');
 const n=[['Directo 10','baseline_10'],['EMG + score','phase_score'],['Condicionado por fase','phase_conditioned'],['Sólo score','phase_only'],['Control aleatorio','random_conditioned']];
 table(s,['Variante','Macro-F1 media ± DE','AUC media'],n.map(([name,id])=>[name,ms(d.phase[id].f1_macro),d.phase[id].roc_auc.mean.toFixed(4)]),56,155,1168,342,[440,438,290],26);
 t(s,'Δ condicionado − directo = −0,0062',56,535,1168,52,33,C.blue,true);
 t(s,'IC 95% [−0,0232, +0,0082]     6 mejoran, 8 empeoran y 1 empata',56,603,1168,51,24);
}
{
 const s=slide('Resultados de cada etapa de la cascada',35,
 'Las matrices tienen objetivos y soportes distintos. La auxiliar usa 1.663 primeras mitades y 1.674 segundas, con macro-F1 0,5420 y AUC 0,6212. La logística condicionada usa las 5.509 ventanas tempranas y tardías. Acierta 1.538 tempranas y 2.141 tardías. No se comparan los totales ni se interpreta la primera matriz como extensión y flexión.',
 'confusion_matrices.csv y auxiliary_phase_fold_metrics.csv');
 t(s,'Etapa 1: mitad de ráfaga EMG',56,160,555,70,29,C.blue,true);cm(s,'auxiliary_phase_head',56,254);
 t(s,'0 = primera mitad, 1 = segunda mitad\nMacro-F1 0,5420 ± 0,1028\nAUC 0,6212 ± 0,1288',56,535,555,110,24);
 t(s,'Etapa 2: clase temporal',671,160,555,70,29,C.blue,true);cm(s,'phase_conditioned',671,254);
 t(s,'0 = temprano, 1 = tardío\nMacro-F1 0,6293 ± 0,1330\nAUC 0,8431 ± 0,0741',671,535,555,110,24);
}
{
 const s=slide('Conclusiones',35,
 'El proyecto conecta registros, procesamiento, entrenamiento y predicción. La señal permite discriminar inicio y final del registro, con variación importante entre grupos. El selector no mejora claramente a RMS y MDF. La representación de mitades de ráfaga tiene discriminación limitada y no mejora claramente el modelo directo. El alcance de estas conclusiones es el objetivo temporal evaluado: no se identifica cada predicción con fatiga fisiológica confirmada ni con extensión o flexión observadas.',
 'Evaluación principal y experimento auxiliar');
 [['Procesamiento','Señal y ventanas con calidad controlada y validación por grupos'],['Modelos principales','Selector 0,6240 frente a RMS + MDF 0,6272 de macro-F1'],['Arquitectura auxiliar','Condicionado 0,6293 frente a directo 0,6355 de macro-F1']].forEach((v,i)=>{
  const y=169+i*126;t(s,v[0],56,y,340,52,29,C.blue,true);t(s,v[1],420,y,804,89,30);
 });
 t(s,'Los resultados sostienen discriminación temporal, con variabilidad entre grupos.',56,566,1168,82,30,C.blue,true);
}
{
 const s=slide('Anexo  Relación entre todos los modelos',0,
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
 const s=slide('Anexo  Sensibilidades y ablaciones',0,
 'Nueve sensibilidades prespecificadas y tres ablaciones. Las primeras usan logística fija RMS y MDF, con soportes distintos según exclusiones. Las ablaciones usan selección interna. Se conservan los resultados originales sin seleccionar el escenario más favorable.',
 'sensitivity_summary.csv y ablation_summary.csv');
 image(s,'sensitivity.png',35,165,650,440);image(s,'ablation.png',700,190,545,355);
 t(s,'El soporte cambia entre sensibilidades. Los resultados principales permanecen fijos.',56,616,1168,40,22,C.muted);
}
for(const [title,name,caption] of [
 ['Anexo  Matrices de los modelos principales','primary_confusions_counts.png','Conteos OOF. Cada procedimiento usa 5.509 ventanas.'],
 ['Anexo  Confusiones principales normalizadas','primary_confusions_normalized.png','Proporciones por fila. Clase 0 temprana, clase 1 tardía.'],
 ['Anexo  Matrices de las variantes auxiliares','phase_confusions_counts.png','Mismo soporte de 5.509 ventanas. Directo reproduce logística 10.'],
 ['Anexo  Confusiones auxiliares normalizadas','phase_confusions_normalized.png','Proporciones por fila. Matrices de la segunda etapa.']]){
 const s=slide(title,0,caption+' La incertidumbre principal se calcula entre grupos.',
 'results/phase_experiment/confusion_matrices.csv');
 image(s,'phase/'+name,56,145,1168,465);t(s,caption,56,622,1168,35,22,C.muted);
}
{
 const s=slide('Anexo  Bibliografía del experimento auxiliar',0,
 'Cheah et al. usan fase medida con IMU y una referencia temporal al final de repeticiones. Potvin estudia efectos cinemáticos sobre EMG. Guzu et al. combinan ejercicio y estados EMG con otro diseño. Motivan estudiar fase, pero no validan nuestras etiquetas ni una mejora de esta cascada.',
 'docs/PHASE_REFERENCES.md');
 table(s,['Referencia','Aporte para la hipótesis'],[
  ['Cheah et al. 2022\n10.54941/ahfe1002597','Fase dinámica medida con IMU'],
  ['Potvin 1997\n10.1152/jappl.1997.82.1.144','Influencia de ángulo y velocidad en EMG'],
  ['Guzu et al. 2024\n10.1109/ELMAR62909.2024.10694353','Reconocimiento de ejercicio y estados EMG']],56,165,1168,360,[610,558]);
 t(s,'La literatura motiva condicionar por movimiento. Esta prueba usa una fase candidata de EMG.',56,581,1168,72,25,C.muted);
}
const main=script.filter(v=>v.seconds>0);
const seconds=main.reduce((a,b)=>a+b.seconds,0);
if(main.length!==16||seconds!==600||deck.slides.items.length!==23)throw new Error('Slide/time contract: '+seconds);
await fs.writeFile(path.join(output,'Guion_defensa.md'),'# Guion de la defensa\n\n16 diapositivas principales. Tiempo previsto: 600 segundos. Siete anexos para preguntas.\n\n'+script.map(v=>`## ${v.index}. ${v.title}\n\n${v.seconds?'Tiempo previsto: '+v.seconds+' s.':'Anexo para preguntas.'}\n\n${v.notes}\n\nFuente: ${v.source}.`).join('\n\n')+'\n');
await fs.writeFile(path.join(build,'slide_content.json'),JSON.stringify({main_slides:16,reserve_slides:7,total:23,seconds,slides:script},null,2)+'\n');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
if(applyPresentationChartFont)for(const s of deck.slides.items)for(const ch of s.charts.items)applyPresentationChartFont(ch,{fontFamily:'Arial'});
const candidatePath=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidatePath);
const finalPath=path.join(output,'EMG_Binary_Defense.pptx');
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:runtimePython,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
  ...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:23,requiredNativeTableOwnerSlides:[...new Set(tableOwners)],
 requiredNativeChartOwnerSlides:[...new Set(chartOwners)],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
for(let i=0;i<deck.slides.items.length;i++){
 const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(finalPath);
