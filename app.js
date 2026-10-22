const $=i=>document.getElementById(i),cl=(x,a=0,b=1)=>Math.min(b,Math.max(a,x)),p2=n=>String(n).padStart(2,'0');
const stamp=d=>d.getFullYear()+'-'+p2(d.getMonth()+1)+'-'+p2(d.getDate())+' '+p2(d.getHours())+':'+p2(d.getMinutes())+':'+p2(d.getSeconds());
const C={nomV:230,ratedI:5,oilFull:4,oilLow:12,tLimit:32,tCrit:45,W:16,maxY:25};
const last={T:28,V:0,I:0,D:5,oil:100,fan:0,rul:null},logs=[],tel=[],buf=[],lv={};let model=null,infer=0,mae=null,lastFan=null,port=null,simOn=false,simTimer=null;
const feat=(T,V,I,o,f)=>[T/50,V/C.nomV/1.3,I/C.ratedI/2,o/100,f];
function comp(T,V,I,o){const K=I/C.ratedI,HS=T+22*Math.pow(K,1.6),th=cl((HS-30)/25),ov=cl((K-1)/.5),vd=cl(Math.abs(V/C.nomV-1)/.15),oi=cl((85-o)/55);return{K,HS,th,ov,vd,oi,s:.5*th+.2*ov+.15*vd+.15*oi}}
function toast(m){const t=$('toast');t.textContent=m;t.style.display='block';clearTimeout(toast.h);toast.h=setTimeout(()=>t.style.display='none',3500)}
/* ---------- event log ---------- */
function log(sev,src,msg){postEvent(sev,src,msg);logs.push({ts:new Date(),sev,src,msg,T:last.T,V:last.V,I:last.I,D:last.D,oil:last.oil,rul:last.rul});renderLog()}
function renderLog(){const f=$('evF').value,r=logs.filter(l=>!f||l.sev===f).slice(-150).reverse();$('ev').innerHTML=r.map(l=>`<tr><td>${stamp(l.ts)}</td><td class="s-${l.sev}">${l.sev}</td><td>${l.src}</td><td>${l.msg}</td><td>${l.T.toFixed(1)}</td><td>${l.V.toFixed(0)}</td><td>${l.I.toFixed(2)}</td><td>${l.rul==null?'–':l.rul.toFixed(1)}</td></tr>`).join('')}
function track(k,l,m){const o=lv[k]||0;if(l===o)return;lv[k]=l;if(l===0)log('INFO',k,m[0]+' (cleared)');else log(l===1?'WARNING':'CRITICAL',k,m[l])}
/* ---------- CNN + Bi-LSTM ---------- */
function gen(){const T0=24+Math.random()*26,dT=(Math.random()-.5)*8,K0=Math.random()*1.7,V0=.8+Math.random()*.4,O0=35+Math.random()*65,dO=(Math.random()<.3?-1:0)*Math.random()*25,x=[];let st=0;
 for(let i=0;i<C.W;i++){const f=i/(C.W-1),T=T0+dT*f+(Math.random()-.5)*.8,K=K0*(1+(Math.random()-.5)*.1),V=C.nomV*V0*(1+(Math.random()-.5)*.02),O=cl(O0+dO*f,0,100),I=K*C.ratedI;x.push(feat(T,V,I,O,T>C.tLimit?1:0));if(i>=C.W-4)st+=comp(T,V,I,O).s/4}
 return{x,y:Math.exp(-3*st)}}
async function train(){try{model=await tf.loadLayersModel('model/model.json');mae=+((await (await fetch('model/meta.json')).json()).mae_years||0);$('tp').style.width='100%';$('tm').textContent='Pretrained CNN + Bi-LSTM loaded · MAE ≈ '+mae.toFixed(2)+' years';log('INFO','Model','Pretrained CNN + Bi-LSTM loaded from model/model.json');return}catch(e){}const X=[],Y=[];for(let n=0;n<700;n++){const s=gen();X.push(s.x);Y.push([s.y])}
 model=tf.sequential({layers:[tf.layers.conv1d({inputShape:[C.W,5],filters:16,kernelSize:3,padding:'same',activation:'relu'}),tf.layers.conv1d({filters:32,kernelSize:3,padding:'same',activation:'relu'}),tf.layers.maxPooling1d({poolSize:2}),tf.layers.bidirectional({layer:tf.layers.lstm({units:24}),mergeMode:'concat'}),tf.layers.dense({units:16,activation:'relu'}),tf.layers.dense({units:1,activation:'sigmoid'})]});
 model.compile({optimizer:tf.train.adam(.004),loss:'meanSquaredError',metrics:['mae']});
 const xs=tf.tensor3d(X),ys=tf.tensor2d(Y),E=22;
 await model.fit(xs,ys,{epochs:E,batchSize:32,shuffle:true,callbacks:{onEpochEnd:async(e,l)=>{$('tp').style.width=((e+1)/E*100)+'%';$('tm').textContent=`Training epoch ${e+1}/${E} · loss ${l.loss.toFixed(4)}`;mae=l.mae*C.maxY;await tf.nextFrame()}}});
 xs.dispose();ys.dispose();$('tm').textContent=`Model ready · 700 sequences · validation-free MAE ≈ ${mae.toFixed(2)} years · inferences: 0`;log('INFO','Model','CNN + Bi-LSTM trained and online for RUL prediction')}
function predict(){const w=buf.slice(-C.W);return tf.tidy(()=>model.predict(tf.tensor3d([w])).dataSync()[0])*C.maxY}
/* ---------- ingest ---------- */
function ingest(T,V,I,D,fan){const tOk=T>-.5&&!isNaN(T),dOk=D>=0;if(tOk)last.T=T;if(dOk){last.D=D;last.oil=cl((C.oilLow-D)/(C.oilLow-C.oilFull))*100}last.V=V;last.I=I;last.fan=fan;
 const c=comp(last.T,V,I,last.oil);buf.push(feat(last.T,V,I,last.oil,fan));if(buf.length>60)buf.shift();
 if(model&&buf.length>=C.W){const r=predict();last.rul=last.rul==null?r:last.rul*.7+r*.3;infer++;if(infer%5==0)$('tm').textContent=`Model ready · MAE ≈ ${mae.toFixed(2)} years · inferences: ${infer}`}
 if(lastFan!==null&&fan!==lastFan)log('INFO','Cooling','Fan & buzzer '+(fan?'ON':'OFF')+` at ${last.T.toFixed(1)} °C`);lastFan=fan;
 track('Temperature',last.T>C.tCrit?2:last.T>C.tLimit?1:0,['Temperature normal',`Temperature ${last.T.toFixed(1)} °C above ${C.tLimit} °C limit`,`Critical temperature ${last.T.toFixed(1)} °C`]);
 track('Loading',c.K>1.25?2:c.K>1?1:0,['Load normal',`Overload: ${(c.K*100).toFixed(0)}% of rated current`,`Severe overload: ${(c.K*100).toFixed(0)}% of rated current`]);
 const dv=Math.abs(V/C.nomV-1);track('Voltage',dv>.15?2:dv>.1?1:0,['Voltage within limits',`Voltage deviation ${(dv*100).toFixed(0)}% (${V.toFixed(0)} V)`,`Critical voltage deviation ${(dv*100).toFixed(0)}% (${V.toFixed(0)} V)`]);
 track('Oil level',last.oil<35?2:last.oil<60?1:0,['Oil level normal',`Oil level low: ${last.oil.toFixed(0)}%`,`Oil level critically low: ${last.oil.toFixed(0)}%`]);
 track('Temp sensor',tOk?0:1,['DHT11 reading restored','DHT11 temperature read failure']);track('Oil sensor',dOk?0:1,['Ultrasonic reading restored','Ultrasonic echo timeout – oil sensor fault']);
 if(last.rul!=null)track('RUL',last.rul<3?2:last.rul<10?1:0,['Predicted RUL recovered',`Predicted RUL ${last.rul.toFixed(1)} years – plan maintenance`,`Predicted RUL ${last.rul.toFixed(1)} years – inspect immediately`]);
 tel.push({ts:new Date(),T:last.T,V,I,D:last.D,oil:last.oil,fan,HS:c.HS,rul:last.rul});if(tel.length>20000)tel.shift();ui(c)}
/* ---------- UI ---------- */
const mut=getComputedStyle(document.documentElement).getPropertyValue('--mut')||'#888';
function mk(id,ds,y1){const sc={x:{ticks:{maxTicksLimit:5,color:mut},grid:{display:false}},y:{ticks:{color:mut},grid:{color:'rgba(128,128,128,.18)'}}};if(y1)sc.y1={position:'right',ticks:{color:mut},grid:{drawOnChartArea:false}};
 return new Chart($(id),{type:'line',data:{labels:[],datasets:ds},options:{animation:false,responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},elements:{point:{radius:0},line:{borderWidth:2,tension:.3}},plugins:{legend:{labels:{boxWidth:10,color:mut}}},scales:sc}})}
const ch1=mk('c1',[{label:'Temp °C',data:[],borderColor:'#ef4444'},{label:'Hotspot °C',data:[],borderColor:'#f59e0b',borderDash:[5,4]}]),
ch2=mk('c2',[{label:'Voltage V',data:[],borderColor:'#38a3f5'},{label:'Current A',data:[],borderColor:'#22c55e',yAxisID:'y1'}],1),ch3=mk('c3',[{label:'RUL y',data:[],borderColor:'#a855f7',fill:true,backgroundColor:'rgba(168,85,247,.12)'}]);
function push(c,l,vals){c.data.labels.push(l);c.data.datasets.forEach((d,i)=>d.data.push(vals[i]));if(c.data.labels.length>120){c.data.labels.shift();c.data.datasets.forEach(d=>d.data.shift())}c.update('none')}
const ACT={th:'Inspect cooling circuit, fan relay and radiators; check ventilation and ambient heat',ov:'Reduce or redistribute load; review rating and tap setting',vd:'Check incoming supply, tap changer position and voltage sensor calibration',oi:'Inspect for oil leaks (gaskets, bushings); top up and run DGA / BDV oil tests'};
function ui(c){const t=new Date().toLocaleTimeString(),r=last.rul,hi=r==null?null:Math.round(cl(r/C.maxY)*100),cls=r==null?'n':r<3?'c':r<10?'w':'',lab=r==null?'Initialising':r<3?'Critical':r<10?'Degrading':'Healthy';
 $('kT').textContent=last.T.toFixed(1)+' °C';$('kH').textContent=c.HS.toFixed(1)+' °C';$('kV').textContent=last.V.toFixed(1)+' V';$('kI').textContent=last.I.toFixed(2)+' A · '+(c.K*100).toFixed(0)+'%';$('kO').textContent=last.oil.toFixed(0)+'% ('+(last.D<0?'ERR':last.D.toFixed(1)+' cm')+')';$('kF').textContent=last.fan?'ON':'OFF';
 $('hi').textContent=hi==null?'--':hi;$('rY').textContent=r==null?'--':r.toFixed(1);$('rH').textContent=r==null?(model?'Collecting '+C.W+'-sample window…':'Training model…'):'≈ '+Math.round(r*8760).toLocaleString()+' operating hours';
 $('st').className='tag '+cls;$('st').textContent=lab;const g=$('gR');g.style.stroke=r==null?'var(--mut)':r<3?'var(--bad)':r<10?'var(--warn)':'var(--ok)';g.setAttribute('stroke-dashoffset',326.7*(1-(hi||0)/100));
 const h=130-cl((last.T-26)/22)*130;document.querySelectorAll('.c').forEach(e=>e.style.fill=`hsl(${h},80%,50%)`);
 const y=165+176*(1-last.oil/100);$('oilR').setAttribute('y',y);$('oilR').setAttribute('height',341-y);$('oilL').setAttribute('y1',y);$('oilL').setAttribute('y2',y);$('gF').setAttribute('y',y);$('gF').setAttribute('height',341-y);$('oilT').textContent=last.oil.toFixed(0)+'%';
 $('fan').classList.toggle('on',!!last.fan);$('bc').classList.toggle('on',!!last.fan);$('tank').style.stroke=last.T>C.tCrit?'var(--bad)':last.T>C.tLimit?'var(--warn)':'var(--mut)';
 $('caps').style.fill=Math.abs(last.V/C.nomV-1)>.1?'var(--bad)':'var(--acc)';$('tv').textContent=`${last.T.toFixed(1)} °C · ${last.V.toFixed(0)} V · ${last.I.toFixed(2)} A · oil ${last.oil.toFixed(0)}%`;
 push(ch1,t,[last.T,c.HS]);push(ch2,t,[last.V,last.I]);if(r!=null)push(ch3,t,[r]);else if(ch3.data.labels.length)push(ch3,t,[null]);
 const cs=[['Thermal',c.th,'th'],['Loading',c.ov,'ov'],['Voltage',c.vd,'vd'],['Oil level',c.oi,'oi']];$('sb').innerHTML=cs.map(x=>`<div class="sm">${x[0]} · ${(x[1]*100).toFixed(0)}%</div><div class="bar"><i style="width:${x[1]*100}%;background:${x[1]>.6?'var(--bad)':x[1]>.3?'var(--warn)':'var(--ok)'}"></i></div>`).join('');
 const top=cs.filter(x=>x[1]>.15).sort((a,b)=>b[1]-a[1]).slice(0,2);$('ma').innerHTML=(top.length?top.map(x=>ACT[x[2]]):['No abnormal stress detected – continue routine monitoring (thermography, annual DGA)']).map(a=>'<li>'+a+'</li>').join('');
 $('mw').textContent=r==null?'Maintenance window: pending prediction':'Maintenance window: '+(r<3?'Immediate – outage within 7 days':r<10?'Within 30 days':r<18?'Next scheduled outage':'Routine annual inspection')}
/* ---------- sources ---------- */
const SC={normal:{K:.55,V:1,x:0},overload:{K:1.35,V:1,x:0},cooling:{K:.9,V:1,x:9},oil:{K:.6,V:1,x:0,leak:.05},sag:{K:.7,V:.82,x:0}},sm={T:27.5,D:5};
function simTick(){const s=SC[$('scn').value],K=s.K+(Math.random()-.5)*.08;const Tss=26+8*K*K+s.x-(last.fan&&!s.x?3.5:0);sm.T+=(Tss-sm.T)*.06+(Math.random()-.5)*.25;sm.D=Math.min(14,sm.D+(s.leak||0));
 ingest(+sm.T.toFixed(1),C.nomV*s.V*(1+(Math.random()-.5)*.01),K*C.ratedI,sm.D+(Math.random()-.5)*.1,sm.T>C.tLimit?1:0)}
function setSim(on){simOn=on;clearInterval(simTimer);if(on)simTimer=setInterval(simTick,1000);$('bSim').textContent='Simulator: '+(on?'ON':'OFF');pill()}
function pill(){const p=$('conn');p.className='pill'+(port||simOn?' on':'');p.textContent=port?'ESP32 live (serial)':simOn?'Simulated sensors':'Idle'}
function parseLine(l){const p=l.trim().split(',').map(Number);if(p.length===5&&p.every(x=>!isNaN(x)))ingest(...p)}
async function connect(){try{if(!navigator.serial)throw new Error('Web Serial needs Chrome/Edge – or use the paste box / simulator.');
 port=await navigator.serial.requestPort();await port.open({baudRate:115200});setSim(false);log('INFO','Connection','ESP32 serial link opened at 115200 baud');pill();
 const d=new TextDecoderStream();port.readable.pipeTo(d.writable);const rd=d.readable.getReader();let b='';
 for(;;){const{value,done}=await rd.read();if(done)break;b+=value;let i;while((i=b.indexOf('\n'))>=0){parseLine(b.slice(0,i));b=b.slice(i+1)}}}
 catch(e){toast(e.message||'Serial connection failed – open this page in its own browser tab');port=null;pill()}}
/* ---------- Excel export ---------- */
async function exportXlsx(){const wb=XLSX.utils.book_new(),f=v=>v==null?'':+(+v).toFixed(2);
 const ev=[['Timestamp','Severity','Source','Event','Temp (°C)','Voltage (V)','Current (A)','Oil distance (cm)','Oil level (%)','Predicted RUL (years)']].concat(logs.map(l=>[stamp(l.ts),l.sev,l.src,l.msg,f(l.T),f(l.V),f(l.I),f(l.D),f(l.oil),f(l.rul)]));
 const te=[['Timestamp','Temp (°C)','Voltage (V)','Current (A)','Oil distance (cm)','Oil level (%)','Fan','Est. hotspot (°C)','Predicted RUL (years)']].concat(tel.map(t=>[stamp(t.ts),f(t.T),f(t.V),f(t.I),f(t.D),f(t.oil),t.fan?'ON':'OFF',f(t.HS),f(t.rul)]));
 const su=[['Report generated',stamp(new Date())],['Exported by',user?user.name+' ('+user.role+')':'-'],['Telemetry samples',tel.length],['Events logged',logs.length],['Critical events',logs.filter(l=>l.sev==='CRITICAL').length],['Warnings',logs.filter(l=>l.sev==='WARNING').length],['Latest predicted RUL (years)',f(last.rul)],['Model','Conv1D(16)-Conv1D(32)-MaxPool-BiLSTM(24)-Dense(16)-Dense(1), window '+C.W],['Nominal V / rated I',C.nomV+' V / '+C.ratedI+' A'],['Fan/buzzer temperature limit (°C)',C.tLimit]];
 [['Event Log',ev,[20,10,13,60,10,11,11,15,13,18]],['Telemetry',te,[20,10,11,11,15,13,7,16,18]],['Summary',su,[34,60]]].forEach(([n,a,w])=>{const s=XLSX.utils.aoa_to_sheet(a);s['!cols']=w.map(x=>({wch:x}));XLSX.utils.book_append_sheet(wb,s,n)});
 const name='Transformer_Event_Log_'+stamp(new Date()).replace(/[: ]/g,'-')+'.xlsx',arr=XLSX.write(wb,{type:'array',bookType:'xlsx'});
 let d=null;try{d=window.claude&&await window.claude.use('downloads')}catch(e){}
 if(d){try{await d.save({filename:name,data:arr})}catch(e){if(e.code!=='declined')toast('Download failed: '+(e.message||e.code))}}else XLSX.writeFile(wb,name)}
/* ---------- wiring ---------- */
$('bConn').onclick=connect;$('bSim').onclick=()=>setSim(!simOn);$('bX').onclick=$('bX2').onclick=exportXlsx;$('evF').onchange=renderLog;
$('bTh').onclick=()=>{const r=document.documentElement,d=getComputedStyle(r).getPropertyValue('--bg').trim()==='#0b111a';r.dataset.theme=d?'light':'dark'};
$('bCsv').onclick=()=>{parseLine($('csv').value)};
$('scn').onchange=()=>{if($('scn').value==='normal')sm.D=5;log('INFO','Simulator','Scenario: '+$('scn').selectedOptions[0].text)};
[['cV','nomV'],['cI','ratedI'],['cF','oilFull'],['cL','oilLow']].forEach(([i,k])=>$(i).onchange=e=>{const v=parseFloat(e.target.value);if(v>0)C[k]=v});
log('INFO','System','Digital twin online – waiting for sensor stream');setSim(true);ui(comp(last.T,0,0,100));
train().catch(e=>{$('tm').textContent='Model failed to load: '+e.message;toast('TensorFlow.js could not start')});
/* ---------- login ---------- */
const PW={ae:['Assistant Engineer','AE@123'],aee:['Assistant Executive Engineer','AEE@123'],ee:['Executive Engineer','EE@123']};let user=null;
var token=null,ws=null;
function postEvent(sev,src,msg){if(!token)return;fetch('/api/events',{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+token},body:JSON.stringify({sev,src,msg,T:last.T,V:last.V,I:last.I,rul:last.rul})}).catch(()=>{})}
async function checkAuth(r,n,p){try{const res=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({role:r,username:n,password:p})});if(res.ok){token=(await res.json()).token;return true}if(res.status===401)return false}catch(e){}return p===PW[r][1]}
function connectWS(){if(ws)return;ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws?token='+encodeURIComponent(token));
 ws.onopen=()=>{setSim(false);log('INFO','Connection','Backend WebSocket connected');const p=$('conn');p.className='pill on';p.textContent='Backend live'};
 ws.onmessage=m=>{try{const d=JSON.parse(m.data);if(d.type==='telemetry')ingest(d.T,d.V,d.I,d.D,d.fan)}catch(e){}};
 ws.onclose=()=>{ws=null;log('WARNING','Connection','Backend link closed');pill()}}
async function signIn(){const r=document.querySelector('input[name=role]:checked').value,n=$('lu').value.trim(),e=$('le');
 if(!n){e.textContent='Please enter a user name.';return}
 if(!(await checkAuth(r,n,$('lp').value))){e.textContent='Incorrect password for '+PW[r][0]+'.';$('lp').value='';log('WARNING','Auth','Failed sign-in attempt by "'+n.replace(/[<>&]/g,'')+'" as '+PW[r][0]);return}
 user={name:n,role:PW[r][0]};e.textContent='';$('lp').value='';$('login').hidden=true;$('app').hidden=false;$('who').textContent=n+' \u00b7 '+PW[r][0];
 log('INFO','Auth','Signed in: '+n.replace(/[<>&]/g,'')+' ('+PW[r][0]+')');[ch1,ch2,ch3].forEach(c=>c.resize());if(token)connectWS()}
function signOut(){if(user)log('INFO','Auth','Signed out: '+user.name.replace(/[<>&]/g,'')+' ('+user.role+')');user=null;if(ws){ws.onclose=null;ws.close();ws=null}token=null;$('app').hidden=true;$('login').hidden=false;$('lu').value=''}
$('lb').onclick=signIn;$('bOut').onclick=signOut;['lu','lp'].forEach(i=>$(i).onkeydown=e=>{if(e.key==='Enter')signIn()});$('lsh').onchange=e=>$('lp').type=e.target.checked?'text':'password';
