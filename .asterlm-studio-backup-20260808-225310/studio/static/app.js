"use strict";

const S = {
  overview: null,
  catalog: null,
  selectedJob: null,
  selectedRun: null,
  selectedConfig: null,
  cleanPlanPath: null,
  trainingPlan: null,
  timer: null,
};

const $ = (q, root=document) => root.querySelector(q);
const $$ = (q, root=document) => [...root.querySelectorAll(q)];

function esc(value) {
  return String(value ?? "")
    .replaceAll("&","&amp;").replaceAll("<","&lt;")
    .replaceAll(">","&gt;").replaceAll('"',"&quot;");
}

function fmtTokens(value) {
  if (value == null) return "—";
  const x = Number(value);
  if (x >= 1e12) return `${(x/1e12).toFixed(2)}T`;
  if (x >= 1e9) return `${(x/1e9).toFixed(2)}B`;
  if (x >= 1e6) return `${(x/1e6).toFixed(2)}M`;
  if (x >= 1e3) return `${(x/1e3).toFixed(1)}K`;
  return `${Math.round(x)}`;
}
function fmtBytes(value) {
  if (value == null) return "—";
  const units=["B","KiB","MiB","GiB","TiB"]; let x=Number(value),i=0;
  while(x>=1024&&i<units.length-1){x/=1024;i++;}
  return `${x.toFixed(i<2?0:2)} ${units[i]}`;
}
function fmtSecs(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  let s=Math.max(0,Math.round(Number(value)));
  const d=Math.floor(s/86400);s%=86400; const h=Math.floor(s/3600);s%=3600; const m=Math.floor(s/60);s%=60;
  if(d)return `${d}d ${h}h`; if(h)return `${h}h ${m}m`; if(m)return `${m}m ${s}s`; return `${s}s`;
}
function ago(ts) {
  if (!ts) return "—";
  const d=Math.max(0,Date.now()/1000-Number(ts));
  if(d<60)return `${Math.round(d)}s ago`;
  if(d<3600)return `${Math.round(d/60)}m ago`;
  if(d<86400)return `${Math.round(d/3600)}h ago`;
  return `${Math.round(d/86400)}d ago`;
}
function clamp(x,a,b){return Math.max(a,Math.min(b,x));}

async function api(path, options={}) {
  const init = {...options, headers: {"Content-Type":"application/json", ...(options.headers||{})}};
  const res = await fetch(path, init);
  const data = await res.json().catch(()=>({error:`HTTP ${res.status}`}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
async function post(path, body) {
  return api(path,{method:"POST",body:JSON.stringify(body)});
}

function toast(message, type="ok") {
  const node=document.createElement("div");
  node.className="toast"; node.textContent=message;
  $("#toast-root").appendChild(node);
  setTimeout(()=>node.remove(),5000);
}
function notice(message,type="") {
  const node=document.createElement("div");
  node.className=`notice ${type}`; node.textContent=message;
  $("#notice-stack").appendChild(node);
  setTimeout(()=>node.remove(),9000);
}
function showError(error){console.error(error);notice(error.message||String(error),"bad");}

const PAGE_TITLES = {
  overview:"The state of the experiment.",
  data:"Build the corpus you actually want.",
  pipeline:"Turn raw material into training data.",
  architecture:"Know what is real, reference, or research.",
  training:"Design and launch the run.",
  experiments:"Read the model while it learns.",
  inference:"Interrogate the finished checkpoints.",
  logs:"Every process, in one place.",
  settings:"Tune the machine without editing shell commands.",
};

function gotoPage(page) {
  $$(".page").forEach(n=>n.classList.toggle("active",n.id===`page-${page}`));
  $$(".nav-item").forEach(n=>n.classList.toggle("active",n.dataset.page===page));
  $("#page-title").textContent=PAGE_TITLES[page]||page;
  history.replaceState(null,"",`#${page}`);
  if(page==="architecture") loadConfigs("model").catch(showError);
  if(page==="experiments") renderRuns();
  if(page==="logs") renderJobs();
}

function statusPill(item) {
  if (item.source_exhausted && !item.complete) return `<span class="status-pill warning">exhausted @ ${fmtTokens(item.tokens)}</span>`;
  if (item.complete) return `<span class="status-pill good">complete</span>`;
  return `<span class="status-pill ref">${Number(item.percent||0).toFixed(1)}%</span>`;
}

function datasetLedger(rows) {
  return rows.map(item=>{
    const pct=clamp(Number(item.percent||0),0,100);
    const cls=item.complete?"good":item.source_exhausted?"warn":"";
    return `<div class="ledger-row">
      <div class="ledger-name"><strong>${esc(item.label||item.id)}</strong><small>${statusPill(item)}</small></div>
      <div class="progress-track"><div class="progress-fill ${cls}" style="width:${pct}%"></div></div>
      <div class="ledger-value">${fmtTokens(item.tokens)}<br/><span style="color:var(--muted)">/ ${fmtTokens(item.target)}</span></div>
    </div>`;
  }).join("");
}

function drawCorpusChart(rows) {
  const canvas=$("#corpus-mini-chart"); if(!canvas)return;
  const dpr=window.devicePixelRatio||1;
  const width=canvas.clientWidth||820, height=canvas.clientHeight||118;
  canvas.width=Math.floor(width*dpr);canvas.height=Math.floor(height*dpr);
  const c=canvas.getContext("2d"); c.scale(dpr,dpr); c.clearRect(0,0,width,height);
  const total=Math.max(1,...rows.map(x=>Number(x.tokens||0)));
  const gap=8, n=Math.max(1,rows.length), bw=(width-gap*(n-1))/n;
  rows.forEach((item,i)=>{
    const h=Math.max(2,(Number(item.tokens||0)/total)*(height-32));
    const x=i*(bw+gap), y=height-18-h;
    c.fillStyle=item.source_exhausted&&!item.complete?"#8d681d":item.complete?"#315b48":"#171613";
    c.fillRect(x,y,bw,h);
    c.fillStyle="#716c63";c.font="8px system-ui";c.textAlign="center";
    c.fillText((item.id||"").replace("_edu","").slice(0,10),x+bw/2,height-5);
  });
}

function renderSystem(info) {
  const mem=info.memory||{}, gpu=info.gpu||{}, disk=info.disk||{}, cpu=info.cpu||{};
  const stats=[
    {label:"GPU",value:gpu.available?`${gpu.utilization?.toFixed(0)||0}%`:"—",detail:gpu.available?`${gpu.name} · ${(gpu.memory_used_mib/1024).toFixed(1)}/${(gpu.memory_total_mib/1024).toFixed(1)} GiB`:"nvidia-smi unavailable"},
    {label:"VRAM / TEMP",value:gpu.available?`${(gpu.memory_used_mib/1024).toFixed(1)}G`:"—",detail:gpu.available?`${gpu.temperature_c}°C · ${gpu.power_w?.toFixed(0)}W / ${gpu.power_limit_w?.toFixed(0)}W`:"—"},
    {label:"HOST MEMORY",value:mem.available_gib!=null?`${mem.available_gib.toFixed(1)}G`:"—",detail:mem.total_gib?`${mem.percent?.toFixed(0)}% used of ${mem.total_gib.toFixed(1)} GiB`:"—"},
    {label:"DISK FREE",value:disk.free_gib!=null?`${disk.free_gib.toFixed(0)}G`:"—",detail:disk.total_gib?`${disk.percent.toFixed(0)}% used · CPU ${cpu.percent?.toFixed(0)||0}%`:"—"},
  ];
  $("#system-stats").innerHTML=stats.map(s=>`<div class="stat"><div class="value">${esc(s.value)}</div><div class="label">${esc(s.label)}</div><div class="detail">${esc(s.detail)}</div></div>`).join("");
}

function activeJobs(jobs) {
  return jobs.filter(j=>["running","stopping"].includes(j.status));
}
function jobCard(job, clickable=false) {
  return `<div class="job-card ${esc(job.status)}" ${clickable?`data-job-id="${esc(job.id)}"`:""}>
    <div><strong>${esc(job.label)}</strong><small>${esc(job.status)} · ${esc(job.resource)} · ${ago(job.started_at)}</small></div>
    <div class="job-actions">${job.status==="running"?`<button class="danger small inline-stop" data-stop-id="${esc(job.id)}">Stop</button>`:""}</div>
  </div>`;
}

function latestRunSummary(runs) {
  if(!runs.length){$("#latest-run-summary").className="run-summary empty-state";$("#latest-run-summary").textContent="No training run metrics yet.";return;}
  const run=runs[0], m=run.latest||{};
  const items=[
    ["RUN",run.name],
    ["TOKENS",fmtTokens(m.tokens_seen)],
    ["LOSS",m.loss!=null?Number(m.loss).toFixed(4):"—"],
    ["TOK/S",m.tokens_per_second!=null?fmtTokens(m.tokens_per_second):"—"],
    ["VRAM",m.cuda_peak_allocated_gb!=null?`${Number(m.cuda_peak_allocated_gb).toFixed(2)}G`:"—"],
  ];
  $("#latest-run-summary").className="run-summary";
  $("#latest-run-summary").innerHTML=items.map(([l,v])=>`<div class="run-metric"><strong>${esc(v)}</strong><span>${l}</span></div>`).join("");
}

function chooseNextAction(o) {
  const jobs=activeJobs(o.jobs||[]);
  if(jobs.length)return ["A job is active.",`${jobs[0].label} is ${jobs[0].status}. Follow it from the live job desk without opening another terminal.`];
  const data=o.datasets||[];
  const incomplete=data.filter(d=>!d.complete && d.tokens>0);
  if(incomplete.length)return ["Decide the data boundary.",`${incomplete.map(x=>`${x.id} ${fmtTokens(x.tokens)}`).join(", ")} are partial. You can train with the materialized amount, replace/fill sources, or resume them independently.`];
  if(!o.capabilities)return ["Audit the runtime.","Run the capability audit before committing to a long GPU experiment."];
  if(!(o.runs||[]).length)return ["Build the clean corpus.","Verify → decontaminate → clean → tokenizer → live architecture gate."];
  return ["Read the latest run.","Open Experiments and inspect loss, throughput, VRAM, routing, gradients and checkpoint milestones."];
}

function renderOverview() {
  const o=S.overview;if(!o)return;
  $("#raw-total").textContent=fmtTokens(o.raw_materialized_tokens);
  renderSystem(o.system||{});
  $("#overview-datasets").innerHTML=datasetLedger(o.datasets||[]);
  $("#data-ledger").innerHTML=datasetLedger(o.datasets||[]);
  drawCorpusChart(o.datasets||[]);
  const jobs=activeJobs(o.jobs||[]);
  $("#overview-jobs").innerHTML=jobs.length?jobs.map(j=>jobCard(j)).join(""):`<div class="empty-state">No Studio job is active.</div>`;
  latestRunSummary(o.runs||[]);
  const [title,copy]=chooseNextAction(o);$("#next-action-title").textContent=title;$("#next-action-copy").textContent=copy;
  const rawTotal=o.raw_materialized_tokens||0;
  const usableTotal=o.latest_clean_tokens||rawTotal;
  $("#plan-available").value=Math.round(usableTotal);
  if(!$("#plan-tokens").dataset.touched) $("#plan-tokens").value=Math.round(usableTotal);
  $("#available-token-badge").textContent=o.latest_clean_tokens
    ? `${fmtTokens(usableTotal)} clean-estimated`
    : `${fmtTokens(rawTotal)} raw materialized`;
  $("#last-refresh").textContent=`updated ${new Date().toLocaleTimeString()}`;
}

function renderPresets() {
  const presets=S.catalog?.presets||{};
  $("#preset-grid").innerHTML=Object.entries(presets).map(([id,p])=>{
    const total=p.sources?Object.values(p.sources).reduce((a,b)=>a+Number(b),0):null;
    return `<div class="preset"><strong>${esc(p.label)}</strong><p>${esc(p.description)}</p><small>${total?fmtTokens(total):"dynamic"}</small><button class="text-button apply-preset" data-preset="${esc(id)}">Apply →</button></div>`;
  }).join("");
}

function dataStatusById(id){return (S.overview?.datasets||[]).find(x=>x.id===id);}
function renderDatasetCatalog() {
  const datasets=S.catalog?.datasets||{};
  $("#dataset-catalog").innerHTML=Object.entries(datasets).map(([id,d])=>{
    const current=dataStatusById(id), currentTokens=current?.tokens||0;
    const status=d.ready?`<span class="status-pill good">wired</span>`:d.gated?`<span class="status-pill warning">gated / validate</span>`:`<span class="status-pill ref">advanced</span>`;
    let suggested=current?.target||d.known_tokens||1_000_000_000;
    return `<article class="dataset-card" data-dataset-card="${esc(id)}">
      <div class="dataset-top"><div><div class="category">${esc(d.category)}</div><h3>${esc(d.label)}</h3></div>${status}</div>
      <p>${esc(d.notes||"")}</p>
      <div class="dataset-current">local: ${fmtTokens(currentTokens)} ${current?`· planned ${fmtTokens(current.target)}`:"· not materialized"}</div>
      <div class="dataset-controls">
        <input class="dataset-target" data-dataset-target="${esc(id)}" type="number" value="${Math.round(suggested)}" step="100000000"/>
        <button class="ink small dataset-download" data-source-id="${esc(id)}">Download / resume</button>
      </div>
    </article>`;
  }).join("");
}

function renderVerifyAndClean() {
  const rows=S.overview?.datasets||[];
  $("#verify-source-list").innerHTML=rows.map(item=>`<div class="check-item"><label><input type="checkbox" class="verify-check" data-path="${esc(item.path)}" checked/><span><strong>${esc(item.label||item.id)}</strong><br/><small>${fmtTokens(item.tokens)} on disk</small></span></label></div>`).join("");
  $("#clean-source-grid").innerHTML=rows.map(item=>{
    const defaultChecked=item.tokens>0 && item.id!=="stack_edu";
    const fim=item.id==="stack_edu"?0.5:0;
    return `<div class="clean-item" data-clean-id="${esc(item.id)}">
      <label><input class="clean-check" type="checkbox" ${defaultChecked?"checked":""}/><span><strong>${esc(item.label||item.id)}</strong> · ${fmtTokens(item.tokens)}</span></label>
      <div class="mini-fields">
        <label>Weight<input class="clean-weight" type="number" value="${Math.max(1,Number(item.tokens||1))}" step="1000000"/></label>
        <label>FIM rate<input class="clean-fim" type="number" value="${fim}" step="0.1" min="0" max="1"/></label>
      </div>
    </div>`;
  }).join("");
}

function catalogStatusClass(status) {
  if(status==="implemented")return "implemented";
  if(status==="reference"||status==="optional-runtime")return status;
  return status||"research-gap";
}
function renderCapabilities(report) {
  const checks=report?.checks||S.catalog?.capability_research||[];
  const summary=report?.summary||{};
  const counts=[
    ["IMPLEMENTED",summary.implemented??checks.filter(x=>x.status==="implemented").length],
    ["REFERENCE",summary.reference??checks.filter(x=>x.status==="reference").length],
    ["OPTIONAL",summary.optional_runtime??checks.filter(x=>x.status==="optional-runtime").length],
    ["GAPS",summary.research_gaps??checks.filter(x=>["research-gap","not-integrated"].includes(x.status)).length],
  ];
  $("#capability-summary").innerHTML=counts.map(([l,v])=>`<div class="stat"><div class="value">${v}</div><div class="label">${l}</div><div class="detail">${report?.gpu?.name||"run live audit for runtime status"}</div></div>`).join("");
  $("#capability-table").innerHTML=checks.map(x=>{
    const runtime=x.runtime_available;
    let status=x.status||"research";
    if(runtime===false && ["implemented","optional-runtime"].includes(status))status+=" / runtime missing";
    return `<article class="cap-card ${catalogStatusClass(x.status)}">
      <div class="eyebrow">${esc(status)}</div>
      <h3>${esc(x.label)}</h3>
      <p>${esc(x.detail||"")}</p>
      ${x.research?`<div class="category">${esc(x.research)}</div>`:""}
    </article>`;
  }).join("");
}

function renderSettings() {
  const s=S.overview?.settings;if(!s)return;
  const d=s.download||{},t=s.training||{};
  $("#set-readers").value=d.parallel_streams;
  $("#set-batch-rows").value=d.parquet_batch_rows;
  $("#set-xet").value=d.xet_concurrency;
  $("#set-rss").value=d.max_rss_gib;
  $("#set-arrow-cpu").value=d.arrow_cpu_threads;
  $("#set-arrow-io").value=d.arrow_io_threads;
  $("#set-zstd").value=d.zstd_threads;
  $("#set-stall").value=d.stall_seconds;
  $("#set-checkpoint-tokens").value=t.checkpoint_tokens;
  $("#set-keep").value=t.keep_last_checkpoints;
}

async function refreshOverview() {
  try {
    S.overview=await api("/api/overview");
    renderOverview();renderDatasetCatalog();renderVerifyAndClean();renderSettings();renderRuns();renderJobs();
    if(S.overview.capabilities)renderCapabilities(S.overview.capabilities);
  } catch(e){showError(e);}
}

async function loadCatalog() {
  S.catalog=await api("/api/catalog");
  renderPresets();renderDatasetCatalog();renderCapabilities(S.overview?.capabilities||null);
}

async function startJob(action,payload={}) {
  const job=await post("/api/job/start",{action,payload});
  toast(`Started: ${job.label}`);
  S.selectedJob=job.id;
  gotoPage("logs");
  await refreshOverview();
  await selectJob(job.id);
  return job;
}
async function stopJob(id) {
  if(!id)return;
  if(!confirm("Send a graceful stop request to this job? Training will resume from its latest durable checkpoint."))return;
  await post("/api/job/stop",{id});toast("Stop requested.");
  await refreshOverview();
}

function presetApply(id) {
  const p=S.catalog?.presets?.[id];if(!p)return;
  if(p.dynamic){
    (S.overview?.datasets||[]).forEach(item=>{
      const input=$(`[data-dataset-target="${CSS.escape(item.id)}"]`);
      if(input)input.value=Math.round(item.tokens||0);
    });
    toast("Targets set to currently materialized amounts.");
    return;
  }
  Object.entries(p.sources||{}).forEach(([sid,t])=>{
    const input=$(`[data-dataset-target="${CSS.escape(sid)}"]`);
    if(input)input.value=t;
  });
  toast(`Applied preset: ${p.label}`);
}

function selectedCleanSources() {
  return $$(".clean-item").filter(n=>$(".clean-check",n).checked).map(n=>{
    const id=n.dataset.cleanId;
    return {id,weight:Number($(".clean-weight",n).value),fim_rate:Number($(".clean-fim",n).value)};
  });
}
async function saveCleanPlan() {
  const body={
    name:$("#clean-name").value,
    output:$("#clean-output").value,
    generated_config:`configs/studio/data/${$("#clean-name").value}_clean.yaml`,
    validation_fraction:Number($("#validation-fraction").value),
    pii_mode:$("#pii-mode").value,
    sources:selectedCleanSources(),
  };
  const result=await post("/api/clean/plan",body);
  S.cleanPlanPath=result.path;
  $("#clean-plan-result").textContent=`Plan: ${result.path}\nGenerated DataConfig: ${body.generated_config}`;
  $("#tokenizer-data").value=body.generated_config;
  $("#plan-data").value=body.generated_config;
  toast("Cleaning plan saved.");
  return result;
}

async function loadConfigs(kind="model") {
  const items=await api(`/api/configs?kind=${encodeURIComponent(kind)}`);
  const target=$("#model-config-list");
  if(!target)return;
  target.innerHTML=items.map(item=>`<div class="config-item" data-config-path="${esc(item.path)}"><strong>${esc(item.name)}</strong><small>${esc(item.path)}</small></div>`).join("");
  target.onclick=(ev)=>{
    const node=ev.target.closest(".config-item");if(!node)return;
    const item=items.find(x=>x.path===node.dataset.configPath);if(!item)return;
    S.selectedConfig=item;
    $("#config-editor-title").textContent=item.name;
    $("#config-save-name").value=`${item.name}_studio`;
    $("#config-kind").value=kind;
    $("#config-editor").value=JSON.stringify(item.config,null,2);
  };
}

function renderRuns() {
  const runs=S.overview?.runs||[];
  const list=$("#run-list");if(!list)return;
  list.innerHTML=runs.length?runs.map(r=>`<div class="run-item ${S.selectedRun===r.path?"active":""}" data-run="${esc(r.path)}"><strong>${esc(r.name)}</strong><small>${fmtTokens(r.latest?.tokens_seen)} · ${r.checkpoint_count} checkpoints · ${ago(r.modified)}</small></div>`).join(""):`<div class="empty-state">No run manifests/metrics yet.</div>`;
}
async function selectRun(path) {
  S.selectedRun=path;renderRuns();
  const [metrics,checkpoints]=await Promise.all([
    api(`/api/metrics?run=${encodeURIComponent(path)}&limit=800`),
    api(`/api/checkpoints?run=${encodeURIComponent(path)}`),
  ]);
  $("#selected-run-title").textContent=path.split("/").pop();
  drawMetricChart(metrics);
  renderMetricSnapshot(metrics);
  renderCheckpoints(checkpoints);
}
function validMetricRows(metrics,key,xkey) {
  return metrics.filter(r=>Number.isFinite(Number(r[key]))&&Number.isFinite(Number(r[xkey]))).map(r=>({x:Number(r[xkey]),y:Number(r[key])}));
}
function drawMetricChart(metrics) {
  const canvas=$("#metrics-chart");if(!canvas)return;
  const key=$("#metric-a").value,xkey=$("#metric-x").value;
  const pts=validMetricRows(metrics,key,xkey);
  const dpr=devicePixelRatio||1,w=canvas.clientWidth||1000,h=canvas.clientHeight||360;
  canvas.width=w*dpr;canvas.height=h*dpr;
  const c=canvas.getContext("2d");c.scale(dpr,dpr);c.clearRect(0,0,w,h);
  c.fillStyle="rgba(255,255,255,.12)";c.fillRect(0,0,w,h);
  c.strokeStyle="#d1c8ba";c.lineWidth=1;
  for(let i=0;i<=5;i++){const y=24+(h-54)*i/5;c.beginPath();c.moveTo(42,y);c.lineTo(w-14,y);c.stroke();}
  if(pts.length<2){c.fillStyle="#716c63";c.font="12px Georgia";c.fillText(`Not enough ${key} points yet.`,50,55);canvas._metrics=metrics;return;}
  let xmin=Math.min(...pts.map(p=>p.x)),xmax=Math.max(...pts.map(p=>p.x)),ymin=Math.min(...pts.map(p=>p.y)),ymax=Math.max(...pts.map(p=>p.y));
  if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;
  const X=x=>42+(x-xmin)/(xmax-xmin)*(w-62),Y=y=>24+(1-(y-ymin)/(ymax-ymin))*(h-54);
  c.strokeStyle="#a5422d";c.lineWidth=2;c.beginPath();
  pts.forEach((p,i)=>i?c.lineTo(X(p.x),Y(p.y)):c.moveTo(X(p.x),Y(p.y)));c.stroke();
  c.fillStyle="#171613";c.font="9px monospace";
  c.fillText(`${key} max ${ymax.toPrecision(5)}`,48,16);
  c.fillText(`${ymin.toPrecision(5)}`,48,h-8);
  c.textAlign="right";c.fillText(`${xkey} ${fmtTokens(xmax)}`,w-15,h-8);c.textAlign="left";
  canvas._metrics=metrics;
}
function renderMetricSnapshot(metrics) {
  const latest=[...metrics].reverse().find(x=>x.loss!=null||x.tokens_per_second!=null)||metrics.at(-1)||{};
  const items=[
    ["tokens",fmtTokens(latest.tokens_seen)],
    ["loss",latest.loss!=null?Number(latest.loss).toFixed(4):"—"],
    ["tokens/s",latest.tokens_per_second!=null?fmtTokens(latest.tokens_per_second):"—"],
    ["VRAM",latest.cuda_peak_allocated_gb!=null?`${Number(latest.cuda_peak_allocated_gb).toFixed(2)}G`:"—"],
    ["ETA",fmtSecs(latest.eta_seconds)],
  ];
  $("#metric-snapshot").innerHTML=items.map(([l,v])=>`<div><strong>${esc(v)}</strong><span>${l}</span></div>`).join("");
}
function renderCheckpoints(rows) {
  $("#checkpoint-table").innerHTML=rows.length?rows.map(x=>`<div class="checkpoint-row"><code>${esc(x.name)}</code><span>${fmtTokens(x.tokens_seen)}</span><span>${esc(x.reason||"—")}</span><span>${x.permanent?'<span class="status-pill good">KEEP</span>':'rolling'}</span><span>${fmtBytes(x.model_bytes)}</span></div>`).join(""):`<div class="empty-state">No checkpoints yet.</div>`;
}

function renderJobs() {
  const jobs=S.overview?.jobs||[];
  const holder=$("#job-history");if(!holder)return;
  holder.innerHTML=jobs.length?jobs.map(j=>jobCard(j,true)).join(""):`<div class="empty-state">No Studio job history yet.</div>`;
}
async function selectJob(id) {
  S.selectedJob=id;
  const result=await api(`/api/job/log?id=${encodeURIComponent(id)}&limit=700`);
  $("#log-title").textContent=result.job.label;
  $("#log-output").textContent=(result.lines||[]).join("");
  $("#log-output").scrollTop=$("#log-output").scrollHeight;
  $("#stop-job").disabled=result.job.status!=="running";
}

async function commonAction(action) {
  if(action==="capability-audit")return startJob("capability_audit",{});
  if(action==="capability-smoke")return startJob("capability_audit",{smoke:true});
  if(action==="hardware-probe")return startJob("hardware_probe",{});
  if(action==="frontier-matrix")return startJob("frontier_matrix",{mode:"quick",steps:3});
}

function bind() {
  $("#nav").addEventListener("click",e=>{const b=e.target.closest("[data-page]");if(b)gotoPage(b.dataset.page);});
  document.addEventListener("click",e=>{const b=e.target.closest("[data-goto]");if(b)gotoPage(b.dataset.goto);});
  document.addEventListener("click",e=>{const b=e.target.closest("[data-action]");if(b)commonAction(b.dataset.action).catch(showError);});
  document.addEventListener("click",e=>{const b=e.target.closest(".inline-stop");if(b){e.stopPropagation();stopJob(b.dataset.stopId).catch(showError);}});
  document.addEventListener("click",e=>{const b=e.target.closest(".apply-preset");if(b)presetApply(b.dataset.preset);});

  $("#dataset-catalog").addEventListener("click",e=>{
    const b=e.target.closest(".dataset-download");if(!b)return;
    const id=b.dataset.sourceId,target=Number($(`[data-dataset-target="${CSS.escape(id)}"]`).value);
    startJob("download_source",{source_id:id,target_tokens:target}).catch(showError);
  });
  $("#custom-download-button").onclick=()=>{
    const id=$("#custom-source-id").value.trim();
    const columns=$("#custom-source-columns").value.split(",").map(x=>x.trim()).filter(Boolean);
    const entry={path:$("#custom-source-path").value.trim(),name:$("#custom-source-name").value.trim()||null,split:$("#custom-source-split").value.trim(),text_field:$("#custom-source-text").value.trim(),shuffle_seed:Number($("#custom-source-seed").value)};
    if(columns.length)entry.columns=columns;
    startJob("download_source",{source_id:id,target_tokens:Number($("#custom-source-target").value),entry}).catch(showError);
  };
  $$(".profile-download").forEach(b=>b.onclick=()=>startJob("download_profile",{profile:b.dataset.profile}).catch(showError));

  $("#verify-selected-last").onclick=async()=>{
    const paths=$$(".verify-check:checked").map(x=>x.dataset.path);
    try{for(const p of paths)await startJob("verify",{path:p,only_last:true});}catch(e){showError(e);}
  };
  $("#verify-selected-full").onclick=async()=>{
    const paths=$$(".verify-check:checked").map(x=>x.dataset.path);
    try{for(const p of paths)await startJob("verify",{path:p,only_last:false});}catch(e){showError(e);}
  };

  $("#make-clean-plan").onclick=()=>saveCleanPlan().catch(showError);
  $("#start-clean").onclick=async()=>{
    try{const result=await saveCleanPlan();await startJob("clean",{plan:result.path});}catch(e){showError(e);}
  };
  $("#train-tokenizer").onclick=()=>startJob("tokenizer",{data:$("#tokenizer-data").value,output:$("#tokenizer-output").value,vocab_size:Number($("#tokenizer-vocab").value),documents:Number($("#tokenizer-documents").value)}).catch(showError);

  $("#config-kind").onchange=()=>loadConfigs($("#config-kind").value).catch(showError);
  $("#save-config").onclick=async()=>{
    try{
      const config=JSON.parse($("#config-editor").value);
      const result=await post("/api/config/save",{kind:$("#config-kind").value,name:$("#config-save-name").value,config});
      toast(`Saved ${result.path}`);await loadConfigs($("#config-kind").value);
    }catch(e){showError(e);}
  };

  $("#plan-tokens").addEventListener("input",()=>$("#plan-tokens").dataset.touched="1");
  $("#generate-plan").onclick=async()=>{
    try{
      const body={name:$("#plan-name").value,tokens:Number($("#plan-tokens").value),available_tokens:Number($("#plan-available").value),data:$("#plan-data").value,checkpoint_tokens:Number($("#plan-checkpoint-tokens").value),allow_repeat:$("#allow-repeat").checked};
      S.trainingPlan=await post("/api/training/plan",body);
      const plan=S.trainingPlan;
      $("#training-plan-result").innerHTML=`<div class="timeline">${(plan.stages||[]).map((s,i)=>`<div class="timeline-step"><span>0${i+1}</span><strong>${Math.round(s.context/1024)}K</strong><p>${fmtTokens(s.tokens)} tokens<br/><code>${esc(s.train_config)}</code></p><button class="text-button use-stage-config" data-train="${esc(s.train_config)}" data-data="${esc(plan.data_config)}" data-init="${esc(s.init_from||"")}">Use stage →</button></div>`).join("")}</div><div class="code-note">Total ${fmtTokens(plan.total_tokens)} · repetition ${plan.repetition_factor?plan.repetition_factor.toFixed(3)+"×":"not declared"}</div>`;
      toast("Training curriculum generated.");
    }catch(e){showError(e);}
  };
  $("#training-plan-result").addEventListener("click",e=>{
    const b=e.target.closest(".use-stage-config");if(!b)return;
    $("#train-config").value=b.dataset.train;$("#train-data").value=b.dataset.data;
    $("#train-init").value=b.dataset.init||"";gotoPage("training");toast("Stage loaded into training desk.");
  });

  $("#training-preflight").onclick=()=>startJob("preflight",{model:$("#train-model").value,train:$("#train-config").value,data:$("#train-data").value,json:"runs/studio-preflight.json"}).catch(showError);
  $("#start-pretrain").onclick=()=>{
    const payload={model:$("#train-model").value,train:$("#train-config").value,data:$("#train-data").value};
    if($("#train-resume").value.trim())payload.resume=$("#train-resume").value.trim();
    else if($("#train-init").value.trim())payload.init_checkpoint=$("#train-init").value.trim();
    if($("#train-hub").value.trim())payload.hub_repo=$("#train-hub").value.trim();
    startJob("train_pretrain",payload).catch(showError);
  };
  $("#start-sft").onclick=()=>startJob("train_sft",{model:$("#train-model").value,train:$("#sft-train").value,data:$("#sft-data").value,checkpoint:$("#sft-checkpoint").value}).catch(showError);
  $("#score-dpo").onclick=()=>startJob("dpo_reference",{model:$("#train-model").value,checkpoint:$("#dpo-checkpoint").value,tokenizer:"artifacts/tokenizer.json",input:$("#dpo-raw").value,output:$("#dpo-data").value,max_length:2048}).catch(showError);
  $("#start-dpo").onclick=()=>startJob("train_dpo",{model:$("#train-model").value,train:$("#dpo-train").value,data:$("#dpo-data").value,checkpoint:$("#dpo-checkpoint").value,max_length:2048}).catch(showError);
  $("#start-reasoning").onclick=()=>startJob("reasoning",{model:$("#train-model").value,checkpoint:$("#reason-checkpoint").value,reasoning:$("#reason-config").value,rl_stop_after:Number($("#reason-cycles").value)}).catch(showError);

  $("#run-list").onclick=e=>{const n=e.target.closest(".run-item");if(n)selectRun(n.dataset.run).catch(showError);};
  $("#metric-a").onchange=async()=>{if(S.selectedRun){const m=await api(`/api/metrics?run=${encodeURIComponent(S.selectedRun)}&limit=800`);drawMetricChart(m);renderMetricSnapshot(m);}};
  $("#metric-x").onchange=$("#metric-a").onchange;
  $("#run-ablations").onclick=()=>startJob("quality_ablations",{data:$("#train-data").value,tokens:100000000}).catch(showError);
  $("#hub-sync").onclick=()=>startJob("hub_sync",{run:$("#hub-sync-run").value,repo:$("#hub-sync-repo").value}).catch(showError);

  $("#run-infer").onclick=()=>startJob("infer",{checkpoint:$("#infer-checkpoint").value,prompt:$("#infer-prompt").value,max_new_tokens:Number($("#infer-max").value),cache_dtype:$("#infer-cache").value,mtp_greedy:false}).catch(showError);
  $("#run-benchmark").onclick=()=>startJob("benchmark",{checkpoint:$("#infer-checkpoint").value,prompt_tokens:8192,new_tokens:Number($("#infer-max").value),cache_dtype:$("#infer-cache").value}).catch(showError);
  $("#run-speculative").onclick=()=>startJob("benchmark_speculative",{checkpoint:$("#infer-checkpoint").value,new_tokens:128}).catch(showError);
  $("#run-cache-benchmark").onclick=()=>startJob("benchmark_cache",{tokens:32768}).catch(showError);
  $("#run-needle").onclick=()=>startJob("needle",{checkpoint:$("#infer-checkpoint").value,lengths:"8192,16384,32768"}).catch(showError);
  $("#run-reason-eval").onclick=()=>startJob("eval_reasoning",{model:$("#eval-reason-model").value,checkpoint:$("#infer-checkpoint").value,data:$("#eval-reason-data").value,samples:Number($("#eval-reason-samples").value),limit:Number($("#eval-reason-limit").value)}).catch(showError);
  $("#export-osp").onclick=()=>startJob("export_osp",{checkpoint:$("#infer-checkpoint").value,output:$("#osp-output").value}).catch(showError);
  $("#export-int4").onclick=()=>startJob("export_torchao",{checkpoint:$("#torchao-input").value,output:$("#torchao-output").value,mode:"int4"}).catch(showError);

  $("#job-history").onclick=e=>{const n=e.target.closest("[data-job-id]");if(n)selectJob(n.dataset.jobId).catch(showError);};
  $("#stop-job").onclick=()=>stopJob(S.selectedJob).catch(showError);

  $("#save-settings").onclick=async()=>{
    try{
      const body={download:{
        parallel_streams:Number($("#set-readers").value),
        parquet_batch_rows:Number($("#set-batch-rows").value),
        xet_concurrency:Number($("#set-xet").value),
        max_rss_gib:Number($("#set-rss").value),
        arrow_cpu_threads:Number($("#set-arrow-cpu").value),
        arrow_io_threads:Number($("#set-arrow-io").value),
        zstd_threads:Number($("#set-zstd").value),
        stall_seconds:Number($("#set-stall").value),
      },training:{
        checkpoint_tokens:Number($("#set-checkpoint-tokens").value),
        keep_last_checkpoints:Number($("#set-keep").value),
      }};
      await post("/api/settings",body);toast("Studio settings saved.");await refreshOverview();
    }catch(e){showError(e);}
  };
}

async function init() {
  bind();
  const page=(location.hash||"#overview").slice(1);gotoPage(PAGE_TITLES[page]?page:"overview");
  try{await Promise.all([loadCatalog(),refreshOverview()]);}catch(e){showError(e);}
  try{await loadConfigs("model");}catch(e){console.warn(e);}
  S.timer=setInterval(async()=>{
    await refreshOverview();
    if(S.selectedJob && $("#page-logs").classList.contains("active")){
      try{await selectJob(S.selectedJob);}catch{}
    }
    if(S.selectedRun && $("#page-experiments").classList.contains("active")){
      try{
        const m=await api(`/api/metrics?run=${encodeURIComponent(S.selectedRun)}&limit=800`);
        drawMetricChart(m);renderMetricSnapshot(m);
      }catch{}
    }
  },2000);
}

window.addEventListener("resize",()=>{
  if(S.overview)drawCorpusChart(S.overview.datasets||[]);
  const canvas=$("#metrics-chart");
  if(canvas?._metrics)drawMetricChart(canvas._metrics);
});
window.addEventListener("DOMContentLoaded",init);
