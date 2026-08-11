"use strict";

const S = {
  overview: null,
  catalog: null,
  selectedJob: null,
  selectedRun: null,
  selectedConfig: null,
  cleanPlanPath: null,
  trainingPlan: null,
  researchRows: [],
  researchTotal: 0,
  researchOffset: 0,
  researchSelected: new Set(),
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
  providers:"Route compute without losing the experiment.",
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
  if(page==="experiments"){
    mountResearchArchive();
    renderRuns();
    loadDiagnostics().catch(showError);
    loadResearchArchive(true).catch(showError);
  }
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
  const gnum=(value,digits=0)=>value==null||!Number.isFinite(Number(value))?"—":Number(value).toFixed(digits);
  const usedGiB=gpu.memory_used_mib==null?null:Number(gpu.memory_used_mib)/1024;
  const totalGiB=gpu.memory_total_mib==null?null:Number(gpu.memory_total_mib)/1024;
  const gpuMemory=(usedGiB==null||totalGiB==null)?"VRAM telemetry unavailable":`${usedGiB.toFixed(1)}/${totalGiB.toFixed(1)} GiB`;
  const power=gpu.power_w==null?"power N/A":`${gnum(gpu.power_w)}W`;
  const limit=gpu.power_limit_w==null?"limit N/A":`${gnum(gpu.power_limit_w)}W limit`;
  const stats=[
    {label:"GPU",value:gpu.available?`${gnum(gpu.utilization)}%`:"—",detail:gpu.available?`${gpu.name} · ${gpuMemory}`:`nvidia-smi unavailable${gpu.error?": "+gpu.error:""}`},
    {label:"VRAM / TEMP",value:gpu.available&&usedGiB!=null?`${usedGiB.toFixed(1)}G`:"—",detail:gpu.available?`${gnum(gpu.temperature_c)}°C · ${power} · ${limit}`:"—"},
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

function renderProviders() {
  const rows=S.overview?.providers||[];
  const grid=$("#provider-grid");
  if(!grid)return;
  for(const id of ["#provider-preferred","#contract-provider"]){
    const select=$(id);
    if(select && !select.querySelector('option[value="gcp"]')){
      const option=document.createElement("option");
      option.value="gcp"; option.textContent="Google Cloud";
      select.appendChild(option);
    }
  }
  grid.innerHTML=rows.map(row=>{
    const state=row.ready?"ready":row.installed?"auth needed":"not installed";
    const cls=row.ready?"good":row.installed?"warning":"ref";
    const profiles=(row.profiles||[]).length?`${row.profiles.length} profile${row.profiles.length===1?"":"s"}`:"no profiles";
    return `<article class="provider-card ${row.ready?"is-ready":""}">
      <div class="provider-card-top"><div><div class="eyebrow">${esc(row.kind)}</div><h3>${esc(row.label)}</h3></div><span class="status-pill ${cls}">${esc(state)}</span></div>
      <p>${esc(row.credit)}</p>
      <div class="provider-facts"><span>${row.installed?"CLI detected":"CLI absent"}</span><span>${esc(profiles)}</span><span>${esc(row.automation)}</span></div>
      ${row.source_url?`<a href="${esc(row.source_url)}" target="_blank" rel="noreferrer">Official terms / pricing ↗</a>`:""}
    </article>`;
  }).join("");
  const aliases=rows.filter(row=>(row.profiles||[]).length);
  $("#provider-profile-ledger").innerHTML=aliases.length?aliases.map(row=>`<div class="profile-row"><strong>${esc(row.label)}</strong><div>${row.profiles.map(profile=>`<span class="profile-chip">${esc(profile)}</span>`).join("")}</div></div>`).join(""):`<div class="empty-state">No provider profile aliases are visible to this WSL environment yet.</div>`;
  const preferred=S.overview?.settings?.providers?.preferred;
  if(preferred && ["modal","gcp","lightning","huggingface_jobs","skypilot"].includes(preferred)) $("#contract-provider").value=preferred;
}

function renderPresets() {
  const presets=S.catalog?.presets||{};
  $("#preset-grid").innerHTML=Object.entries(presets).map(([id,p])=>{
    const total=p.sources?Object.values(p.sources).reduce((a,b)=>a+Number(b),0):null;
    return `<div class="preset"><strong>${esc(p.label)}</strong><p>${esc(p.description)}</p><small>${total?fmtTokens(total):"dynamic"}</small><button class="text-button apply-preset" data-preset="${esc(id)}" ${p.retired?"disabled":""}>${p.retired?"Retired":"Apply →"}</button></div>`;
  }).join("");
}

function dataStatusById(id){return (S.overview?.datasets||[]).find(x=>x.id===id);}
function renderDatasetCatalog() {
  const datasets=S.catalog?.datasets||{};
  $("#dataset-catalog").innerHTML=Object.entries(datasets).map(([id,d])=>{
    const current=dataStatusById(id), currentTokens=current?.tokens||0;
    const status=d.retired?`<span class="status-pill ref">retired</span>`:d.ready?`<span class="status-pill good">wired</span>`:d.gated?`<span class="status-pill warning">gated / validate</span>`:`<span class="status-pill ref">advanced</span>`;
    let suggested=current?.target||d.known_tokens||1_000_000_000;
    return `<article class="dataset-card" data-dataset-card="${esc(id)}">
      <div class="dataset-top"><div><div class="category">${esc(d.category)}</div><h3>${esc(d.label)}</h3></div>${status}</div>
      <p>${esc(d.notes||"")}</p>
      <div class="dataset-current">local: ${fmtTokens(currentTokens)} ${current?`· planned ${fmtTokens(current.target)}`:"· not materialized"}</div>
      <div class="dataset-controls">
        <input class="dataset-target" data-dataset-target="${esc(id)}" type="number" value="${Math.round(suggested)}" step="100000000" ${d.retired?"disabled":""}/>
        <button class="ink small dataset-download" data-source-id="${esc(id)}" ${d.retired?"disabled":""}>${d.retired?"Excluded":"Download / resume"}</button>
      </div>
    </article>`;
  }).join("");
}

function renderVerifyAndClean() {
  const rows=S.overview?.datasets||[];
  $("#verify-source-list").innerHTML=rows.map(item=>`<div class="check-item"><label><input type="checkbox" class="verify-check" data-path="${esc(item.path)}" ${item.retired?"":"checked"}/><span><strong>${esc(item.label||item.id)}</strong><br/><small>${fmtTokens(item.tokens)} on disk${item.retired?" · retired":""}</small></span></label></div>`).join("");
  $("#clean-source-grid").innerHTML=rows.map(item=>{
    const defaultChecked=item.tokens>0 && !item.retired && item.id!=="stack_edu";
    const fim=item.id==="stack_edu"?0.5:0;
    return `<div class="clean-item" data-clean-id="${esc(item.id)}">
      <label><input class="clean-check" type="checkbox" ${defaultChecked?"checked":""} ${item.retired?"disabled":""}/><span><strong>${esc(item.label||item.id)}</strong> · ${fmtTokens(item.tokens)}${item.retired?" · retired":""}</span></label>
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

function renderExecutionBackends(rows=[]) {
  const grid=$("#execution-backend-grid");
  if(!grid)return;
  const labels={aster_local:"Aster local",megatron_core:"Megatron Core",torchtitan:"TorchTitan",deepspeed:"DeepSpeed",probe:"Capability probe"};
  grid.innerHTML=rows.map(row=>{
    let stage="not ready", cls="research-gap";
    if(row.usable){stage="usable",cls="implemented";}
    else if(row.promoted){stage="promoted / topology blocked",cls="optional-runtime";}
    else if(row.adapter_implemented){stage="adapter testing",cls="optional-runtime";}
    else if(row.source_matches_lock){stage="source pinned",cls="reference";}
    const facts=[
      row.importable?`package ${row.installed_version||"detected"}`:"package missing",
      row.source_repository?(row.source_matches_lock?"source commit matched":"source lock unmatched"):"Aster source",
      row.adapter_implemented?"adapter implemented":"adapter missing",
      row.topology_supported?"topology supported":"topology unvalidated",
    ];
    const blockers=(row.blockers||[]).slice(0,3).join(" · ")||"No active blockers.";
    return `<article class="cap-card ${cls}">
      <div class="eyebrow">${esc(stage)}</div>
      <h3>${esc(labels[row.backend]||row.backend)}</h3>
      <div class="provider-facts">${facts.map(f=>`<span>${esc(f)}</span>`).join("")}</div>
      <p>${esc(blockers)}</p>
    </article>`;
  }).join("")||`<div class="empty-state">Execution backend evidence is unavailable.</div>`;
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
  const p=s.providers||{};
  if($("#provider-preferred")) $("#provider-preferred").value=p.preferred||"local";
  if($("#provider-max-spend")) $("#provider-max-spend").value=p.max_spend_usd_per_job??30;
  if($("#provider-confirm-cost")) $("#provider-confirm-cost").checked=p.require_cost_confirmation!==false;
}

async function refreshOverview() {
  try {
    S.overview=await api("/api/overview");
    renderOverview();renderDatasetCatalog();renderVerifyAndClean();renderSettings();renderProviders();renderExecutionBackends(S.overview.execution_backends||[]);renderRuns();renderJobs();
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
  if(p.retired){toast("That historical preset is retired and cannot be applied.");return;}
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

function mountResearchArchive() {
  if($("#research-summary"))return;
  const layout=$("#page-experiments .experiment-layout");if(!layout)return;
  layout.insertAdjacentHTML("beforebegin",`<article class="paper-panel research-archive-panel">
    <div class="panel-head"><div><div class="eyebrow">PERMANENT RESEARCH INDEX</div><h3>Every result, comparable and traceable</h3></div><button id="research-reindex" class="ghost small">Reindex now</button></div>
    <p class="muted">The searchable index has unbounded retention. Pagination limits only this view; raw artifacts, revisions, protocols, failures and evidence paths remain preserved.</p>
    <div id="research-summary" class="research-summary"><div class="empty-state">Loading archive summaryâ€¦</div></div>
    <div class="research-toolbar">
      <label>Search<input id="research-query" placeholder="candidate, campaign, config or path"/></label>
      <label>Backend<select id="research-backend"><option value="">All backends</option></select></label>
      <label>Status<select id="research-status"><option value="">All statuses</option></select></label>
      <button id="research-filter" class="ink">Apply</button><button id="research-compare" class="ghost" disabled>Compare selected</button>
    </div>
    <div id="research-compatibility" class="research-compatibility"></div>
    <div class="research-table-wrap"><table class="research-table"><thead><tr><th></th><th>Trial / provenance</th><th>Protocol</th><th>tokens/s</th><th>Eval loss</th><th>Time to target</th><th>GPU</th><th>Peak VRAM</th><th>Parameters</th></tr></thead><tbody id="research-trial-rows"><tr><td colspan="9" class="empty-state">Loading indexed trialsâ€¦</td></tr></tbody></table></div>
    <div class="research-pagination"><span id="research-range" class="muted"></span><button id="research-more" class="ghost small">Load more</button></div>
  </article>
  <div class="research-lower-grid">
    <article class="paper-panel"><div class="panel-head"><div><div class="eyebrow">MATCHED COMPARISON</div><h3>Systems evidence without false equivalence</h3></div></div><div id="research-comparison" class="research-comparison empty-state">Select two or more indexed trials to compare throughput, utilization, VRAM and protocol compatibility.</div></article>
    <article class="paper-panel"><div class="panel-head"><div><div class="eyebrow">FINDINGS LEDGER</div><h3>Decisions with evidence</h3></div></div><div id="research-findings" class="research-findings"><div class="empty-state">Loading findingsâ€¦</div></div></article>
  </div>`);
  $("#research-filter").onclick=()=>loadResearchArchive(true).catch(showError);
  $("#research-query").onkeydown=e=>{if(e.key==="Enter")loadResearchArchive(true).catch(showError);};
  $("#research-more").onclick=()=>loadResearchArchive(false).catch(showError);
  $("#research-compare").onclick=()=>loadResearchComparison().catch(showError);
  $("#research-reindex").onclick=async()=>{
    const button=$("#research-reindex");button.disabled=true;button.textContent="Indexingâ€¦";
    try{await post("/api/research/reindex",{});await loadResearchArchive(true);toast("Research archive reindexed.");}
    finally{button.disabled=false;button.textContent="Reindex now";}
  };
  $("#research-trial-rows").onchange=e=>{
    const box=e.target.closest(".research-select");if(!box)return;
    if(box.checked)S.researchSelected.add(box.dataset.trialId);else S.researchSelected.delete(box.dataset.trialId);
    $("#research-compare").disabled=S.researchSelected.size<2;
    $("#research-compare").textContent=S.researchSelected.size?`Compare selected (${S.researchSelected.size})`:"Compare selected";
  };
}

function renderResearchSummary(summary) {
  const fastest=summary.fastest_observed;
  const cards=[
    [fmtTokens(summary.trials),"indexed trials"],
    [fmtTokens(summary.artifact_revisions),"artifact revisions"],
    [fmtTokens(summary.findings),"durable findings"],
    [fastest?.tokens_per_second?fmtTokens(fastest.tokens_per_second):"â€”","fastest observed tok/s"],
    [summary.last_indexed?ago(summary.last_indexed):"never",summary.refresh_in_progress?"refreshing in background":"index freshness"],
  ];
  $("#research-summary").innerHTML=cards.map(([value,label])=>`<div><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("");
  const backend=$("#research-backend"),status=$("#research-status");
  const selectedBackend=backend.value,selectedStatus=status.value;
  backend.innerHTML='<option value="">All backends</option>'+(summary.backends||[]).map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join("");
  status.innerHTML='<option value="">All statuses</option>'+(summary.statuses||[]).map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join("");
  backend.value=selectedBackend;status.value=selectedStatus;
}

function researchTrialHtml(row) {
  const checked=S.researchSelected.has(row.id)?"checked":"";
  const protocol=[row.sequence_length?`${fmtTokens(row.sequence_length)} ctx`:null,row.micro_batch_size!=null?`mb${row.micro_batch_size}`:null,row.gradient_accumulation!=null?`acc${row.gradient_accumulation}`:null,row.optimizer].filter(Boolean).join(" Â· ");
  const params=row.total_parameters?`${fmtTokens(row.total_parameters)} total / ${fmtTokens(row.active_parameters)} active`:"â€”";
  const loss=row.eval_loss??row.loss;
  const lossSpread=row.eval_loss_stdev!=null?` +/- ${Number(row.eval_loss_stdev).toFixed(4)}`:"";
  return `<tr>
    <td><input class="research-select" type="checkbox" data-trial-id="${esc(row.id)}" ${checked}/></td>
    <td><strong>${esc(row.variant||row.name)}</strong><small>${esc(row.backend||row.status)} Â· ${esc((row.git_commit||"unbound").slice(0,8))}</small><code title="${esc(row.matrix_path)}">${esc(row.matrix_path)}</code></td>
    <td><span>${esc(protocol||"â€”")}</span><small>${row.gradient_checkpointing?`checkpoint seg ${esc(row.checkpoint_segment_size)}`:"no activation checkpoint"}</small></td>
    <td class="numeric">${row.tokens_per_second!=null?fmtTokens(row.tokens_per_second):"â€”"}</td>
    <td class="numeric">${loss!=null?`${Number(loss).toFixed(4)}${lossSpread}`:"â€”"}</td>
    <td class="numeric">${row.time_to_common_loss!=null?`${Number(row.time_to_common_loss).toFixed(1)} s`:"â€”"}</td>
    <td class="numeric">${row.gpu_utilization!=null?`${Number(row.gpu_utilization).toFixed(1)}%`:"â€”"}</td>
    <td class="numeric">${row.peak_allocated_gib!=null?`${Number(row.peak_allocated_gib).toFixed(2)} GiB`:"â€”"}</td>
    <td><span>${esc(params)}</span><small>${esc(row.gpu_name||"hardware unrecorded")}</small></td>
  </tr>`;
}

function renderResearchTrials(append=false) {
  const body=$("#research-trial-rows");
  if(!append)body.innerHTML="";
  body.insertAdjacentHTML("beforeend",S.researchRows.slice(append?S.researchOffset:0).map(researchTrialHtml).join(""));
  if(!S.researchRows.length)body.innerHTML='<tr><td colspan="9" class="empty-state">No indexed trials match these filters.</td></tr>';
  $("#research-range").textContent=`Showing ${S.researchRows.length} of ${S.researchTotal} matching trials`;
  $("#research-more").hidden=S.researchRows.length>=S.researchTotal;
}

function renderResearchFindings(payload) {
  $("#research-findings").innerHTML=payload.rows.length?payload.rows.map(item=>`<div class="research-finding">
    <div><span class="status-pill ${item.status==="confirmed"?"good":"warning"}">${esc(item.status)}</span><time>${esc(item.created_utc?.slice(0,10)||"")}</time></div>
    <strong>${esc(item.title)}</strong><p>${esc(item.summary)}</p>
    <div class="finding-tags">${item.tags.map(tag=>`<span>${esc(tag)}</span>`).join("")}</div>
    ${item.evidence.map(path=>`<code title="${esc(path)}">${esc(path)}</code>`).join("")}
  </div>`).join(""):'<div class="empty-state">No findings have been recorded yet.</div>';
}

async function loadResearchArchive(reset=true) {
  mountResearchArchive();
  if(reset){S.researchOffset=0;S.researchRows=[];}
  const params=new URLSearchParams({limit:"100",offset:String(S.researchRows.length),query:$("#research-query").value.trim(),backend:$("#research-backend").value,status:$("#research-status").value});
  const calls=[api(`/api/research/trials?${params}`)];
  if(reset)calls.push(api("/api/research/summary"),api("/api/research/findings?limit=100"));
  const [trials,summary,findings]=await Promise.all(calls);
  const oldLength=S.researchRows.length;
  S.researchRows.push(...trials.rows);S.researchTotal=trials.total;S.researchOffset=oldLength;
  renderResearchTrials(oldLength>0);
  if(summary)renderResearchSummary(summary);
  if(findings)renderResearchFindings(findings);
}

function comparisonBar(row,key,max,label,unit) {
  const value=Number(row[key]);const width=Number.isFinite(value)&&max>0?Math.max(1,value/max*100):0;
  return `<div class="comparison-bar-row"><code>${esc(row.variant||row.name)}</code><div class="comparison-track"><span style="width:${width}%"></span></div><strong>${Number.isFinite(value)?`${value.toLocaleString(undefined,{maximumFractionDigits:2})}${unit}`:"â€”"}</strong><small>${esc(label)}</small></div>`;
}

async function loadResearchComparison() {
  const ids=[...S.researchSelected];if(ids.length<2)return;
  const result=await api(`/api/research/compare?ids=${encodeURIComponent(ids.join(","))}`);
  const mismatches=result.dimensions.filter(x=>!x.match&&x.role!=="treatment");
  const treatments=result.dimensions.filter(x=>!x.match&&x.role==="treatment"),holder=$("#research-comparison");
  const metrics=[
    ["tokens_per_second","tokens/s (higher is better)",""],
    ["eval_loss","eval loss (lower is better)",""],
    ["token_curve_auc","token-curve loss AUC (lower is better)",""],
    ["equal_wall_loss","equal-wall loss (lower is better)",""],
    ["time_to_common_loss","time to common quality (lower is better)"," s"],
    ["tokens_to_common_loss","tokens to common quality (lower is better)",""],
    ["gpu_utilization","GPU utilization (higher is better)","%"],
    ["peak_allocated_gib","peak VRAM (lower is better)"," GiB"],
  ];
  const charts=metrics.map(([key,label,unit])=>{
    const max=Math.max(0,...result.trials.map(x=>Number(x[key])||0));
    return `<section><h4>${esc(label)}</h4>${result.trials.map(row=>comparisonBar(row,key,max,label,unit)).join("")}</section>`;
  }).join("");
  const comparisonKind=result.comparison_kind||(result.strictly_comparable?"matched_protocol":"protocol_mismatch");
  const verdict={
    matched_protocol:"Matched systems protocol",
    controlled_treatment:"Controlled treatment comparison",
    protocol_mismatch:"Protocol mismatch detected",
  }[comparisonKind]||"Comparison status unavailable";
  holder.className="research-comparison";
  holder.innerHTML=`<div class="comparison-verdict ${comparisonKind==="protocol_mismatch"?"mismatch":"matched"}"><strong>${esc(verdict)}</strong><span>${result.same_model?"same model fingerprint":"different model fingerprints"}</span></div>
    ${treatments.length?`<div class="compatibility-chips">${treatments.map(x=>`<span>${esc(x.label)} is the treatment</span>`).join("")}</div>`:""}
    ${mismatches.length?`<div class="compatibility-chips">${mismatches.map(x=>`<span>${esc(x.label)} differs unexpectedly</span>`).join("")}</div>`:""}${charts}<p class="muted">${esc(result.note)}</p>`;
}

function renderRuns() {
  const runs=S.overview?.runs||[];
  const list=$("#run-list");if(!list)return;
  list.innerHTML=runs.length?runs.map(r=>`<div class="run-item ${S.selectedRun===r.path?"active":""}" data-run="${esc(r.path)}"><strong>${esc(r.name)}</strong><small>${esc(r.status||"legacy")} · ${esc(r.provider||"local")} · ${fmtTokens(r.latest?.tokens_seen||r.experiment?.completed_tokens)} · ${r.checkpoint_count} checkpoints · ${ago(r.modified)}</small>${r.run_id?`<code>${esc(r.run_id)}</code>`:""}</div>`).join(""):`<div class="empty-state">No run manifests/metrics yet.</div>`;
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

async function loadDiagnostics() {
  const holder=$("#diagnostic-matrix-list");if(!holder)return;
  const rows=await api("/api/diagnostics?limit=60");
  holder.innerHTML=rows.length?rows.map(matrix=>{
    const variants=Object.entries(matrix.aggregate||{}).map(([name,value])=>{
      const tps=Number(value?.median_tokens_per_second);
      const util=Number(value?.median_gpu_utilization);
      const reps=value?.successful_repetitions;
      return `<div class="diagnostic-variant"><code>${esc(name)}</code><span>${Number.isFinite(tps)?`${fmtTokens(tps)} tok/s`:"—"}</span><span>${Number.isFinite(util)?`${util.toFixed(1)}% GPU`:"—"}</span><span>${reps!=null?`${esc(reps)} reps`:"—"}</span></div>`;
    }).join("");
    const failure=matrix.failures?.length?`<span class="status-pill warning">${matrix.failures.length} failed</span>`:`<span class="status-pill good">${matrix.successful_trials}/${matrix.trial_count} ok</span>`;
    return `<details class="diagnostic-matrix"><summary><span><strong>${esc(matrix.name)}</strong><small>${ago(matrix.modified)} · ${esc((matrix.git_commit||"").slice(0,8)||"unbound")}</small></span>${failure}</summary><div class="diagnostic-body">${variants||'<div class="empty-state">Matrix is still running or has no aggregate.</div>'}<code class="diagnostic-path">${esc(matrix.path)}</code></div></details>`;
  }).join(""):`<div class="empty-state">No matrix.json diagnostics found under runs/.</div>`;
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
  const auxiliaryList=$("#page-data .action-list");
  if(auxiliaryList&&!auxiliaryList.querySelector('[data-profile="posttrain-modern"]')){
    auxiliaryList.insertAdjacentHTML("beforeend",'<div><strong>Modern post-training candidates</strong><span>LongAlign 64K / OpenThoughts3 / modern preference / tool-use</span><button data-profile="posttrain-modern" class="ghost small profile-download">Download</button></div><div><strong>Quarantined agent traces</strong><span>2026 Agent SFT; isolated from general assistant style</span><button data-profile="posttrain-agent" class="ghost small profile-download">Download</button></div>');
  }
  $("#nav").addEventListener("click",e=>{const b=e.target.closest("[data-page]");if(b)gotoPage(b.dataset.page);});
  document.addEventListener("click",e=>{const b=e.target.closest("[data-goto]");if(b)gotoPage(b.dataset.goto);});
  document.addEventListener("click",e=>{const b=e.target.closest("[data-action]");if(b)commonAction(b.dataset.action).catch(showError);});
  document.addEventListener("click",e=>{const b=e.target.closest(".inline-stop");if(b){e.stopPropagation();stopJob(b.dataset.stopId).catch(showError);}});
  document.addEventListener("click",e=>{const b=e.target.closest(".apply-preset");if(b)presetApply(b.dataset.preset);});
  document.addEventListener("click",e=>{const b=e.target.closest(".runtime-install");if(!b)return;const profile=b.dataset.runtimeProfile;if(!confirm(`Install Aster runtime profile '${profile}'? Studio will dry-run pip first and refuse any plan that replaces the current PyTorch build.`))return;startJob("runtime_setup",{profile}).catch(showError);});

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
  $("#save-provider-policy").onclick=async()=>{
    try{
      await post("/api/settings",{providers:{
        preferred:$("#provider-preferred").value,
        max_spend_usd_per_job:Number($("#provider-max-spend").value),
        require_cost_confirmation:$("#provider-confirm-cost").checked,
      }});
      toast("Compute dispatch policy saved.");
      await refreshOverview();
    }catch(e){showError(e);}
  };
  $("#create-provider-contract").onclick=async()=>{
    try{
      const contract=await post("/api/provider/contract",{
        provider:$("#contract-provider").value,
        profile_alias:$("#contract-profile").value.trim(),
        model:$("#contract-model").value.trim(),
        train:$("#contract-train").value.trim(),
        data:$("#contract-data").value.trim(),
        hub_repo:$("#contract-hub-repo").value.trim(),
        timeout_minutes:Number($("#contract-timeout").value),
        estimated_spend_usd:Number($("#contract-spend").value),
        cost_confirmed:$("#contract-cost-confirmed").checked,
      });
      $("#provider-contract-result").textContent=JSON.stringify(contract,null,2);
      toast(`Contract ${contract.contract_id} is ${contract.status}.`);
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
