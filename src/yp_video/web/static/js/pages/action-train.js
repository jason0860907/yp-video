/**
 * Action Train page - SPOT training for action labels.
 */
import {
  api, API, SSEClient, card, pageHeader, sectionTitle, btnSmall, emptyState,
  escapeHtml, showToast, inputCls, selectCls, renderJobProgress,
} from '../shared.js';

let videos = [];
let status = null;
let source = 'vnl_1_5';
let job = null;
let client = null;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Action Train', 'Train SPOT action labels from JSONL datasets', `
        ${btnSmall('Export JSONL', 'id="act-train-export" title="Download action annotation dataset"')}
        ${btnSmall('Open Label', 'id="act-train-label" title="Open Action Label"', 'primary')}
      `)}

      <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_27rem] gap-5">
        ${card(`
          <div class="space-y-4">
            ${sectionTitle('Datasets', '', `${btnSmall('Refresh', 'id="act-train-refresh"')}`)}
            <div id="act-train-stats" class="grid grid-cols-2 md:grid-cols-4 gap-3"></div>
            <div id="act-train-source-summary" class="rounded-xl border border-border bg-surface-100/35 p-3"></div>
            <div id="act-train-videos" class="space-y-1 max-h-[48vh] overflow-y-auto pr-1"></div>
          </div>
        `)}

        ${card(`
          <div class="space-y-4">
            ${sectionTitle('Training', '', `<span id="act-train-ready" class="text-[11px] text-text-muted"></span>`)}

            <div class="space-y-3">
              <label class="block space-y-1.5">
                <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Source</span>
                <select id="act-train-source" class="${selectCls} w-full">
                  <option value="vnl_1_5">VNL 1.5 JSONL</option>
                  <option value="action_annotations">YP Action Labels</option>
                </select>
              </label>

              <div class="grid grid-cols-1 gap-2">
                <label class="block space-y-1.5">
                  <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Dataset</span>
                  <input id="act-train-dataset" class="${inputCls} w-full" value="vnl_1.5">
                </label>
                <label class="block space-y-1.5">
                  <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Frames</span>
                  <input id="act-train-frame-dir" class="${inputCls} w-full font-mono text-[11px]" value="data/vnl_1.5/frames_224p">
                </label>
                <label class="block space-y-1.5">
                  <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Output</span>
                  <input id="act-train-save-dir" class="${inputCls} w-full font-mono text-[11px]" placeholder="auto: ~/yp-spot/exp/...">
                </label>
                <label class="block space-y-1.5">
                  <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Init Checkpoint</span>
                  <input id="act-train-init-checkpoint" class="${inputCls} w-full font-mono text-[11px]" value="exp/vnl15_official_150/checkpoint_best.pt">
                </label>
              </div>

              <div class="grid grid-cols-2 gap-2">
                ${field('Feature', 'act-train-feature', select(`
                  <option value="rny008_gsm">rny008_gsm</option>
                  <option value="rny002_gsm">rny002_gsm</option>
                  <option value="convnextt_gsm">convnextt_gsm</option>
                  <option value="rn18_gsm">rn18_gsm</option>
                `))}
                ${field('Temporal', 'act-train-temporal', select(`
                  <option value="gru">gru</option>
                  <option value="deeper_gru">deeper_gru</option>
                  <option value="mstcn">mstcn</option>
                  <option value="asformer">asformer</option>
                `))}
                ${field('Epochs', 'act-train-epochs', input('number', '150', '1', '1000'))}
                ${field('Batch', 'act-train-batch', input('number', '8', '1', '64'))}
                ${field('Clip Len', 'act-train-clip', input('number', '64', '8', '256'))}
                ${field('Workers', 'act-train-workers', input('number', '4', '0', '32'))}
                ${field('GPU', 'act-train-gpu', input('number', '0', '0', '7'))}
                ${field('LR', 'act-train-lr', input('number', '0.001', '0', '', 'any'))}
                ${field('Warmup', 'act-train-warmup', input('number', '3', '0', '100'))}
                ${field('Criterion', 'act-train-criterion', select(`
                  <option value="map">map</option>
                  <option value="loss">loss</option>
                `))}
                ${field('Start Val', 'act-train-start-val', input('number', '0', '0', '1000'))}
                ${field('Epoch Frames', 'act-train-epoch-frames', input('number', '', '1', '', '1', 'optional'))}
              </div>

              <div class="grid grid-cols-2 gap-2" id="act-train-split-fields">
                ${field('Val Ratio', 'act-train-val-ratio', input('number', '0.2', '0.01', '0.9', '0.01'))}
                ${field('Split Seed', 'act-train-split-seed', input('number', '42'))}
              </div>

              <div class="flex flex-wrap items-center gap-3 text-xs text-text-secondary">
                <label class="inline-flex items-center gap-2 cursor-pointer">
                  <input id="act-train-predict-location" type="checkbox" checked class="accent-primary w-3.5 h-3.5">
                  Predict location
                </label>
                <label class="inline-flex items-center gap-2 cursor-pointer">
                  <input id="act-train-stop-vllm" type="checkbox" class="accent-primary w-3.5 h-3.5">
                  Stop vLLM
                </label>
              </div>

              <div class="flex items-center gap-2">
                <button id="act-train-start" class="flex-1 bg-primary hover:bg-primary-light text-white border border-primary-light/30 px-3 py-2 rounded-lg text-xs font-medium transition-colors duration-150">Start Training</button>
                <button id="act-train-cancel" class="hidden bg-white/[0.06] text-text-secondary hover:text-text-primary hover:bg-white/[0.10] border border-border hover:border-border-light px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all duration-200">Cancel</button>
              </div>
            </div>

            <div id="act-train-job" class="hidden rounded-xl border border-border bg-surface-100/35 p-3 space-y-2"></div>
          </div>
        `)}
      </div>
    </div>`;

  document.getElementById('act-train-refresh').addEventListener('click', loadData);
  document.getElementById('act-train-export').addEventListener('click', exportDataset);
  document.getElementById('act-train-label').addEventListener('click', () => { window.location.hash = '#/action-annotate'; });
  document.getElementById('act-train-source').addEventListener('change', (e) => {
    source = e.target.value;
    applySourceDefaults();
    renderDataset();
  });
  document.getElementById('act-train-start').addEventListener('click', startTraining);
  document.getElementById('act-train-cancel').addEventListener('click', cancelTraining);
  loadData();
}

export function activate() {}

export function deactivate() {
  client?.stop();
  client = null;
}

function field(label, id, control) {
  return `
    <label class="block space-y-1.5 min-w-0">
      <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">${label}</span>
      ${control.replace('__ID__', id)}
    </label>`;
}

function input(type, value, min = '', max = '', step = '1', placeholder = '') {
  return `<input id="__ID__" type="${type}" value="${escapeHtml(value)}" ${min !== '' ? `min="${min}"` : ''} ${max !== '' ? `max="${max}"` : ''} step="${step}" placeholder="${escapeHtml(placeholder)}" class="${inputCls} w-full">`;
}

function select(options) {
  return `<select id="__ID__" class="${selectCls} w-full">${options}</select>`;
}

async function loadData() {
  try {
    const [videoData, statusData] = await Promise.all([
      api(API.actionAnnotate.videos),
      api(API.actionTrain.status),
    ]);
    videos = videoData;
    status = statusData;
    if (status?.active_job) {
      job = status.active_job;
      if (job.params?.source) source = job.params.source;
      subscribeTrainingJob(job.id);
    }
    applySourceDefaults();
    renderDataset();
    renderJob();
  } catch (e) {
    showToast(`Failed to load action training state: ${e.message}`, 'error');
  }
}

function applySourceDefaults() {
  document.getElementById('act-train-source').value = source;
  const datasetEl = document.getElementById('act-train-dataset');
  const framesEl = document.getElementById('act-train-frame-dir');
  const initEl = document.getElementById('act-train-init-checkpoint');
  const splitEl = document.getElementById('act-train-split-fields');
  const defaultInit = status?.default_init_checkpoint || 'exp/vnl15_official_150/checkpoint_best.pt';
  if (initEl && (!initEl.value || initEl.value === 'exp/vnl15_official_150/checkpoint_best.pt')) {
    initEl.value = defaultInit;
  }
  if (source === 'vnl_1_5') {
    datasetEl.value = 'vnl_1.5';
    framesEl.value = 'data/vnl_1.5/frames_224p';
    splitEl.classList.add('hidden');
  } else {
    datasetEl.value = 'yp_actions';
    framesEl.value = '~/videos/action-frames';
    splitEl.classList.remove('hidden');
  }
  updateStartState();
}

function datasetStats() {
  return videos.reduce((acc, video) => {
    const count = Math.max(0, Number(video.event_count) || 0);
    if (video.has_action_annotation) acc.videos += 1;
    acc.actions += count;
    if (video.kind === 'broadcast') acc.broadcast += count;
    if (video.kind === 'sideline') acc.sideline += count;
    return acc;
  }, { videos: 0, actions: 0, broadcast: 0, sideline: 0 });
}

function sourceReady() {
  if (source === 'vnl_1_5') return Boolean(status?.vnl_1_5?.ready);
  return datasetStats().actions > 0;
}

function renderDataset() {
  const stats = datasetStats();
  const vnl = status?.vnl_1_5 || {};
  const ready = sourceReady();
  document.getElementById('act-train-ready').textContent = ready ? 'ready' : 'not ready';
  document.getElementById('act-train-stats').innerHTML = [
    statCell('Action Videos', stats.videos),
    statCell('Action Labels', stats.actions),
    statCell('VNL Train', vnl.train_events || 0),
    statCell('VNL Val', vnl.val_events || 0),
  ].join('');

  const summary = document.getElementById('act-train-source-summary');
  summary.innerHTML = source === 'vnl_1_5' ? vnlSummary(vnl) : actionSummary(stats);

  const list = document.getElementById('act-train-videos');
  const labeled = videos.filter(v => v.has_action_annotation);
  if (!labeled.length) {
    list.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>',
      'No action labels',
      '',
    );
    updateStartState();
    return;
  }

  list.innerHTML = labeled
    .sort((a, b) => b.event_count - a.event_count || a.name.localeCompare(b.name))
    .map(video => `
      <button type="button" class="w-full flex items-center gap-3 px-3 py-2 rounded-lg border border-border bg-white/[0.035] hover:bg-white/[0.06] text-left transition-colors duration-150" data-video="${escapeHtml(video.name)}">
        <span class="w-2 h-2 rounded-full ${video.kind === 'broadcast' ? 'bg-primary-light' : 'bg-accent-light'} flex-shrink-0"></span>
        <span class="min-w-0 flex-1 truncate text-sm text-text-primary">${escapeHtml(video.name)}</span>
        <span class="text-[11px] text-text-muted font-heading tabular-nums">${video.event_count} action</span>
      </button>
    `).join('');

  list.querySelectorAll('[data-video]').forEach(btn => {
    btn.addEventListener('click', () => {
      window.location.hash = '#/action-annotate';
    });
  });
  updateStartState();
}

function statCell(label, value) {
  return `
    <div class="rounded-xl border border-border bg-surface-100/45 p-3">
      <div class="text-lg font-heading font-semibold text-text-primary tabular-nums">${value}</div>
      <div class="text-[11px] text-text-muted mt-0.5">${label}</div>
    </div>`;
}

function vnlSummary(vnl) {
  const rows = [
    ['Train', `${vnl.train_videos || 0} videos / ${vnl.train_events || 0} events`],
    ['Val', `${vnl.val_videos || 0} videos / ${vnl.val_events || 0} events`],
    ['Frames', vnl.frame_dir_exists ? vnl.frame_dir : 'missing'],
    ['Init', status?.default_init_checkpoint || 'missing'],
  ];
  return summaryRows(rows, vnl.ready);
}

function actionSummary(stats) {
  const rows = [
    ['Labels', `${stats.videos} videos / ${stats.actions} events`],
    ['Source', status?.action_annotations?.label_dir || '~/videos/action-annotations'],
    ['Frames', document.getElementById('act-train-frame-dir')?.value || '~/videos/action-frames'],
  ];
  return summaryRows(rows, stats.actions > 0);
}

function summaryRows(rows, ok) {
  return `
    <div class="flex items-center justify-between gap-3 mb-2">
      <span class="text-xs font-heading text-text-primary">${source === 'vnl_1_5' ? 'VNL 1.5 JSONL' : 'YP Action Labels'}</span>
      <span class="text-[10px] px-2 py-0.5 rounded-full border ${ok ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/30 bg-amber-500/10 text-amber-300'}">${ok ? 'ready' : 'not ready'}</span>
    </div>
    <div class="space-y-1">
      ${rows.map(([label, value]) => `
        <div class="flex items-center gap-3 text-[11px]">
          <span class="w-14 text-text-muted">${escapeHtml(label)}</span>
          <span class="min-w-0 flex-1 truncate font-heading text-text-secondary tabular-nums" title="${escapeHtml(value)}">${escapeHtml(value)}</span>
        </div>`).join('')}
    </div>`;
}

function updateStartState() {
  const btn = document.getElementById('act-train-start');
  if (!btn) return;
  const running = job && ['pending', 'running'].includes(job.status);
  const disabled = running || !sourceReady() || !status?.spot_available;
  btn.disabled = disabled;
  btn.className = disabled
    ? 'flex-1 bg-white/[0.04] text-text-muted border border-border px-3 py-2 rounded-lg text-xs font-medium cursor-not-allowed'
    : 'flex-1 bg-primary hover:bg-primary-light text-white border border-primary-light/30 px-3 py-2 rounded-lg text-xs font-medium transition-colors duration-150';
  btn.textContent = running ? 'Training...' : 'Start Training';
}

function exportDataset() {
  const stats = datasetStats();
  if (!stats.actions) {
    showToast('No saved action annotations to export yet', 'warning');
    return;
  }
  window.location.href = `/api${API.actionAnnotate.export}`;
}

function numberValue(id, fallback = null) {
  const raw = document.getElementById(id)?.value;
  if (raw == null || raw === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function textValue(id, fallback = '') {
  const value = document.getElementById(id)?.value?.trim();
  return value || fallback;
}

async function startTraining() {
  const btn = document.getElementById('act-train-start');
  btn.disabled = true;
  try {
    const body = {
      source,
      dataset: textValue('act-train-dataset'),
      frame_dir: textValue('act-train-frame-dir'),
      save_dir: textValue('act-train-save-dir', null),
      init_checkpoint: textValue('act-train-init-checkpoint', null),
      gpu: numberValue('act-train-gpu', 0),
      feature_arch: textValue('act-train-feature', 'rny008_gsm'),
      temporal_arch: textValue('act-train-temporal', 'gru'),
      clip_len: numberValue('act-train-clip', 64),
      batch_size: numberValue('act-train-batch', 8),
      num_epochs: numberValue('act-train-epochs', 150),
      warm_up_epochs: numberValue('act-train-warmup', 3),
      learning_rate: numberValue('act-train-lr', 0.001),
      num_workers: numberValue('act-train-workers', 4),
      criterion: textValue('act-train-criterion', 'map'),
      start_val_epoch: numberValue('act-train-start-val', 0),
      epoch_num_frames: numberValue('act-train-epoch-frames', null),
      val_ratio: numberValue('act-train-val-ratio', 0.2),
      split_seed: numberValue('act-train-split-seed', 42),
      predict_location: document.getElementById('act-train-predict-location').checked,
      stop_vllm: document.getElementById('act-train-stop-vllm').checked,
    };
    job = await api(API.actionTrain.start, { method: 'POST', body });
    showToast('Action training started', 'success');
    renderJob();
    subscribeTrainingJob(job.id);
  } catch (e) {
    showToast(`Action training failed to start: ${e.message}`, 'error');
  } finally {
    updateStartState();
  }
}

function subscribeTrainingJob(jobId) {
  client?.stop();
  client = new SSEClient(API.jobs.eventsSSE(jobId), {
    onMessage: (data) => {
      job = data;
      renderJob();
      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        client?.stop();
        client = null;
        if (data.status === 'completed') showToast('Action training complete', 'success');
        if (data.status === 'failed') showToast(`Action training failed: ${data.error || data.message}`, 'error');
      }
      updateStartState();
    },
  }).start();
}

function renderJob() {
  const el = document.getElementById('act-train-job');
  const cancelBtn = document.getElementById('act-train-cancel');
  if (!el || !cancelBtn) return;
  const running = job && ['pending', 'running'].includes(job.status);
  cancelBtn.classList.toggle('hidden', !running);
  if (!job) {
    el.classList.add('hidden');
    el.innerHTML = '';
    return;
  }
  el.classList.remove('hidden');
  el.innerHTML = `
    ${renderJobProgress(job, { showLogs: true, truncateMsg: false })}
    ${logPanel(job)}
  `;
}

function logPanel(jobData) {
  const logs = Array.isArray(jobData.logs) ? jobData.logs.slice(-80) : [];
  if (!logs.length) return '';
  return `
    <details class="pt-1" open>
      <summary class="text-[10px] text-text-muted cursor-pointer hover:text-text-primary">Logs (${jobData.logs.length})</summary>
      <pre class="mt-2 max-h-72 overflow-y-auto rounded-lg bg-black/40 border border-white/5 p-2 font-mono text-[10px] text-text-secondary whitespace-pre-wrap break-words">${logs.map(escapeHtml).join('\n')}</pre>
    </details>`;
}

async function cancelTraining() {
  if (!job?.id) return;
  try {
    await api(API.jobs.cancel(job.id), { method: 'POST' });
    showToast('Action training cancelled', 'warning');
  } catch (e) {
    showToast(`Cancel failed: ${e.message}`, 'error');
  }
}
