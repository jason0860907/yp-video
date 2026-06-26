/**
 * Action Train page - SPOT training for action labels.
 */
import {
  api, API, SSEClient, card, pageHeader, sectionTitle, btnSmall,
  escapeHtml, showToast, inputCls, selectCls, renderJobProgress,
} from '../shared.js';

let status = null;
let source = 'action_annotations';
let job = null;
let client = null;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Action Train', 'Train SPOT action labels from JSONL datasets', `
        ${btnSmall('Export JSONL', 'id="act-train-export" title="Download action annotation dataset"')}
      `)}

      ${card(`
        <div class="space-y-5">
          <div class="space-y-4">
            ${sectionTitle('Dataset', '', `${btnSmall('Refresh', 'id="act-train-refresh"')}`)}
            <div id="act-train-stats" class="grid grid-cols-2 md:grid-cols-5 gap-3"></div>
            <div id="act-train-source-summary" class="rounded-xl border border-border bg-surface-100/35 p-3"></div>
          </div>

          <div class="border-t border-border pt-5 space-y-4">
            ${sectionTitle('Training', '', `<span id="act-train-ready" class="text-[11px] text-text-muted"></span>`)}

            <div class="space-y-3">
              <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-2">
                ${field('Source', 'act-train-source', select(`
                  <option value="vnl_1_5">VNL 1.5 JSONL</option>
                  <option value="action_annotations">YP Action Labels</option>
                `), 'md:col-span-2 xl:col-span-1')}
                ${field('Dataset', 'act-train-dataset', input('text', 'vnl_1.5'))}
                ${field('Frames', 'act-train-frame-dir', input('text', 'data/vnl_1.5/frames_224p', '', '', '1', '', 'font-mono text-[11px]'), 'md:col-span-2 xl:col-span-2')}
                ${field('Checkpoint Dir', 'act-train-checkpoint-dir', input('text', '', '', '', '1', 'auto: ~/videos/action-checkpoints/...', 'font-mono text-[11px]'), 'md:col-span-2')}
                ${field('Init Checkpoint', 'act-train-init-checkpoint', select('<option value="">Loading…</option>'), 'md:col-span-2')}
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
                ${field('Audio', 'act-train-audio', select(`
                  <option value="logmel">logmel (late fusion)</option>
                  <option value="none">none (visual only)</option>
                `))}
                ${field('Epochs', 'act-train-epochs', input('number', '50', '1', '1000'))}
                ${field('Batch', 'act-train-batch', input('number', '8', '1', '64'))}
                ${field('Clip Len', 'act-train-clip', input('number', '64', '8', '256'))}
                ${field('Workers', 'act-train-workers', input('number', '4', '0', '32'))}
                ${field('GPU', 'act-train-gpu', input('number', '0', '0', '7'))}
                ${field('LR', 'act-train-lr', input('number', '0.0003', '0', '', 'any'))}
                ${field('Warmup', 'act-train-warmup', input('number', '3', '0', '100'))}
                ${field('Criterion', 'act-train-criterion', select(`
                  <option value="map">map</option>
                  <option value="loss">loss</option>
                `))}
                ${field('Start Val', 'act-train-start-val', input('number', '0', '0', '1000'))}
                ${field('Epoch Frames', 'act-train-epoch-frames', input('number', '', '1', '', '1', 'optional'))}

                <div id="act-train-mode-wrap" class="contents">
                  ${field('Data Mode', 'act-train-training-mode', select(`
                    <option value="all" selected>All Data</option>
                    <option value="split">Train/Test Split</option>
                  `))}
                  ${field('Camera View', 'act-train-camera-view', select(`
                    <option value="all" selected>All Views</option>
                    <option value="broadcast">Broadcast</option>
                    <option value="sideline">Sideline</option>
                  `))}
                </div>

                <div id="act-train-split-fields" class="contents">
                  ${field('Val Ratio', 'act-train-val-ratio', input('number', '0.2', '0.01', '0.9', '0.01'))}
                  ${field('Split Seed', 'act-train-split-seed', input('number', '42'))}
                </div>
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
        </div>
      `)}
    </div>`;

  document.getElementById('act-train-refresh').addEventListener('click', loadData);
  document.getElementById('act-train-export').addEventListener('click', exportDataset);
  document.getElementById('act-train-source').addEventListener('change', (e) => {
    source = e.target.value;
    applySourceDefaults();
    renderDataset();
  });
  document.getElementById('act-train-training-mode').addEventListener('change', () => {
    updateTrainingModeControls();
    renderDataset();
  });
  document.getElementById('act-train-camera-view').addEventListener('change', renderDataset);
  document.getElementById('act-train-start').addEventListener('click', startTraining);
  document.getElementById('act-train-cancel').addEventListener('click', cancelTraining);
  loadData();
}

export function activate() {}

export function deactivate() {
  client?.stop();
  client = null;
}

function field(label, id, control, extraClass = '') {
  return `
    <label class="block space-y-1.5 min-w-0 ${extraClass}">
      <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">${label}</span>
      ${control.replace('__ID__', id)}
    </label>`;
}

function input(type, value, min = '', max = '', step = '1', placeholder = '', extraClass = '') {
  const numberAttrs = type === 'number'
    ? `${min !== '' ? ` min="${min}"` : ''}${max !== '' ? ` max="${max}"` : ''}${step !== '' ? ` step="${step}"` : ''}`
    : '';
  return `<input id="__ID__" type="${type}" value="${escapeHtml(value)}"${numberAttrs} placeholder="${escapeHtml(placeholder)}" class="${inputCls} w-full ${extraClass}">`;
}

function select(options) {
  return `<select id="__ID__" class="${selectCls} w-full">${options}</select>`;
}

async function loadData() {
  try {
    status = await api(API.actionTrain.status);
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
  const modeEl = document.getElementById('act-train-training-mode');
  const modeWrap = document.getElementById('act-train-mode-wrap');
  const splitEl = document.getElementById('act-train-split-fields');
  populateInitCheckpoints();
  if (source === 'vnl_1_5') {
    datasetEl.value = 'vnl_1.5';
    framesEl.value = 'data/vnl_1.5/frames_224p';
    modeWrap.classList.add('hidden');
    splitEl.classList.add('hidden');
  } else {
    datasetEl.value = 'yp_actions';
    framesEl.value = '~/videos/action-frames';
    if (job?.params?.training_mode && modeEl) modeEl.value = job.params.training_mode;
    const viewEl = document.getElementById('act-train-camera-view');
    if (job?.params?.camera_view && viewEl) viewEl.value = job.params.camera_view;
    modeWrap.classList.remove('hidden');
    updateTrainingModeControls();
  }
  updateStartState();
}

function populateInitCheckpoints() {
  const initEl = document.getElementById('act-train-init-checkpoint');
  if (!initEl) return;
  const opts = Array.isArray(status?.init_checkpoints) ? status.init_checkpoints : [];
  const prev = initEl.value;
  if (!opts.length) {
    initEl.innerHTML = '<option value="">No checkpoints found</option>';
    return;
  }
  initEl.innerHTML = opts
    .map((o) => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`)
    .join('');
  const values = opts.map((o) => o.value);
  if (prev && values.includes(prev)) initEl.value = prev;
  else initEl.value = values[0];
}

function trainingMode() {
  return document.getElementById('act-train-training-mode')?.value || 'split';
}

function cameraView() {
  return document.getElementById('act-train-camera-view')?.value || 'all';
}

function updateTrainingModeControls() {
  const splitEl = document.getElementById('act-train-split-fields');
  if (!splitEl) return;
  const showSplit = source === 'action_annotations' && trainingMode() === 'split';
  splitEl.classList.toggle('hidden', !showSplit);
}

function datasetStats() {
  const action = status?.action_annotations || {};
  return {
    videos: Math.max(0, Number(action.videos) || 0),
    actions: Math.max(0, Number(action.events) || 0),
    frames: Math.max(0, Number(action.frames) || 0),
    broadcast: 0,
    sideline: 0,
  };
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
    statCell('Action Frames', stats.frames.toLocaleString()),
    statCell('VNL Train', vnl.train_events || 0),
    statCell('VNL Val', vnl.val_events || 0),
  ].join('');

  const summary = document.getElementById('act-train-source-summary');
  summary.innerHTML = source === 'vnl_1_5' ? vnlSummary(vnl) : actionSummary(stats);
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
  ];
  return summaryRows(rows, vnl.ready);
}

function actionSummary(stats) {
  const rows = [
    ['Labels', `${stats.videos} videos / ${stats.actions} events / ${stats.frames.toLocaleString()} frames`],
    ['Mode', trainingMode() === 'all' ? 'all data' : 'train/test split'],
    ['View', cameraView() === 'all' ? 'all views' : cameraView()],
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
      training_mode: source === 'action_annotations' ? trainingMode() : 'split',
      camera_view: source === 'action_annotations' ? cameraView() : 'all',
      dataset: textValue('act-train-dataset'),
      frame_dir: textValue('act-train-frame-dir'),
      checkpoint_dir: textValue('act-train-checkpoint-dir', null),
      init_checkpoint: textValue('act-train-init-checkpoint', null),
      gpu: numberValue('act-train-gpu', 0),
      feature_arch: textValue('act-train-feature', 'rny008_gsm'),
      temporal_arch: textValue('act-train-temporal', 'gru'),
      audio_backend: textValue('act-train-audio', 'logmel'),
      clip_len: numberValue('act-train-clip', 64),
      batch_size: numberValue('act-train-batch', 8),
      num_epochs: numberValue('act-train-epochs', 50),
      warm_up_epochs: numberValue('act-train-warmup', 3),
      learning_rate: numberValue('act-train-lr', 0.0003),
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
  const detail = actionTrainDetail(job);
  el.innerHTML = `
    ${renderJobProgress(job, { detail, showLogs: true, truncateMsg: false })}
    ${logPanel(job)}
  `;
}

function actionTrainDetail(jobData) {
  const p = jobData?.params?.action_train_progress;
  if (!p) return '';
  const phaseProgress = Number.isFinite(Number(p.phase_progress))
    ? `${Math.round(Number(p.phase_progress) * 100)}%`
    : '';
  const step = Number.isFinite(Number(p.step)) && Number.isFinite(Number(p.total))
    ? `${p.step}/${p.total}${phaseProgress ? ` (${phaseProgress})` : ''}`
    : '';
  const latestMap = Number.isFinite(Number(p.latest_val_map))
    ? `${(Number(p.latest_val_map) * 100).toFixed(2)}%`
    : '';
  const best = Number.isFinite(Number(p.best_value))
    ? `${p.best_epoch != null ? `E${Number(p.best_epoch) + 1} ` : ''}${p.best_value <= 1 ? (Number(p.best_value) * 100).toFixed(2) + '%' : Number(p.best_value).toFixed(4)}`
    : '';
  const rows = [
    ['Epoch', `${p.epoch_display || 1}/${p.epochs || jobData?.params?.epochs || '?'}`],
    ['Phase', p.phase_label || p.phase || ''],
    ['Step', step],
    ['Current Loss', formatMetric(p.current_loss)],
    ['Last Train', formatMetric(p.latest_train_loss)],
    ['Last Val', formatMetric(p.latest_val_loss)],
    ['Last mAP', latestMap],
    ['Best', best],
  ].filter(([, value]) => value !== '' && value != null);
  return `
    <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
      ${rows.map(([label, value]) => `
        <div class="rounded-lg border border-border bg-surface-200/35 px-2.5 py-2 min-w-0">
          <div class="text-[9px] uppercase tracking-wider text-text-muted">${escapeHtml(label)}</div>
          <div class="mt-0.5 text-[11px] text-text-secondary font-heading tabular-nums truncate" title="${escapeHtml(value)}">${escapeHtml(value)}</div>
        </div>
      `).join('')}
    </div>`;
}

function formatMetric(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(4) : '';
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
