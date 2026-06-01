/**
 * Action Predict page - SPOT pre-label jobs for action annotations.
 */
import {
  api, API, SSEClient, card, pageHeader, sectionTitle, btnSmall, showToast, showConfirm,
  emptyState, escapeHtml, inputCls, selectCls, renderJobProgress, renderJobItems,
} from '../shared.js';

let videos = [];
let spotInfo = { available: false, checkpoints: [], default_checkpoint: '' };
let selectedVideos = new Set();
let jobs = new Map();
let clients = new Map();
let kindFilter = 'all';
let statusFilter = 'unlabeled';

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Action Predict', 'Run SPOT action pre-label jobs', `
        ${btnSmall('Open Label', 'id="act-pred-label" title="Open Action Label"', 'primary')}
      `)}

      <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_24rem] gap-5">
        ${card(`
          <div class="space-y-4">
            ${sectionTitle('Videos', '', `
              <select id="act-pred-kind" class="${selectCls} text-xs" title="Filter by cut kind">
                <option value="all">All kinds</option>
                <option value="broadcast">Broadcast</option>
                <option value="sideline">Sideline</option>
              </select>
              <select id="act-pred-status" class="${selectCls} text-xs" title="Filter by action label status">
                <option value="unlabeled">Unlabeled</option>
                <option value="all">All</option>
                <option value="labeled">Labeled</option>
              </select>
            `)}
            <div class="flex items-center justify-between gap-3 text-xs text-text-muted">
              <label class="inline-flex items-center gap-2 cursor-pointer">
                <input id="act-pred-select-all" type="checkbox" class="accent-primary w-3.5 h-3.5">
                Select visible
              </label>
              <span id="act-pred-count" class="font-heading tabular-nums"></span>
            </div>
            <div id="act-pred-videos" class="space-y-1 max-h-[56vh] overflow-y-auto pr-1"></div>
          </div>
        `)}

        <div class="space-y-5">
          ${card(`
            <div class="space-y-4">
              ${sectionTitle('Prediction Settings', '', `${btnSmall('Refresh', 'id="act-pred-refresh"')}`)}
              <label class="block space-y-1.5">
                <span class="text-[11px] text-text-muted uppercase tracking-wider font-medium">Checkpoint</span>
                <select id="act-pred-checkpoint" class="w-full ${selectCls}"></select>
              </label>
              <div class="grid grid-cols-2 gap-3">
                <label class="block space-y-1.5">
                  <span class="text-[11px] text-text-muted uppercase tracking-wider font-medium">Min score</span>
                  <input id="act-pred-score" type="number" min="0" max="1" step="0.05" value="0.15" class="w-full ${inputCls}">
                </label>
                <label class="block space-y-1.5">
                  <span class="text-[11px] text-text-muted uppercase tracking-wider font-medium">Batch</span>
                  <input id="act-pred-batch" type="number" min="1" max="128" step="1" value="32" class="w-full ${inputCls}">
                </label>
              </div>
              <div class="grid grid-cols-2 gap-3">
                <label class="block space-y-1.5">
                  <span class="text-[11px] text-text-muted uppercase tracking-wider font-medium">Clip len</span>
                  <input id="act-pred-clip-len" type="number" min="8" max="256" step="8" value="64" class="w-full ${inputCls}">
                </label>
                <label class="block space-y-1.5">
                  <span class="text-[11px] text-text-muted uppercase tracking-wider font-medium">Workers</span>
                  <input id="act-pred-workers" type="number" min="0" max="16" step="1" value="8" class="w-full ${inputCls}">
                </label>
              </div>
              <div class="space-y-2">
                <label class="flex items-center gap-2 text-xs text-text-secondary cursor-pointer">
                  <input id="act-pred-overwrite" type="checkbox" class="accent-primary w-3.5 h-3.5">
                  Overwrite existing action labels
                </label>
                <label class="flex items-center gap-2 text-xs text-text-secondary cursor-pointer">
                  <input id="act-pred-stop-vllm" type="checkbox" class="accent-primary w-3.5 h-3.5">
                  Stop vLLM first
                </label>
              </div>
              <button id="act-pred-run" class="w-full bg-primary/10 text-primary-light hover:bg-primary/20 border border-primary/15 hover:border-primary/25 px-3 py-2 rounded-lg text-xs font-medium cursor-pointer transition-colors duration-150">Run SPOT</button>
            </div>
          `)}

          ${card(`
            <div class="space-y-3">
              ${sectionTitle('Jobs', '')}
              <div id="act-pred-jobs" class="space-y-3"></div>
            </div>
          `)}
        </div>
      </div>
    </div>`;

  bindEvents();
  loadInitial();
}

export function activate() {}

export function deactivate() {
  clients.forEach(client => client.stop());
  clients.clear();
}

function bindEvents() {
  document.getElementById('act-pred-label').addEventListener('click', () => { window.location.hash = '#/action-annotate'; });
  document.getElementById('act-pred-refresh').addEventListener('click', loadInitial);
  document.getElementById('act-pred-kind').addEventListener('change', (e) => {
    kindFilter = e.target.value;
    renderVideos();
  });
  document.getElementById('act-pred-status').addEventListener('change', (e) => {
    statusFilter = e.target.value;
    renderVideos();
  });
  document.getElementById('act-pred-select-all').addEventListener('change', (e) => {
    const visible = visibleVideos();
    if (e.target.checked) {
      visible.forEach(video => selectedVideos.add(video.name));
    } else {
      visible.forEach(video => selectedVideos.delete(video.name));
    }
    renderVideos();
  });
  document.getElementById('act-pred-run').addEventListener('click', runSelected);
}

async function loadInitial() {
  try {
    const [videoList, spotData] = await Promise.all([
      api(API.actionAnnotate.videos),
      api(API.actionAnnotate.spot).catch(e => ({ available: false, error: e.message, checkpoints: [] })),
    ]);
    videos = videoList;
    spotInfo = spotData;
    selectedVideos = new Set([...selectedVideos].filter(name => videos.some(v => v.name === name)));
    renderSpotControls();
    renderVideos();
    renderJobs();
  } catch (e) {
    showToast(`Failed to load Action Predict: ${e.message}`, 'error');
  }
}

function visibleVideos() {
  return videos.filter(video => {
    if (kindFilter !== 'all' && video.kind !== kindFilter) return false;
    if (statusFilter === 'unlabeled' && hasActiveActionAnnotation(video)) return false;
    if (statusFilter === 'labeled' && !hasActiveActionAnnotation(video)) return false;
    return true;
  });
}

function hasActiveActionAnnotation(video) {
  return Boolean(video?.has_action_annotation || video?.has_action_final_annotation || video?.has_action_pre_annotation);
}

function renderSpotControls() {
  const select = document.getElementById('act-pred-checkpoint');
  const run = document.getElementById('act-pred-run');
  if (!select || !run) return;

  select.innerHTML = '';
  if (!spotInfo.available) {
    select.innerHTML = `<option value="">SPOT unavailable: ${escapeHtml(spotInfo.error || '~/yp-spot not ready')}</option>`;
    run.disabled = true;
    run.classList.add('opacity-50', 'cursor-not-allowed');
    return;
  }

  const checkpoints = spotInfo.checkpoints || [];
  if (!checkpoints.length) {
    select.innerHTML = '<option value="">No checkpoint found</option>';
    run.disabled = true;
    run.classList.add('opacity-50', 'cursor-not-allowed');
    return;
  }

  select.innerHTML = checkpoints.map(ckpt => {
    const selected = ckpt.path === spotInfo.default_checkpoint ? 'selected' : '';
    const suffix = ckpt.is_best ? ' best' : ` epoch ${ckpt.epoch}`;
    return `<option value="${escapeHtml(ckpt.path)}" ${selected}>${escapeHtml(ckpt.name)} · ${suffix}</option>`;
  }).join('');
  run.disabled = false;
  run.classList.remove('opacity-50', 'cursor-not-allowed');
}

function renderVideos() {
  const visible = visibleVideos();
  const list = document.getElementById('act-pred-videos');
  const count = document.getElementById('act-pred-count');
  const all = document.getElementById('act-pred-select-all');
  if (count) count.textContent = `${selectedVideos.size} selected / ${visible.length} shown`;
  if (all) all.checked = visible.length > 0 && visible.every(video => selectedVideos.has(video.name));

  if (!visible.length) {
    list.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3"/></svg>',
      'No videos',
      '',
    );
    return;
  }

  list.innerHTML = visible.map(video => `
    <label class="flex items-center gap-3 px-3 py-2 rounded-lg border border-border bg-white/[0.035] hover:bg-white/[0.06] cursor-pointer transition-colors duration-150">
      <input type="checkbox" class="act-pred-video accent-primary w-3.5 h-3.5" value="${escapeHtml(video.name)}" ${selectedVideos.has(video.name) ? 'checked' : ''}>
      <span class="w-2 h-2 rounded-full ${video.kind === 'broadcast' ? 'bg-primary-light' : 'bg-accent-light'} flex-shrink-0"></span>
      <span class="min-w-0 flex-1 truncate text-sm text-text-primary">${escapeHtml(video.name)}</span>
      <span class="text-[11px] ${hasActiveActionAnnotation(video) ? 'text-emerald-300' : 'text-text-muted'} font-heading tabular-nums">${video.event_count || 0}</span>
    </label>
  `).join('');

  list.querySelectorAll('.act-pred-video').forEach(input => {
    input.addEventListener('change', () => {
      if (input.checked) selectedVideos.add(input.value);
      else selectedVideos.delete(input.value);
      renderVideos();
    });
  });
}

async function runSelected() {
  const names = [...selectedVideos];
  if (!names.length) return showToast('Select at least one video', 'warning');
  const selectedRecords = names.map(name => videos.find(v => v.name === name)).filter(Boolean);
  const existing = selectedRecords.filter(hasActiveActionAnnotation);
  const overwrite = document.getElementById('act-pred-overwrite').checked;
  if (existing.length && !overwrite) {
    showToast(`${existing.length} selected video(s) already have action labels`, 'warning');
    return;
  }
  if (existing.length && overwrite) {
    const ok = await showConfirm({
      title: 'Overwrite action labels?',
      body: `This will replace the active action labels for ${existing.length} video(s). Saved labels will be replaced by new pre-labels.`,
      confirmText: 'Overwrite',
      variant: 'danger',
    });
    if (!ok) return;
  }

  const btn = document.getElementById('act-pred-run');
  btn.disabled = true;
  try {
    const job = await api(API.actionAnnotate.prelabelBatch, {
      method: 'POST',
      body: {
        videos: names,
        checkpoint: document.getElementById('act-pred-checkpoint').value,
        min_score: Number(document.getElementById('act-pred-score').value) || 0.15,
        batch_size: Number(document.getElementById('act-pred-batch').value) || 32,
        num_workers: Number(document.getElementById('act-pred-workers').value),
        clip_len: Number(document.getElementById('act-pred-clip-len').value) || 64,
        use_amp: true,
        overwrite,
        stop_vllm: document.getElementById('act-pred-stop-vllm').checked,
      },
    });
    jobs.set(job.id, job);
    subscribeJob(job.id);
    renderJobs();
    showToast(`Started SPOT batch for ${names.length} video(s)`, 'success');
  } catch (e) {
    showToast(`SPOT start failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

function subscribeJob(jobId) {
  clients.get(jobId)?.stop();
  const client = new SSEClient(API.jobs.eventsSSE(jobId), {
    onMessage: (job) => {
      jobs.set(job.id, job);
      renderJobs();
      if (['completed', 'failed', 'cancelled'].includes(job.status)) {
        clients.get(job.id)?.stop();
        clients.delete(job.id);
        if (job.status === 'completed') loadInitial();
      }
    },
  }).start();
  clients.set(jobId, client);
}

function renderJobs() {
  const el = document.getElementById('act-pred-jobs');
  if (!el) return;
  const items = [...jobs.values()].sort((a, b) => jobCreatedAt(b) - jobCreatedAt(a));
  if (!items.length) {
    el.innerHTML = '<div class="text-xs text-text-muted">No action prediction jobs</div>';
    return;
  }
  el.innerHTML = items.map(job => renderJobProgress(job, {
    detail: renderJobItems(job),
    showLogs: true,
    truncateMsg: false,
  })).join('');
}

function jobCreatedAt(job) {
  const value = job?.created_at;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed / 1000;
  }
  return 0;
}
