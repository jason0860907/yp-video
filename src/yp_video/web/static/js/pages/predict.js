/**
 * Predict page — TAD inference with multi-video selection and result visualization.
 */
import { api, API, SSEClient, formatTime, card, pageHeader, sectionTitle, stepBadge, btnPrimary, btnSmall, showToast, showConfirm, emptyState, inputCls, selectCls, kindTabs, updateKindTabs, badges, renderJobProgress, filterChips, bindFilterChips } from '../shared.js';

let sseClients = [];
let state = { videos: [], checkpoints: [], results: [], jobs: [] };
let videoEl = null;
let timelineCanvas = null;
let animFrame = null;
let _selectedIdx = -1;
let _detections = [];
let _duration = 0;
let predKindFilter = 'all';

// Per-property tri-state filter, same shape as Train.
// null = any, true = must have, false = must NOT have. AND across props.
const filterState = {
  annotated: null,
  pre_annotated: null,
  features: null,
  prediction: null,
};

function _currentModel() {
  return document.getElementById('pred-model')?.value || 'large';
}

function matchesFilter(v) {
  if (predKindFilter !== 'all' && v.kind !== predKindFilter) return false;
  for (const [prop, want] of Object.entries(filterState)) {
    if (want === null) continue;
    let has;
    if (prop === 'annotated')         has = !!v.has_annotation;
    else if (prop === 'pre_annotated') has = !!v.has_pre_annotation;
    else if (prop === 'prediction')    has = !!v.has_prediction;
    else if (prop === 'features')      has = !!(v.features && v.features[_currentModel()]);
    else continue;
    if (has !== want) return false;
  }
  return true;
}

function visibleVideos() {
  return state.videos.filter(matchesFilter);
}

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('TAD Predict', 'Run TAD inference on videos')}

      ${card(`
        <div class="space-y-4">
          ${sectionTitle(
            'Videos',
            '',
            `${kindTabs('pred')}
             <select id="pred-model" class="${selectCls}" title="Feature model — affects which videos have features available">
               <option value="base">ViT-B (768d)</option>
               <option value="large" selected>ViT-L (1024d)</option>
               <option value="giant">ViT-g (1408d)</option>
               <option value="gigantic">ViT-G (1664d)</option>
             </select>
             ${btnSmall('Select All', 'id="pred-select-all"')}
             ${btnSmall('Deselect All', 'id="pred-deselect-all"')}
             ${filterChips('pred', ['annotated', 'pre_annotated', 'features', 'prediction'])}`
          )}
          <div id="pred-videos" class="space-y-0.5 max-h-72 overflow-y-auto pr-1"></div>
        </div>
      `)}

      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(1, 'accent')}
            <div>
              ${sectionTitle('Prediction Settings', 'Select checkpoint and configure inference parameters')}
            </div>
          </div>
          <div class="ml-10 grid grid-cols-2 gap-4">
            <div>
              <div class="flex items-center justify-between mb-1.5">
                <label class="block text-[11px] text-text-muted uppercase tracking-wider font-medium">Checkpoint</label>
                <label class="flex items-center gap-1.5 text-[11px] text-text-muted cursor-pointer">
                  <input id="pred-show-all-ckpts" type="checkbox" class="accent-primary w-3 h-3">
                  show all
                </label>
              </div>
              <select id="pred-checkpoint" class="w-full ${selectCls}">
                <option value="">Select checkpoint...</option>
              </select>
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Threshold</label>
              <input id="pred-threshold" type="number" value="0.3" min="0" max="1" step="0.05" class="w-full ${inputCls}">
            </div>
          </div>
          <div class="ml-10 grid grid-cols-3 gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Device</label>
              <select id="pred-device" class="w-full ${selectCls}">
                <option value="cuda">CUDA</option>
                <option value="cpu">CPU</option>
              </select>
            </div>
            <div class="flex items-end pb-1">
              <label class="flex items-center gap-2.5 cursor-pointer text-sm text-text-secondary hover:text-text-primary transition-colors duration-200">
                <input id="pred-cut" type="checkbox" class="accent-primary cursor-pointer w-4 h-4 rounded">
                Cut rallies
              </label>
            </div>
          </div>
          <div class="ml-10 flex items-center gap-3 pt-1">
            ${btnPrimary('Start Prediction', 'id="pred-start"')}
            <span id="pred-count" class="text-xs text-text-muted font-heading tabular-nums ml-auto"></span>
          </div>
        </div>
      `)}

      <div id="pred-progress" class="hidden">
        ${card(`
          <div class="space-y-3">
            <div class="flex items-center gap-2.5 mb-3">
              <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse-dot"></span>
              <h3 class="text-sm font-heading font-semibold text-text-primary">Progress</h3>
            </div>
            <div id="pred-jobs-progress" class="space-y-3"></div>
            <div id="pred-retry-wrap" class="hidden pt-1">
              ${btnSmall('Retry Failed', 'id="pred-retry-failed"', 'primary')}
            </div>
          </div>
        `)}
      </div>

      ${card(`
        <div class="space-y-4">
          ${sectionTitle(
            'Results',
            `${state.results.length || ''} predictions`.trim(),
            btnSmall('Refresh', 'id="pred-refresh"')
          )}
          <div id="pred-results"></div>
        </div>
      `)}

      <div id="pred-detail" class="hidden"></div>
    </div>`;

  loadData();
  bindEvents();
}

export function activate() {
  document.addEventListener('keydown', handleKeydown);
  window.addEventListener('resize', resizeTimeline);
  startTimelineLoop();
}

export function deactivate() {
  document.removeEventListener('keydown', handleKeydown);
  window.removeEventListener('resize', resizeTimeline);
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
  if (videoEl && !videoEl.paused) videoEl.pause();
  sseClients.forEach(c => c.stop());
  sseClients = [];
}

function handleKeydown(e) {
  if (!videoEl || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  switch (e.key) {
    case ' ': e.preventDefault(); videoEl.paused ? videoEl.play() : videoEl.pause(); break;
    case 'ArrowLeft': e.preventDefault(); videoEl.currentTime = Math.max(0, videoEl.currentTime - 5); break;
    case 'ArrowRight': e.preventDefault(); videoEl.currentTime += 5; break;
    case 'ArrowUp': e.preventDefault(); jumpDetection(-1); break;
    case 'ArrowDown': e.preventDefault(); jumpDetection(1); break;
  }
}

function jumpDetection(dir) {
  if (_detections.length === 0) return;
  _selectedIdx = Math.max(0, Math.min(_detections.length - 1, _selectedIdx + dir));
  const d = _detections[_selectedIdx];
  videoEl.currentTime = d.start;
  videoEl.play();
  renderDetections();
  const el = document.querySelector(`.det-item[data-idx="${_selectedIdx}"]`);
  if (el) el.scrollIntoView({ block: 'nearest' });
}

function bindEvents() {
  document.getElementById('pred-start').addEventListener('click', startPrediction);
  document.getElementById('pred-refresh').addEventListener('click', loadResults);
  document.getElementById('pred-retry-failed').addEventListener('click', retryFailed);
  // Bulk-select operates on the kind-filtered subset so tabs narrow the
  // selection scope (e.g. "select all broadcast" without touching sideline).
  const setSelectionForVisible = (pred) => {
    const visible = new Set(visibleVideos());
    state.videos.forEach(v => { if (visible.has(v)) v.selected = pred(v); });
    renderVideos();
  };
  document.getElementById('pred-select-all').addEventListener('click',
    () => setSelectionForVisible(() => true));
  document.getElementById('pred-deselect-all').addEventListener('click',
    () => setSelectionForVisible(() => false));
  document.getElementById('pred-model').addEventListener('change', renderVideos);
  document.querySelectorAll('.kind-tab[data-prefix="pred"]').forEach(btn => {
    btn.addEventListener('click', () => {
      predKindFilter = btn.dataset.kind;
      renderVideos();
    });
  });
  // Tri-state filter chips: narrow the visible list by per-property AND
  // filters. Combined with Select All this replaces the old "Unpredicted"
  // / "✅ Annotated" shortcut buttons with something composable.
  bindFilterChips('pred', filterState, renderVideos);

  // Delegated handlers for dynamic lists — bound once at page render so
  // re-rendering the list innerHTML doesn't have to re-bind N rows of
  // listeners (and doesn't leak the old ones).
  document.getElementById('pred-videos').addEventListener('change', (e) => {
    if (!e.target.matches('.pred-check')) return;
    state.videos[parseInt(e.target.dataset.idx)].selected = e.target.checked;
    updatePredCount();
  });
  document.getElementById('pred-results').addEventListener('click', (e) => {
    const item = e.target.closest('[data-name]');
    if (item) viewResult(item.dataset.name);
  });
}

async function loadCheckpoints() {
  const showAll = document.getElementById('pred-show-all-ckpts')?.checked ?? false;
  const cpSel = document.getElementById('pred-checkpoint');
  const prev = cpSel.value;
  try {
    const checkpoints = await api(API.train.checkpoints({ show_all: showAll }));
    state.checkpoints = checkpoints;
    cpSel.innerHTML = '<option value="">Select checkpoint...</option>';
    const KIND_TAG = { best: '⭐', last: '🆕', epoch: '' };
    checkpoints.forEach(cp => {
      const opt = document.createElement('option');
      opt.value = cp.path;
      const tag = KIND_TAG[cp.kind] ?? '';
      opt.textContent = `${tag} ${cp.name} (${cp.size_mb.toFixed(1)} MB)`.trim();
      cpSel.appendChild(opt);
    });
    // Restore previous selection if still present
    if (prev && checkpoints.some(c => c.path === prev)) cpSel.value = prev;
  } catch (e) {
    showToast(`Failed to load checkpoints: ${e.message}`, 'error');
  }
}

async function loadData() {
  try {
    const videos = await api(API.predict.videos);
    state.videos = videos.map(v => ({ ...v, selected: !v.has_prediction }));
    renderVideos();

    await loadCheckpoints();
    document.getElementById('pred-show-all-ckpts').addEventListener('change', loadCheckpoints);

    await loadResults();
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}

function renderVideos() {
  const el = document.getElementById('pred-videos');
  updateKindTabs('pred', predKindFilter, state.videos);
  if (state.videos.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>',
      'No cut videos found',
      'Cut some videos first'
    );
    return;
  }

  const vis = visibleVideos();
  if (vis.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 4h18M3 12h18M3 20h18"/></svg>',
      'No videos match the filters',
      'Adjust the kind tab or filter chips above',
    );
    updatePredCount();
    return;
  }

  const model = document.getElementById('pred-model')?.value || 'large';
  // Index back into the master `state.videos` so checkbox toggles still mutate
  // the right entry after filtering.
  el.innerHTML = vis.map(v => {
    const i = state.videos.indexOf(v);
    const annBadge = v.has_annotation ? badges.annotated()
      : (v.has_pre_annotation ? badges.preAnnotated() : '');
    const hasFeat = v.features?.[model];
    const featBadge = hasFeat ? badges.hasFeatures() : badges.noFeatures();
    const predBadge = v.has_prediction ? badges.predictedPill() : badges.pendingPill();
    const rowDim = hasFeat ? '' : ' opacity-60';
    return `
    <div class="group flex items-center gap-3 p-2.5 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200${rowDim}">
      <input type="checkbox" data-idx="${i}" class="pred-check cursor-pointer accent-primary w-3.5 h-3.5" ${v.selected ? 'checked' : ''}>
      <span class="text-sm text-text-primary flex-1 truncate group-hover:text-white transition-colors duration-200">${v.name}</span>
      ${annBadge}
      ${featBadge}
      ${predBadge}
    </div>`;
  }).join('');

  updatePredCount();
}

function updatePredCount() {
  const el = document.getElementById('pred-count');
  if (!el) return;
  const sel = state.videos.filter(v => v.selected).length;
  el.textContent = state.videos.length ? `${sel} / ${state.videos.length} selected` : '';
}

async function loadResults() {
  try {
    state.results = await api(API.predict.results);
    renderResults();
  } catch { /* silently fail */ }
}

function renderResults() {
  const el = document.getElementById('pred-results');
  if (state.results.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>',
      'No results yet',
      'Run a prediction to see results'
    );
    return;
  }

  el.innerHTML = `<div class="space-y-1">${state.results.map(name => `
    <div class="group flex items-center gap-3 p-2.5 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200 cursor-pointer" data-name="${name}">
      <div class="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/15 flex items-center justify-center flex-shrink-0">
        <svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
      </div>
      <span class="text-sm text-text-primary group-hover:text-white transition-colors duration-200 truncate flex-1">${name}</span>
      <svg class="w-4 h-4 text-text-muted opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>
    </div>
  `).join('')}</div>`;
  // Click handler is delegated from #pred-results in bindEvents().
}

async function startPrediction() {
  const selected = state.videos.filter(v => v.selected).map(v => v.name);
  if (selected.length === 0) return showToast('No videos selected', 'warning');

  const checkpoint = document.getElementById('pred-checkpoint').value;
  if (!checkpoint) return showToast('Select a checkpoint', 'warning');

  let stopVllm = false;
  const vllmStatus = await api(API.system.vllmStatus).catch(() => null);
  if (vllmStatus?.status === 'running') {
    const ok = await showConfirm({
      title: 'Stop vLLM for prediction?',
      body:
        'vLLM is running and holds most of the GPU VRAM.\n' +
        'Prediction needs the GPU for V-JEPA and would OOM otherwise.\n\n' +
        'vLLM will be automatically restarted once prediction finishes.',
      confirmText: 'Stop & Predict',
      cancelText: 'Cancel',
      variant: 'warning',
    });
    if (!ok) return;
    stopVllm = true;
  }

  const btn = document.getElementById('pred-start');
  btn.disabled = true;
  document.getElementById('pred-retry-wrap').classList.add('hidden');

  // Stop any existing SSE clients
  sseClients.forEach(c => c.stop());
  sseClients = [];

  try {
    const job = await api(API.predict.start, {
      method: 'POST',
      body: {
        videos: selected,
        checkpoint,
        threshold: parseFloat(document.getElementById('pred-threshold').value),
        device: document.getElementById('pred-device').value,
        cut_rallies: document.getElementById('pred-cut').checked,
        model: document.getElementById('pred-model').value,
        stop_vllm: stopVllm,
      },
    });

    state.jobs = [job];
    document.getElementById('pred-progress').classList.remove('hidden');
    renderJobsProgress();

    const client = new SSEClient(API.jobs.eventsSSE(job.id), {
      onMessage: (data) => {
        state.jobs = [data];
        renderJobsProgress();

        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
          client.stop();
          if (data.status === 'failed') {
            showToast(`Prediction failed: ${data.error || 'Unknown error'}`, 'error');
          } else {
            showToast(data.message || 'Prediction complete!', 'success');
          }
          btn.disabled = false;
          loadVideos();
        }
      },
      onError: () => {
        btn.disabled = false;
      },
    }).start();
    sseClients.push(client);
  } catch (e) {
    showToast(`Failed to start prediction: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

function retryFailed() {
  startPrediction();
}

async function loadVideos() {
  try {
    const videos = await api(API.predict.videos);
    state.videos = videos.map(v => ({ ...v, selected: !v.has_prediction }));
    renderVideos();
  } catch { /* silently fail */ }
}

function renderJobsProgress() {
  const el = document.getElementById('pred-jobs-progress');
  if (!el) return;
  el.innerHTML = state.jobs.map(job =>
    renderJobProgress(job, { showLogs: true, truncateMsg: false })
  ).join('');
}

// \u2500\u2500 Detail view (clicked-result inline player) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
//
// viewResult() is split into three pieces so each step is < 50 lines:
//   1. fetch + normalize the data
//   2. inject the DOM + bind events (one-shot per result)
//   3. cleanup on close
// renderDetections() owns the row template; clicks on rows are handled
// by a single delegated listener bound in _setupDetailEvents.

async function viewResult(name) {
  try {
    const data = await api(API.predict.result(name));
    _detections = (data.results || []).map(r => ({
      start: r.start ?? r.segment?.[0] ?? 0,
      end: r.end ?? r.segment?.[1] ?? 0,
      label: r.label || 'rally',
      score: r.confidence ?? r.score ?? null,
    })).sort((a, b) => a.start - b.start);
    _selectedIdx = -1;
    _duration = data.duration || 0;
    const videoName = (data.video || '').split('/').pop() || '';

    const detailEl = document.getElementById('pred-detail');
    detailEl.innerHTML = _buildDetailHTML(videoName || name);
    detailEl.classList.remove('hidden');
    _setupDetailEvents(videoName);
    renderDetections();
    resizeTimeline();
    startTimelineLoop();
    detailEl.scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    showToast(`Failed to load result: ${e.message}`, 'error');
  }
}

function _buildDetailHTML(title) {
  return `
    <div class="flex flex-col lg:flex-row gap-5">
      <div class="flex-1 min-w-0 space-y-4">
        ${card(`
          <div class="space-y-4">
            <div class="flex items-center justify-between">
              ${sectionTitle(title, `${_detections.length} detections`)}
              ${btnSmall('Close', 'id="pred-detail-close"')}
            </div>
            <div class="relative bg-black rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-lg shadow-black/30">
              <video id="pred-player" class="w-full max-h-[50vh]" controls></video>
            </div>
            <div class="space-y-2">
              <div class="relative rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-inner" style="background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)">
                <canvas id="pred-timeline" class="w-full h-12 cursor-pointer" title="Click to seek"></canvas>
              </div>
              <div class="flex items-center justify-between px-0.5">
                <span id="pred-time" class="text-sm font-heading text-text-primary tabular-nums bg-surface-200/50 px-2.5 py-1 rounded-lg border border-border">00:00</span>
                <div class="flex items-center gap-3 text-[11px] text-text-muted">
                  <span class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-emerald-500/60 inline-block"></span> rally</span>
                  <span class="flex items-center gap-1.5"><span class="w-1.5 h-3 rounded bg-orange-500 inline-block"></span> playhead</span>
                  <kbd class="px-1.5 py-0.5 rounded bg-surface-200 border border-border text-[10px] font-heading text-text-secondary">\u2191\u2193</kbd>
                  <span>jump rally</span>
                </div>
              </div>
            </div>
          </div>
        `)}
      </div>

      <div class="lg:w-[400px] lg:flex-shrink-0">
        ${card(`
          <div class="space-y-3">
            ${sectionTitle(`Detections <span id="pred-det-count" class="text-text-muted font-normal">(${_detections.length})</span>`)}
            <div class="h-px bg-border"></div>
            <div id="pred-det-list" class="space-y-1 max-h-[60vh] overflow-y-auto pr-1 scrollbar-thin"></div>
          </div>
        `)}
      </div>
    </div>`;
}

function _setupDetailEvents(videoName) {
  videoEl = document.getElementById('pred-player');
  timelineCanvas = document.getElementById('pred-timeline');

  if (videoName) {
    videoEl.src = `/api/predict/video/${encodeURIComponent(videoName)}`;
    videoEl.load();
  }

  videoEl.addEventListener('loadedmetadata', () => {
    _duration = videoEl.duration;
    resizeTimeline();
  });
  videoEl.addEventListener('timeupdate', () => {
    const timeEl = document.getElementById('pred-time');
    if (timeEl) timeEl.textContent = formatTime(videoEl.currentTime);
    // Auto-pause at end of selected segment so the user can listen to the
    // tail of a single rally without scrubbing back.
    if (_selectedIdx >= 0 && _selectedIdx < _detections.length) {
      const d = _detections[_selectedIdx];
      if (!videoEl.paused && videoEl.currentTime >= d.end) {
        videoEl.pause();
        videoEl.currentTime = d.end;
      }
    }
  });

  timelineCanvas.addEventListener('click', (e) => {
    if (!_duration) return;
    const rect = timelineCanvas.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    videoEl.currentTime = ratio * _duration;
  });

  document.getElementById('pred-detail-close').addEventListener('click', _closeDetailView);

  // Single delegated listener for all rows + their preview button instead
  // of re-binding each row on every render.
  document.getElementById('pred-det-list').addEventListener('click', (e) => {
    const row = e.target.closest('.det-item');
    if (!row) return;
    const idx = parseInt(row.dataset.idx);
    _selectedIdx = idx;
    const d = _detections[idx];
    videoEl.currentTime = e.target.closest('[data-action="preview"]')
      ? Math.max(d.start, d.end - 5)
      : d.start;
    videoEl.play();
    renderDetections();
  });
}

function _closeDetailView() {
  if (videoEl && !videoEl.paused) videoEl.pause();
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
  videoEl = null;
  timelineCanvas = null;
  _detections = [];
  _selectedIdx = -1;
  document.getElementById('pred-detail').classList.add('hidden');
}

function renderDetections() {
  const el = document.getElementById('pred-det-list');
  if (!el) return;
  el.innerHTML = _detections.map(_renderDetectionRow).join('');
  // Click handlers are delegated from #pred-det-list in _setupDetailEvents.
}

function _renderDetectionRow(d, i) {
  const selected = _selectedIdx === i;
  const durationSec = (d.end - d.start).toFixed(1);
  const scoreColor = d.score > 0.7 ? 'text-emerald-400' : d.score > 0.4 ? 'text-amber-400' : 'text-text-muted';
  const scoreBg = d.score > 0.7 ? 'bg-emerald-500/10 ring-emerald-500/20'
                : d.score > 0.4 ? 'bg-amber-500/10 ring-amber-500/20'
                : 'bg-white/5 ring-white/10';
  const rowCls = selected
    ? 'border-emerald-500/40 bg-emerald-500/[0.08]'
    : 'border-emerald-500/15 bg-emerald-500/[0.04] hover:bg-emerald-500/[0.08]';

  return `
    <div class="det-item flex items-center gap-2.5 px-3 py-2.5 rounded-xl border ${rowCls} cursor-pointer transition-all duration-200 group" data-idx="${i}">
      <span class="text-[10px] font-heading text-text-muted/60 w-5 text-right select-none">${i + 1}</span>
      <span class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2 py-0.5 rounded-full font-medium select-none">${d.label}</span>
      <div class="flex items-center gap-1.5 ml-auto">
        <span class="text-[11px] text-text-primary font-heading tabular-nums">${formatTime(d.start)}</span>
        <span class="text-text-muted/40 text-[10px]">\u2192</span>
        <span class="text-[11px] text-text-primary font-heading tabular-nums">${formatTime(d.end)}</span>
      </div>
      <span class="text-[10px] text-text-muted font-heading tabular-nums bg-surface-200/40 px-1.5 py-0.5 rounded">${durationSec}s</span>
      ${d.score != null ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-heading font-medium tabular-nums ring-1 ${scoreColor} ${scoreBg}">${(d.score * 100).toFixed(0)}%</span>` : ''}
      <button class="text-primary-light hover:text-white cursor-pointer transition-colors duration-200" data-action="preview" title="Jump to last 5s">
        <svg class="w-4 h-4 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
      </button>
    </div>`;
}

// -- Timeline --
function resizeTimeline() {
  if (!timelineCanvas) return;
  const rect = timelineCanvas.getBoundingClientRect();
  if (rect.width === 0) return;
  timelineCanvas.width = rect.width * devicePixelRatio;
  timelineCanvas.height = rect.height * devicePixelRatio;
}

function startTimelineLoop() {
  if (animFrame) cancelAnimationFrame(animFrame);
  function draw() {
    animFrame = requestAnimationFrame(draw);
    if (!timelineCanvas || !_duration) return;

    const ctx = timelineCanvas.getContext('2d');
    const w = timelineCanvas.width;
    const h = timelineCanvas.height;
    const dpr = devicePixelRatio;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.02)';
    ctx.fillRect(0, 0, w, h);

    // Detection segments
    for (let i = 0; i < _detections.length; i++) {
      const d = _detections[i];
      const x1 = (d.start / _duration) * w;
      const x2 = (d.end / _duration) * w;
      if (_selectedIdx === i) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.7)';
      } else {
        const grad = ctx.createLinearGradient(x1, 0, x1, h);
        grad.addColorStop(0, 'rgba(34, 197, 94, 0.5)');
        grad.addColorStop(1, 'rgba(34, 197, 94, 0.25)');
        ctx.fillStyle = grad;
      }
      ctx.beginPath();
      ctx.roundRect(x1, 2 * dpr, Math.max(x2 - x1, 2), h - 4 * dpr, 3 * dpr);
      ctx.fill();
    }

    // Playhead
    if (videoEl && !isNaN(videoEl.currentTime)) {
      const px = (videoEl.currentTime / _duration) * w;
      ctx.fillStyle = '#F97316';
      ctx.shadowColor = 'rgba(249,115,22,0.6)';
      ctx.shadowBlur = 6 * dpr;
      ctx.fillRect(px - 1 * dpr, 0, 2 * dpr, h);
      ctx.shadowBlur = 0;
    }
  }
  draw();
}
