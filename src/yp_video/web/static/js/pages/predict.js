/**
 * Predict page — TAD inference with multi-video selection. Result browsing
 * and editing live on the Review page.
 */
import { api, API, SSEClient, card, pageHeader, sectionTitle, stepBadge, btnPrimary, btnSmall, showToast, showConfirm, emptyState, inputCls, selectCls, kindTabs, updateKindTabs, badges, renderJobProgress, filterChips, bindFilterChips } from '../shared.js';

let sseClients = [];
let state = { videos: [], checkpoints: [], jobs: [] };
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
  return document.getElementById('pred-model')?.value || 'base';
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
               <option value="base" selected>ViT-B (768d)</option>
               <option value="large">ViT-L (1024d)</option>
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

    </div>`;

  loadData();
  bindEvents();
}

export function activate() {}

export function deactivate() {
  sseClients.forEach(c => c.stop());
  sseClients = [];
}

function bindEvents() {
  document.getElementById('pred-start').addEventListener('click', startPrediction);
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

  const model = document.getElementById('pred-model')?.value || 'base';
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
            const msg = data.message || 'Prediction complete!';
            showToast(`${msg} <a href="#/review" class="underline ml-1">Open Review →</a>`, 'success');
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
