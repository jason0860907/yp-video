/**
 * Detect page — VLM rally detection + vlm_to_rally conversion.
 */
import { api, API, SSEClient, card, pageHeader, sectionTitle, stepBadge, btnPrimary, btnSecondary, btnSmall, showToast, emptyState, inputCls, renderJobProgress } from '../shared.js';

let sseClients = [];
let state = { videos: [], jobs: [], kindFilter: 'all' };

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('VLM Predict', 'Run VLM rally detection on cut videos')}

      ${card(`
        <div class="space-y-4">
          ${sectionTitle(
            'Videos',
            '',
            `${btnSmall('Select All', 'id="det-select-all"')}
             ${btnSmall('Deselect All', 'id="det-deselect-all"')}
             ${btnSmall('Undetected', 'id="det-select-undetected"', 'primary')}`
          )}
          <div class="inline-flex rounded-lg border border-border bg-surface-100 p-0.5" role="tablist" aria-label="Cut kind">
            <button type="button" data-kind="all"       class="det-kind-tab px-3 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="true">All <span class="opacity-60 ml-1" data-count="all">0</span></button>
            <button type="button" data-kind="broadcast" class="det-kind-tab px-3 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="false">Broadcast <span class="opacity-60 ml-1" data-count="broadcast">0</span></button>
            <button type="button" data-kind="sideline"  class="det-kind-tab px-3 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="false">Sideline <span class="opacity-60 ml-1" data-count="sideline">0</span></button>
          </div>
          <div id="det-videos" class="space-y-0.5 max-h-72 overflow-y-auto pr-1"></div>
        </div>
      `)}

      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(1)}
            ${sectionTitle('Detection Settings', 'Configure VLM sliding-window parameters')}
          </div>
          <div class="ml-10 grid grid-cols-3 gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider">Batch Size</label>
              <input id="det-batch" type="number" value="16" min="1" max="128" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider">Clip Duration</label>
              <input id="det-clip" type="number" value="6" min="1" step="0.5" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider">Slide Interval</label>
              <input id="det-slide" type="number" value="2" min="0.5" step="0.5" class="w-full ${inputCls}">
            </div>
          </div>
          <div class="ml-10 flex items-center gap-3 pt-1">
            ${btnPrimary('Start Detection', 'id="det-start"')}
            <span id="det-count" class="text-xs text-text-muted font-heading tabular-nums ml-auto"></span>
          </div>
        </div>
      `)}

      <div id="det-progress" class="hidden">
        ${card(`
          <div class="space-y-3">
            <div class="flex items-center gap-2.5 mb-3">
              <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse-dot"></span>
              <h3 class="text-sm font-heading font-semibold text-text-primary">Progress</h3>
            </div>
            <div id="det-jobs-progress" class="space-y-3"></div>
            <div id="det-retry-wrap" class="hidden pt-1">
              ${btnSmall('Retry Failed', 'id="det-retry-failed"', 'primary')}
            </div>
          </div>
        `)}
      </div>

      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(2, 'accent')}
            ${sectionTitle('Convert to Rally', 'Merge clip detections into rally annotations (voting + smoothing)')}
          </div>
          <div class="ml-10 grid grid-cols-2 gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider">Min Duration (s)</label>
              <input id="det-min-dur" type="number" value="3" min="0" step="0.5" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider">Min Score</label>
              <input id="det-min-score" type="number" value="0.5" min="0" max="1" step="0.1" class="w-full ${inputCls}">
            </div>
          </div>
          <div class="ml-10 flex items-center gap-3 pt-1">
            ${btnSecondary('Convert Detections', 'id="det-convert"')}
          </div>
        </div>
      `)}

    </div>`;

  loadVideos();
  bindEvents();
}

export function activate() {}
export function deactivate() {
  sseClients.forEach(c => c.stop());
  sseClients = [];
}

function bindEvents() {
  document.getElementById('det-start').addEventListener('click', startDetection);
  document.getElementById('det-retry-failed').addEventListener('click', retryFailed);
  document.getElementById('det-convert').addEventListener('click', convertDetections);
  // Bulk-select buttons act on the currently visible (filtered) tab so they
  // never toggle videos the user can't see.
  document.getElementById('det-select-all').addEventListener('click', () => {
    visibleVideos().forEach(v => v.selected = true);
    renderVideos();
  });
  document.getElementById('det-deselect-all').addEventListener('click', () => {
    visibleVideos().forEach(v => v.selected = false);
    renderVideos();
  });
  document.getElementById('det-select-undetected').addEventListener('click', () => {
    visibleVideos().forEach(v => v.selected = !v.has_detection);
    renderVideos();
  });
  document.querySelectorAll('.det-kind-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      state.kindFilter = btn.dataset.kind;
      renderVideos();
    });
  });
}

function visibleVideos() {
  if (state.kindFilter === 'all') return state.videos;
  return state.videos.filter(v => v.kind === state.kindFilter);
}

async function loadVideos() {
  try {
    const [videos, vllmStatus] = await Promise.all([
      api(API.system.videos()),
      api(API.system.vllmStatus).catch(() => null),
    ]);
    state.videos = videos.map(v => ({ ...v, selected: !v.has_detection }));
    renderVideos();
    if (vllmStatus?.max_num_seqs) {
      document.getElementById('det-batch').value = vllmStatus.max_num_seqs;
    }
  } catch (e) {
    showToast(`Failed to load videos: ${e.message}`, 'error');
  }
}

function renderVideos() {
  const el = document.getElementById('det-videos');

  // Tab counts always reflect the full set, not the filtered view.
  const counts = {
    all: state.videos.length,
    broadcast: state.videos.filter(v => v.kind === 'broadcast').length,
    sideline: state.videos.filter(v => v.kind === 'sideline').length,
  };
  document.querySelectorAll('.det-kind-tab').forEach(btn => {
    const k = btn.dataset.kind;
    const active = k === state.kindFilter;
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    btn.classList.toggle('bg-primary', active);
    btn.classList.toggle('text-white', active);
    btn.classList.toggle('text-text-secondary', !active);
    btn.classList.toggle('hover:bg-white/[0.04]', !active);
    const cnt = btn.querySelector(`[data-count="${k}"]`);
    if (cnt) cnt.textContent = counts[k];
  });

  const visible = visibleVideos();
  if (visible.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>',
      state.videos.length ? `No ${state.kindFilter} cuts` : 'No cut videos found',
      state.videos.length ? 'Switch tab or cut more videos' : 'Cut some videos first'
    );
    updateDetCount();
    return;
  }

  el.innerHTML = visible.map((v) => {
    const i = state.videos.indexOf(v);
    const kindBadge = v.kind === 'sideline'
      ? '<span class="inline-flex items-center text-[10px] text-amber-300 bg-amber-500/10 ring-1 ring-amber-500/20 px-1.5 py-0.5 rounded font-heading uppercase tracking-wide">side</span>'
      : '<span class="inline-flex items-center text-[10px] text-sky-300 bg-sky-500/10 ring-1 ring-sky-500/20 px-1.5 py-0.5 rounded font-heading uppercase tracking-wide">cast</span>';
    return `
    <div class="group flex items-center gap-3 p-2.5 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200">
      <input type="checkbox" data-idx="${i}" class="det-check cursor-pointer accent-primary w-3.5 h-3.5" ${v.selected ? 'checked' : ''}>
      ${kindBadge}
      <span class="text-sm text-text-primary flex-1 truncate group-hover:text-white transition-colors duration-200">${v.name}</span>
      ${v.has_detection
        ? '<span class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>detected</span>'
        : '<span class="inline-flex items-center gap-1.5 text-[11px] text-text-muted bg-white/5 ring-1 ring-white/10 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>pending</span>'}
    </div>`;
  }).join('');

  el.querySelectorAll('.det-check').forEach(cb => {
    cb.addEventListener('change', (e) => {
      state.videos[parseInt(e.target.dataset.idx)].selected = e.target.checked;
      updateDetCount();
    });
  });
  updateDetCount();
}

function updateDetCount() {
  const el = document.getElementById('det-count');
  if (!el) return;
  const visible = visibleVideos();
  const sel = visible.filter(v => v.selected).length;
  el.textContent = visible.length ? `${sel} / ${visible.length} selected` : '';
}

async function startDetection() {
  const selected = state.videos.filter(v => v.selected).map(v => v.name);
  if (selected.length === 0) return showToast('No videos selected', 'warning');

  const btn = document.getElementById('det-start');
  btn.disabled = true;
  document.getElementById('det-retry-wrap').classList.add('hidden');

  // Stop any existing SSE clients
  sseClients.forEach(c => c.stop());
  sseClients = [];

  try {
    const job = await api(API.detect.start, {
      method: 'POST',
      body: {
        videos: selected,
        batch_size: parseInt(document.getElementById('det-batch').value),
        clip_duration: parseFloat(document.getElementById('det-clip').value),
        slide_interval: parseFloat(document.getElementById('det-slide').value),
        min_duration: parseFloat(document.getElementById('det-min-dur').value),
        min_score: parseFloat(document.getElementById('det-min-score').value),
      },
    });

    state.jobs = [job];
    document.getElementById('det-progress').classList.remove('hidden');
    renderJobsProgress();

    const client = new SSEClient(API.jobs.eventsSSE(job.id), {
      onMessage: (data) => {
        state.jobs = [data];
        renderJobsProgress();

        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
          client.stop();
          if (data.status === 'failed') {
            showToast(`Detection failed: ${data.error || 'Unknown error'}`, 'error');
          } else {
            showToast(data.message || 'Detection complete!', 'success');
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
    showToast(`Failed to start detection: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

function retryFailed() {
  startDetection();
}

function renderJobsProgress() {
  const el = document.getElementById('det-jobs-progress');
  if (!el) return;
  el.innerHTML = state.jobs.map(job => renderJobProgress(job)).join('');
}

async function convertDetections() {
  const btn = document.getElementById('det-convert');
  btn.disabled = true;
  btn.textContent = 'Converting...';

  try {
    const res = await api(API.detect.convert, {
      method: 'POST',
      body: {
        min_duration: parseFloat(document.getElementById('det-min-dur').value),
        min_score: parseFloat(document.getElementById('det-min-score').value),
      },
    });
    showToast(`Converted ${res.count} videos`, 'success');
  } catch (e) {
    showToast(`Conversion failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Convert Detections';
  }
}
