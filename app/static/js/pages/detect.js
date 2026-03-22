/**
 * Detect page — VLM rally detection + vlm_to_rally conversion.
 */
import { api, SSEClient, card, pageHeader, sectionTitle, stepBadge, btnPrimary, btnSecondary, btnSmall, createProgressBar, showToast, emptyState, inputCls } from '../shared.js';

let sseClient = null;
let state = { videos: [], jobId: null };

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Detect', 'Run VLM rally detection on cut videos')}

      ${card(`
        <div class="space-y-4">
          ${sectionTitle(
            'Videos',
            '',
            `${btnSmall('Select All', 'id="det-select-all"')}
             ${btnSmall('Undetected', 'id="det-select-undetected"', 'primary')}`
          )}
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
              <input id="det-batch" type="number" value="32" min="1" max="128" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider">Clip Duration</label>
              <input id="det-clip" type="number" value="6" min="1" step="0.5" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider">Slide Interval</label>
              <input id="det-slide" type="number" value="3" min="0.5" step="0.5" class="w-full ${inputCls}">
            </div>
          </div>
          <div class="ml-10 flex items-center gap-3 pt-1">
            ${btnPrimary('Start Detection', 'id="det-start"')}
          </div>
        </div>
      `)}

      <div id="det-progress" class="hidden">
        ${card(`
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2.5">
                <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse-dot"></span>
                <h3 class="text-sm font-heading font-semibold text-text-primary">Progress</h3>
              </div>
              <span id="det-progress-pct" class="text-xs font-heading text-primary-light tabular-nums"></span>
            </div>
            <div id="det-progress-bar"></div>
            <p id="det-progress-msg" class="text-xs text-text-muted"></p>
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

export function destroy() {
  if (sseClient) { sseClient.stop(); sseClient = null; }
  state = { videos: [], jobId: null };
}

function bindEvents() {
  document.getElementById('det-start').addEventListener('click', startDetection);
  document.getElementById('det-convert').addEventListener('click', convertDetections);
  document.getElementById('det-select-all').addEventListener('click', () => {
    state.videos.forEach(v => v.selected = true);
    renderVideos();
  });
  document.getElementById('det-select-undetected').addEventListener('click', () => {
    state.videos.forEach(v => v.selected = !v.has_detection);
    renderVideos();
  });
}

async function loadVideos() {
  try {
    const videos = await api('/detect/videos');
    state.videos = videos.map(v => ({ ...v, selected: !v.has_detection }));
    renderVideos();
  } catch (e) {
    showToast(`Failed to load videos: ${e.message}`, 'error');
  }
}

function renderVideos() {
  const el = document.getElementById('det-videos');
  if (state.videos.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>',
      'No cut videos found',
      'Cut some videos first'
    );
    return;
  }

  el.innerHTML = state.videos.map((v, i) => `
    <div class="group flex items-center gap-3 p-2.5 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200">
      <input type="checkbox" data-idx="${i}" class="det-check cursor-pointer accent-primary w-3.5 h-3.5" ${v.selected ? 'checked' : ''}>
      <span class="text-sm text-text-primary flex-1 truncate group-hover:text-white transition-colors duration-200">${v.name}</span>
      ${v.has_detection
        ? '<span class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>detected</span>'
        : '<span class="inline-flex items-center gap-1.5 text-[11px] text-text-muted bg-white/5 ring-1 ring-white/10 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>pending</span>'}
    </div>
  `).join('');

  el.querySelectorAll('.det-check').forEach(cb => {
    cb.addEventListener('change', (e) => {
      state.videos[parseInt(e.target.dataset.idx)].selected = e.target.checked;
    });
  });
}

async function startDetection() {
  const selected = state.videos.filter(v => v.selected).map(v => v.name);
  if (selected.length === 0) return showToast('No videos selected', 'warning');

  const btn = document.getElementById('det-start');
  btn.disabled = true;

  try {
    const res = await api('/detect/start', {
      method: 'POST',
      body: {
        videos: selected,
        batch_size: parseInt(document.getElementById('det-batch').value),
        clip_duration: parseFloat(document.getElementById('det-clip').value),
        slide_interval: parseFloat(document.getElementById('det-slide').value),
      },
    });

    state.jobId = res.id;
    document.getElementById('det-progress').classList.remove('hidden');

    sseClient = new SSEClient(`/api/jobs/${res.id}/events`, {
      onMessage: (data) => {
        const pct = Math.round((data.progress || 0) * 100);
        document.getElementById('det-progress-pct').textContent = `${pct}%`;
        document.getElementById('det-progress-bar').innerHTML = createProgressBar(data.progress);
        document.getElementById('det-progress-msg').textContent = data.message || '';

        if (data.status === 'completed') {
          sseClient?.stop();
          showToast('Detection complete!', 'success');
          btn.disabled = false;
          loadVideos();
        } else if (data.status === 'failed') {
          sseClient?.stop();
          showToast(`Detection failed: ${data.error || 'Unknown error'}`, 'error');
          btn.disabled = false;
        }
      },
      onError: () => { btn.disabled = false; },
    }).start();
  } catch (e) {
    showToast(`Failed to start detection: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

async function convertDetections() {
  const btn = document.getElementById('det-convert');
  btn.disabled = true;
  btn.textContent = 'Converting...';

  try {
    const res = await api('/detect/convert', {
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
