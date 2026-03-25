/**
 * Predict page — TAD inference with result visualization.
 */
import { api, SSEClient, formatTime, card, pageHeader, sectionTitle, stepBadge, btnPrimary, btnSmall, createProgressBar, showToast, emptyState, inputCls, selectCls } from '../shared.js';

let sseClient = null;
let state = { videos: [], checkpoints: [], results: [] };

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Predict', 'Run TAD inference on videos')}

      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(1, 'accent')}
            <div>
              ${sectionTitle('Run Prediction', 'Select a video and checkpoint to run TAD inference')}
            </div>
          </div>
          <div class="ml-10 grid grid-cols-2 gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Video</label>
              <select id="pred-video" class="w-full ${selectCls}">
                <option value="">Select video...</option>
              </select>
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Checkpoint</label>
              <select id="pred-checkpoint" class="w-full ${selectCls}">
                <option value="">Select checkpoint...</option>
              </select>
            </div>
          </div>
          <div class="ml-10 grid grid-cols-3 gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Threshold</label>
              <input id="pred-threshold" type="number" value="0.3" min="0" max="1" step="0.05" class="w-full ${inputCls}">
            </div>
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
          </div>
          <div id="pred-progress" class="ml-10 hidden space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2.5">
                <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse-dot"></span>
                <span class="text-xs font-heading text-text-secondary">Running inference</span>
              </div>
            </div>
            <div id="pred-bar"></div>
            <p id="pred-msg" class="text-xs text-text-muted"></p>
          </div>
        </div>
      `)}

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

export function activate() {}
export function deactivate() {}

function bindEvents() {
  document.getElementById('pred-start').addEventListener('click', startPrediction);
  document.getElementById('pred-refresh').addEventListener('click', loadResults);
}

async function loadData() {
  try {
    const [videos, checkpoints] = await Promise.all([
      api('/predict/videos'),
      api('/train/checkpoints'),
    ]);
    state.videos = videos;
    state.checkpoints = checkpoints;

    const vidSel = document.getElementById('pred-video');
    videos.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.name;
      opt.textContent = `${v.name}${v.has_prediction ? ' (done)' : ''}`;
      vidSel.appendChild(opt);
    });

    const cpSel = document.getElementById('pred-checkpoint');
    checkpoints.forEach(cp => {
      const opt = document.createElement('option');
      opt.value = cp.path;
      opt.textContent = `${cp.name} (${cp.size_mb.toFixed(1)} MB)`;
      cpSel.appendChild(opt);
    });

    await loadResults();
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}

async function loadResults() {
  try {
    state.results = await api('/predict/results');
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

  el.querySelectorAll('[data-name]').forEach(item => {
    item.addEventListener('click', () => viewResult(item.dataset.name));
  });
}

async function viewResult(name) {
  try {
    const data = await api(`/predict/results/${encodeURIComponent(name)}`);
    const detailEl = document.getElementById('pred-detail');
    detailEl.classList.remove('hidden');

    const results = data.results || [];
    detailEl.innerHTML = card(`
      <div class="space-y-4">
        ${sectionTitle(
          name,
          `${results.length} prediction${results.length !== 1 ? 's' : ''} found`,
          btnSmall('Close', 'id="pred-detail-close"')
        )}
        ${results.length === 0 ? emptyState(
          '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"/></svg>',
          'No predictions found'
        ) : `
          <div class="space-y-1.5">
            ${results.map((r, i) => {
              const start = r.start ?? r.segment?.[0] ?? 0;
              const end = r.end ?? r.segment?.[1] ?? 0;
              const duration = (end - start).toFixed(1);
              const scoreColor = r.score > 0.7 ? 'text-emerald-400' : r.score > 0.4 ? 'text-amber-400' : 'text-text-muted';
              const scoreBg = r.score > 0.7 ? 'bg-emerald-500/10 ring-emerald-500/20' : r.score > 0.4 ? 'bg-amber-500/10 ring-amber-500/20' : 'bg-white/5 ring-white/10';
              return `
                <div class="flex items-center gap-3 p-3 rounded-xl bg-surface-50/50 border border-border hover:bg-surface-100/50 transition-colors duration-150">
                  <span class="w-6 h-6 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-[10px] font-heading font-bold text-text-muted flex-shrink-0">${i + 1}</span>
                  <div class="flex items-center gap-2 flex-1 min-w-0">
                    <span class="text-xs text-text-primary font-heading tabular-nums whitespace-nowrap">${formatTime(start)}</span>
                    <svg class="w-3 h-3 text-text-muted flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6"/></svg>
                    <span class="text-xs text-text-primary font-heading tabular-nums whitespace-nowrap">${formatTime(end)}</span>
                    <span class="text-[11px] text-text-muted font-heading tabular-nums">${duration}s</span>
                  </div>
                  ${r.label ? `<span class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2.5 py-0.5 rounded-full font-medium">${r.label}</span>` : ''}
                  ${r.score != null ? `<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-heading font-medium tabular-nums ring-1 ${scoreColor} ${scoreBg}">${(r.score * 100).toFixed(1)}%</span>` : ''}
                </div>`;
            }).join('')}
          </div>
        `}
      </div>
    `);

    const closeBtn = document.getElementById('pred-detail-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        detailEl.classList.add('hidden');
      });
    }

    detailEl.scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    showToast(`Failed to load result: ${e.message}`, 'error');
  }
}

async function startPrediction() {
  const video = document.getElementById('pred-video').value;
  const checkpoint = document.getElementById('pred-checkpoint').value;
  if (!video || !checkpoint) return showToast('Select video and checkpoint', 'warning');

  const btn = document.getElementById('pred-start');
  btn.disabled = true;

  try {
    const res = await api('/predict/start', {
      method: 'POST',
      body: {
        video,
        checkpoint,
        threshold: parseFloat(document.getElementById('pred-threshold').value),
        device: document.getElementById('pred-device').value,
        cut_rallies: document.getElementById('pred-cut').checked,
      },
    });

    document.getElementById('pred-progress').classList.remove('hidden');
    sseClient = new SSEClient(`/api/jobs/${res.id}/events`, {
      onMessage: (data) => {
        document.getElementById('pred-bar').innerHTML = createProgressBar(data.progress);
        document.getElementById('pred-msg').textContent = data.message || '';
        if (data.status === 'completed' || data.status === 'failed') {
          sseClient?.stop();
          btn.disabled = false;
          showToast(data.status === 'completed' ? 'Prediction complete!' : `Failed: ${data.error}`, data.status === 'completed' ? 'success' : 'error');
          loadResults();
        }
      },
      onError: () => { btn.disabled = false; },
    }).start();
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
    btn.disabled = false;
  }
}
