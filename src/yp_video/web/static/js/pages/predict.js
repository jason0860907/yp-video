/**
 * Predict page — TAD inference with result visualization and video review.
 */
import { api, SSEClient, formatTime, card, pageHeader, sectionTitle, stepBadge, btnPrimary, btnSmall, createProgressBar, showToast, emptyState, inputCls, selectCls } from '../shared.js';

let sseClient = null;
let state = { videos: [], checkpoints: [], results: [] };
let videoEl = null;
let timelineCanvas = null;
let animFrame = null;
let _selectedIdx = -1;
let _detections = [];
let _duration = 0;

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
  // Scroll selected into view
  const el = document.querySelector(`.det-item[data-idx="${_selectedIdx}"]`);
  if (el) el.scrollIntoView({ block: 'nearest' });
}

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
    _detections = results.map(r => ({
      start: r.start ?? r.segment?.[0] ?? 0,
      end: r.end ?? r.segment?.[1] ?? 0,
      label: r.label || 'rally',
      score: r.confidence ?? r.score ?? null,
    })).sort((a, b) => a.start - b.start);
    _selectedIdx = -1;
    _duration = data.duration || 0;

    // Extract video filename from the full path in meta
    const videoPath = data.video || '';
    const videoName = videoPath.split('/').pop() || '';

    detailEl.innerHTML = `
      <div class="flex flex-col lg:flex-row gap-5">
        <!-- Left: Video + Timeline -->
        <div class="flex-1 min-w-0 space-y-4">
          ${card(`
            <div class="space-y-4">
              <div class="flex items-center justify-between">
                ${sectionTitle(videoName || name, `${_detections.length} detections`)}
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
                  </div>
                </div>
              </div>
            </div>
          `)}
        </div>

        <!-- Right: Detections list -->
        <div class="lg:w-[400px] lg:flex-shrink-0">
          ${card(`
            <div class="space-y-3">
              ${sectionTitle('Detections <span id="pred-det-count" class="text-text-muted font-normal">(' + _detections.length + ')</span>')}
              <div class="h-px bg-border"></div>
              <div id="pred-det-list" class="space-y-1 max-h-[60vh] overflow-y-auto pr-1 scrollbar-thin"></div>
            </div>
          `)}
        </div>
      </div>`;

    // Setup video
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
      // Auto-pause at end of selected segment
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

    document.getElementById('pred-detail-close').addEventListener('click', () => {
      if (videoEl && !videoEl.paused) videoEl.pause();
      videoEl = null;
      timelineCanvas = null;
      _detections = [];
      _selectedIdx = -1;
      detailEl.classList.add('hidden');
    });

    renderDetections();
    resizeTimeline();
    startTimelineLoop();
    detailEl.scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    showToast(`Failed to load result: ${e.message}`, 'error');
  }
}

function renderDetections() {
  const el = document.getElementById('pred-det-list');
  if (!el) return;

  el.innerHTML = _detections.map((d, i) => {
    const selected = _selectedIdx === i;
    const durationSec = (d.end - d.start).toFixed(1);
    const scoreColor = d.score > 0.7 ? 'text-emerald-400' : d.score > 0.4 ? 'text-amber-400' : 'text-text-muted';
    const scoreBg = d.score > 0.7 ? 'bg-emerald-500/10 ring-emerald-500/20' : d.score > 0.4 ? 'bg-amber-500/10 ring-amber-500/20' : 'bg-white/5 ring-white/10';
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
        <button class="det-preview text-primary-light hover:text-white cursor-pointer transition-colors duration-200" data-idx="${i}" title="Jump to last 5s">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
        </button>
      </div>`;
  }).join('');

  // Click row → seek to start and play
  el.querySelectorAll('.det-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (e.target.closest('.det-preview')) return;
      const idx = parseInt(item.dataset.idx);
      _selectedIdx = idx;
      const d = _detections[idx];
      videoEl.currentTime = d.start;
      videoEl.play();
      renderDetections();
    });
  });
  // >> button → jump to last 5s of rally
  el.querySelectorAll('.det-preview').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(e.currentTarget.dataset.idx);
      _selectedIdx = idx;
      const d = _detections[idx];
      videoEl.currentTime = Math.max(d.start, d.end - 5);
      videoEl.play();
      renderDetections();
    });
  });
}

// ── Timeline ──
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
