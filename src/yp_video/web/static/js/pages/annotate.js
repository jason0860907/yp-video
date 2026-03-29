/**
 * Annotate page — Rally annotation with timeline visualization.
 */
import { api, formatTime, parseTime, card, pageHeader, sectionTitle, btnSmall, showToast, emptyState, selectCls, kbdHint } from '../shared.js';

let state = { results: [], annotations: [], videoName: '', duration: 0 };
let videoEl = null;
let timelineCanvas = null;
let animFrame = null;
let _markStart = null;
let _selectedIdx = -1;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-5">
      ${pageHeader('Annotate', 'Review and edit rally annotations', `
        <select id="ann-results" class="${selectCls}">
          <option value="">Select result file...</option>
        </select>
        ${btnSmall('Load', 'id="ann-load"', 'primary')}
      `)}

      <div class="flex flex-col lg:flex-row gap-5">
        <!-- Left: Video + Timeline -->
        <div class="flex-1 min-w-0 space-y-4">
          ${card(`
            <div class="space-y-4">
              <div class="relative bg-black rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-lg shadow-black/30">
                <video id="ann-player" class="w-full max-h-[45vh]" controls></video>
              </div>

              <div class="space-y-2">
                <div class="relative rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-inner" style="background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)">
                  <canvas id="ann-timeline" class="w-full h-12 cursor-pointer" title="Click to seek"></canvas>
                </div>
                <div class="flex items-center justify-between px-0.5">
                  <span id="ann-time" class="text-sm font-heading text-text-primary tabular-nums bg-surface-200/50 px-2.5 py-1 rounded-lg border border-border">00:00</span>
                  <div class="flex items-center gap-2">
                    ${btnSmall('Start [', 'id="ann-mark-start"', 'primary')}
                    ${btnSmall('End ]', 'id="ann-mark-end"', 'primary')}
                    ${btnSmall('Rally \u21b5', 'id="ann-add-rally"', 'success')}
                    ${btnSmall('Non-Rally', 'id="ann-add-nonrally"')}
                  </div>
                </div>
              </div>

              <div id="ann-mark-info" class="hidden p-3 rounded-xl bg-primary/10 border border-primary/20 flex items-center gap-2.5">
                <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse"></span>
                <span class="text-xs text-primary-light">Start marked at <strong id="ann-mark-time" class="font-heading">0:00</strong> &mdash; press <kbd class="px-1.5 py-0.5 rounded bg-surface-200 border border-border text-[10px] font-heading text-text-secondary ml-0.5 mr-0.5">]</kbd> to set end</span>
              </div>
            </div>
          `)}
        </div>

        <!-- Right: Annotations list (fixed width) -->
        <div class="lg:w-[420px] lg:flex-shrink-0">
          ${card(`
            <div class="space-y-4">
              ${sectionTitle(
                'Annotations <span id="ann-count" class="text-text-muted font-normal">(0)</span>',
                '',
                `${btnSmall('Save', 'id="ann-save"', 'success')}
                 ${btnSmall('Clear', 'id="ann-clear"', 'danger')}`
              )}
              <div class="h-px bg-border"></div>
              <div id="ann-list" class="space-y-1.5 max-h-[55vh] overflow-y-auto pr-1 scrollbar-thin"></div>
            </div>
          `)}
        </div>
      </div>

      ${kbdHint([
        ['Space', 'play/pause'],
        ['[', 'mark start'],
        [']', 'mark end'],
        ['Enter', 'rally'],
        ['\u2190 \u2192', '\u00b15s'],
        ['T', 'rally/non-rally'],
        ['Del', 'remove'],
      ])}
    </div>`;

  videoEl = document.getElementById('ann-player');
  timelineCanvas = document.getElementById('ann-timeline');
  loadResults();
  bindEvents();
  activate();
}

export function activate() {
  document.addEventListener('keydown', handleKeydown);
  window.addEventListener('resize', resizeTimeline);
  startTimelineLoop();
  resizeTimeline();
}

export function deactivate() {
  document.removeEventListener('keydown', handleKeydown);
  window.removeEventListener('resize', resizeTimeline);
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
  if (videoEl && !videoEl.paused) videoEl.pause();
}

function bindEvents() {
  document.getElementById('ann-load').addEventListener('click', loadFile);
  document.getElementById('ann-mark-start').addEventListener('click', markStart);
  document.getElementById('ann-mark-end').addEventListener('click', markEnd);
  document.getElementById('ann-add-rally').addEventListener('click', () => addAnnotation('rally'));
  document.getElementById('ann-add-nonrally').addEventListener('click', () => addAnnotation('non-rally'));
  document.getElementById('ann-save').addEventListener('click', saveAnnotations);
  document.getElementById('ann-clear').addEventListener('click', clearAll);

  videoEl.addEventListener('loadedmetadata', () => {
    state.duration = videoEl.duration;
    resizeTimeline();
  });
  videoEl.addEventListener('timeupdate', () => {
    document.getElementById('ann-time').textContent = formatTime(videoEl.currentTime);
    // Auto-pause when reaching end of selected segment
    if (_selectedIdx >= 0 && _selectedIdx < state.annotations.length) {
      const a = state.annotations[_selectedIdx];
      if (!videoEl.paused && videoEl.currentTime >= a.end) {
        videoEl.pause();
        videoEl.currentTime = a.end;
        _selectedIdx = -1;
      }
    }
  });

  timelineCanvas.addEventListener('click', (e) => {
    if (!state.duration) return;
    const rect = timelineCanvas.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    videoEl.currentTime = ratio * state.duration;
  });

}

function handleKeydown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  switch (e.key) {
    case ' ': e.preventDefault(); videoEl.paused ? videoEl.play() : videoEl.pause(); break;
    case '[': e.preventDefault(); markStart(); break;
    case ']': e.preventDefault(); markEnd(); break;
    case 'Enter': e.preventDefault(); addAnnotation('rally'); break;
    case 'ArrowLeft': e.preventDefault(); videoEl.currentTime = Math.max(0, videoEl.currentTime - 5); break;
    case 'ArrowRight': e.preventDefault(); videoEl.currentTime += 5; break;
    case 't': case 'T':
      if (_selectedIdx >= 0 && _selectedIdx < state.annotations.length) {
        const a = state.annotations[_selectedIdx];
        a.label = a.label === 'rally' ? 'non-rally' : 'rally';
        renderAnnotations();
      }
      break;
    case 'Delete': case 'Backspace':
      if (_selectedIdx >= 0 && _selectedIdx < state.annotations.length) {
        state.annotations.splice(_selectedIdx, 1);
        _selectedIdx = -1;
        renderAnnotations();
      }
      break;
  }
}

async function loadResults() {
  try {
    const results = await api('/annotate/results');
    state.results = results;
    const sel = document.getElementById('ann-results');
    results.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.name;
      const tag = r.source.includes('annotation') ? '✅' : '⚡';
      opt.textContent = `${tag} ${r.name}`;
      sel.appendChild(opt);
    });
  } catch (e) {
    showToast(`Failed to load results: ${e.message}`, 'error');
  }
}

async function loadFile() {
  const name = document.getElementById('ann-results').value;
  if (!name) return;

  try {
    const data = await api(`/annotate/results/${encodeURIComponent(name)}`);
    const videoPath = data.video || data.source_video || data.metadata?.video || '';
    state.videoName = videoPath;
    if (videoPath) {
      videoEl.src = `/api/annotate/video/${encodeURIComponent(videoPath)}`;
      videoEl.load();
    }

    state.annotations = (data.results || []).map(r => ({
      start: r.start ?? r.start_time ?? 0,
      end: r.end ?? r.end_time ?? 0,
      label: r.label || (r.in_rally ? 'rally' : 'non-rally'),
    }));
    state.annotations.sort((a, b) => a.start - b.start);
    renderAnnotations();
    showToast(`Loaded ${state.annotations.length} annotations`, 'success');
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}

function markStart() {
  if (!videoEl.src) return;
  _markStart = videoEl.currentTime;
  document.getElementById('ann-mark-info').classList.remove('hidden');
  document.getElementById('ann-mark-time').textContent = formatTime(_markStart);
}

function markEnd() {
  if (_markStart == null) return;
  const end = videoEl.currentTime;
  if (end <= _markStart) return showToast('End must be after start', 'warning');
  state.annotations.push({ start: _markStart, end, label: 'rally' });
  state.annotations.sort((a, b) => a.start - b.start);
  _markStart = null;
  document.getElementById('ann-mark-info').classList.add('hidden');
  renderAnnotations();
}

function addAnnotation(label) {
  if (_markStart == null) return showToast('Mark start first with [', 'warning');
  const end = videoEl.currentTime;
  if (end <= _markStart) return showToast('End must be after start', 'warning');
  state.annotations.push({ start: _markStart, end, label });
  state.annotations.sort((a, b) => a.start - b.start);
  _markStart = null;
  document.getElementById('ann-mark-info').classList.add('hidden');
  renderAnnotations();
}

function renderAnnotations() {
  const el = document.getElementById('ann-list');
  document.getElementById('ann-count').textContent = `(${state.annotations.length})`;

  if (state.annotations.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>',
      'No annotations',
      'Use [ ] to mark segments'
    );
    return;
  }

  el.innerHTML = state.annotations.map((a, i) => {
    const isRally = a.label === 'rally';
    const selected = _selectedIdx === i;
    const rowCls = isRally
      ? (selected ? 'border-emerald-500/40 bg-emerald-500/[0.08]' : 'border-emerald-500/15 bg-emerald-500/[0.04] hover:bg-emerald-500/[0.08]')
      : (selected ? 'border-primary/40 bg-primary/[0.08]' : 'border-border bg-surface-50/30 hover:bg-white/[0.04]');
    const labelCls = isRally
      ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/25'
      : 'bg-white/[0.06] text-text-muted ring-1 ring-white/10';
    const durationSec = (a.end - a.start).toFixed(1);

    return `
      <div class="ann-item flex items-center gap-2.5 px-3 py-2.5 rounded-xl border ${rowCls} cursor-pointer transition-all duration-200 group" data-idx="${i}">
        <span class="text-[10px] font-heading text-text-muted/60 w-4 text-right select-none">${i + 1}</span>
        <span class="ann-label ${labelCls} px-2.5 py-0.5 rounded-full text-[11px] font-medium select-none">${a.label}</span>
        <div class="flex items-center gap-1.5 ml-auto">
          <input type="text" value="${formatTime(a.start)}" class="ann-start bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading focus:border-primary-light outline-none transition-colors duration-200 tabular-nums" data-idx="${i}">
          <span class="text-text-muted/40 text-[10px]">\u2192</span>
          <input type="text" value="${formatTime(a.end)}" class="ann-end bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading focus:border-primary-light outline-none transition-colors duration-200 tabular-nums" data-idx="${i}">
        </div>
        <span class="text-[10px] text-text-muted font-heading tabular-nums bg-surface-200/40 px-1.5 py-0.5 rounded">${durationSec}s</span>
        <button class="ann-preview text-primary-light hover:text-white cursor-pointer transition-colors duration-200" data-idx="${i}" title="Jump to end">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
        </button>
        <button class="ann-delete text-red-400/60 hover:text-red-400 cursor-pointer opacity-0 group-hover:opacity-100 transition-all duration-200" data-idx="${i}" title="Delete">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>`;
  }).join('');

  // Click whole row → jump to rally start and play
  el.querySelectorAll('.ann-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (e.target.closest('.ann-start, .ann-end, .ann-preview, .ann-delete')) return;
      const idx = parseInt(item.dataset.idx);
      _selectedIdx = idx;
      videoEl.currentTime = state.annotations[idx].start;
      videoEl.play();
      renderAnnotations();
    });
  });
  // Edit start time (mm:ss → seconds)
  el.querySelectorAll('.ann-start').forEach(inp => {
    inp.addEventListener('change', (e) => {
      state.annotations[parseInt(e.target.dataset.idx)].start = parseTime(e.target.value);
      renderAnnotations();
    });
  });
  // Edit end time (mm:ss → seconds)
  el.querySelectorAll('.ann-end').forEach(inp => {
    inp.addEventListener('change', (e) => {
      state.annotations[parseInt(e.target.dataset.idx)].end = parseTime(e.target.value);
      renderAnnotations();
    });
  });
  el.querySelectorAll('.ann-preview').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(e.currentTarget.dataset.idx);
      _selectedIdx = idx;
      const a = state.annotations[idx];
      videoEl.currentTime = Math.max(a.start, a.end - 5);
      videoEl.play();
      renderAnnotations();
    });
  });
  el.querySelectorAll('.ann-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(e.currentTarget.dataset.idx);
      state.annotations.splice(idx, 1);
      if (_selectedIdx === idx) _selectedIdx = -1;
      else if (_selectedIdx > idx) _selectedIdx--;
      renderAnnotations();
    });
  });
}

async function saveAnnotations() {
  if (!state.videoName) return showToast('No video loaded', 'warning');
  try {
    await api('/annotate/annotations', {
      method: 'POST',
      body: { video: state.videoName, duration: state.duration, annotations: state.annotations },
    });
    showToast('Annotations saved!', 'success');
  } catch (e) {
    showToast(`Save failed: ${e.message}`, 'error');
  }
}

function clearAll() {
  if (state.annotations.length === 0) return;
  if (!confirm('Clear all annotations?')) return;
  state.annotations = [];
  _selectedIdx = -1;
  renderAnnotations();
}

// ── Timeline ──
function resizeTimeline() {
  if (!timelineCanvas) return;
  const rect = timelineCanvas.getBoundingClientRect();
  if (rect.width === 0) return; // hidden container
  timelineCanvas.width = rect.width * devicePixelRatio;
  timelineCanvas.height = rect.height * devicePixelRatio;
}

function startTimelineLoop() {
  function draw() {
    animFrame = requestAnimationFrame(draw);
    if (!timelineCanvas || !state.duration) return;

    const ctx = timelineCanvas.getContext('2d');
    const w = timelineCanvas.width;
    const h = timelineCanvas.height;
    const dpr = devicePixelRatio;

    ctx.clearRect(0, 0, w, h);

    // Background with subtle grid
    ctx.fillStyle = 'rgba(255,255,255,0.02)';
    ctx.fillRect(0, 0, w, h);

    // Annotation segments
    for (const a of state.annotations) {
      const x1 = (a.start / state.duration) * w;
      const x2 = (a.end / state.duration) * w;
      if (a.label === 'rally') {
        const grad = ctx.createLinearGradient(x1, 0, x1, h);
        grad.addColorStop(0, 'rgba(34, 197, 94, 0.5)');
        grad.addColorStop(1, 'rgba(34, 197, 94, 0.25)');
        ctx.fillStyle = grad;
      } else {
        ctx.fillStyle = 'rgba(100, 116, 139, 0.2)';
      }
      ctx.beginPath();
      ctx.roundRect(x1, 2 * dpr, x2 - x1, h - 4 * dpr, 3 * dpr);
      ctx.fill();
    }

    // Playhead
    if (videoEl && !isNaN(videoEl.currentTime)) {
      const px = (videoEl.currentTime / state.duration) * w;
      ctx.fillStyle = '#F97316';
      ctx.shadowColor = 'rgba(249,115,22,0.6)';
      ctx.shadowBlur = 6 * dpr;
      ctx.fillRect(px - 1 * dpr, 0, 2 * dpr, h);
      ctx.shadowBlur = 0;
    }

    // Mark start
    if (_markStart != null) {
      const mx = (_markStart / state.duration) * w;
      ctx.fillStyle = '#6366F1';
      ctx.shadowColor = 'rgba(99,102,241,0.6)';
      ctx.shadowBlur = 6 * dpr;
      ctx.fillRect(mx - 1 * dpr, 0, 2 * dpr, h);
      ctx.shadowBlur = 0;
    }
  }
  draw();
}
