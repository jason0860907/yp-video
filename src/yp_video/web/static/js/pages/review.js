/**
 * Review page — Review TAD predictions and save corrected annotations.
 */
import { api, formatTime, parseTime, card, pageHeader, sectionTitle, btnSmall, showToast, emptyState, selectCls, kbdHint } from '../shared.js';

let state = { results: [], annotations: [], videoName: '', duration: 0, kindFilter: 'all' };
let videoEl = null;
let timelineCanvas = null;
let animFrame = null;
let _markStart = null;
let _selectedIdx = -1;
let _highlight = { inside: -1, prev: -1, next: -1 };

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-5">
      ${pageHeader('Review', 'Review TAD predictions and correct annotations', `
        <div class="inline-flex rounded-lg border border-border bg-surface-100 p-0.5" role="tablist" aria-label="Cut kind">
          <button type="button" data-kind="all"       class="rev-kind-tab px-3 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="true">All <span class="opacity-60 ml-1" data-count="all">0</span></button>
          <button type="button" data-kind="broadcast" class="rev-kind-tab px-3 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="false">Broadcast <span class="opacity-60 ml-1" data-count="broadcast">0</span></button>
          <button type="button" data-kind="sideline"  class="rev-kind-tab px-3 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="false">Sideline <span class="opacity-60 ml-1" data-count="sideline">0</span></button>
        </div>
        <select id="rev-filter" class="${selectCls}" title="Filter by split / quality">
          <option value="all">All files</option>
          <option value="val">Validation only</option>
          <option value="train">Training only</option>
          <option value="predict-only">Predict only (no annotation)</option>
          <option value="failing">Failing (mAP &lt; 30%)</option>
          <option value="val-failing">Val + failing</option>
        </select>
        <select id="rev-results" class="${selectCls}">
          <option value="">Select result file...</option>
        </select>
        ${btnSmall('Load', 'id="rev-load"', 'primary')}
      `)}

      <div class="flex flex-col lg:flex-row gap-5">
        <!-- Left: Video + Timeline -->
        <div class="flex-1 min-w-0 space-y-4">
          ${card(`
            <div class="space-y-4">
              <div class="relative bg-black rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-lg shadow-black/30">
                <video id="rev-player" class="w-full max-h-[45vh]" controls></video>
              </div>

              <div class="space-y-2">
                <div class="relative rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-inner" style="background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)">
                  <canvas id="rev-timeline" class="w-full h-12 cursor-pointer" title="Click to seek"></canvas>
                </div>
                <div class="flex items-center justify-between px-0.5">
                  <span id="rev-time" class="text-sm font-heading text-text-primary tabular-nums bg-surface-200/50 px-2.5 py-1 rounded-lg border border-border">00:00</span>
                  <div class="flex items-center gap-2">
                    ${btnSmall('Start [', 'id="rev-mark-start"', 'primary')}
                    ${btnSmall('End ]', 'id="rev-mark-end"', 'primary')}
                    ${btnSmall('Rally \u21b5', 'id="rev-add-rally"', 'success')}
                    ${btnSmall('Non-Rally', 'id="rev-add-nonrally"')}
                  </div>
                </div>
              </div>

              <div id="rev-mark-info" class="hidden p-3 rounded-xl bg-primary/10 border border-primary/20 flex items-center gap-2.5">
                <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse"></span>
                <span class="text-xs text-primary-light">Start marked at <strong id="rev-mark-time" class="font-heading">0:00</strong> &mdash; press <kbd class="px-1.5 py-0.5 rounded bg-surface-200 border border-border text-[10px] font-heading text-text-secondary ml-0.5 mr-0.5">]</kbd> to set end</span>
              </div>
            </div>
          `)}
        </div>

        <!-- Right: Annotations list -->
        <div class="lg:w-[420px] lg:flex-shrink-0">
          ${card(`
            <div class="space-y-4">
              ${sectionTitle(
                'Annotations <span id="rev-count" class="text-text-muted font-normal">(0)</span>',
                '',
                `${btnSmall('Save', 'id="rev-save"', 'success')}
                 ${btnSmall('Clear', 'id="rev-clear"', 'danger')}`
              )}
              <div class="h-px bg-border"></div>
              <div id="rev-list" class="space-y-1.5 max-h-[55vh] overflow-y-auto pr-1 scrollbar-thin"></div>
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

  videoEl = document.getElementById('rev-player');
  timelineCanvas = document.getElementById('rev-timeline');
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
  if (videoEl) {
    videoEl.pause();
    videoEl.removeAttribute('src');
    videoEl.load();
  }
}

function bindEvents() {
  document.getElementById('rev-load').addEventListener('click', loadFile);
  document.getElementById('rev-mark-start').addEventListener('click', markStart);
  document.getElementById('rev-mark-end').addEventListener('click', markEnd);
  document.getElementById('rev-add-rally').addEventListener('click', () => addAnnotation('rally'));
  document.getElementById('rev-add-nonrally').addEventListener('click', () => addAnnotation('non-rally'));
  document.getElementById('rev-save').addEventListener('click', saveAnnotations);
  document.getElementById('rev-clear').addEventListener('click', clearAll);

  videoEl.addEventListener('loadedmetadata', () => {
    state.duration = videoEl.duration;
    resizeTimeline();
  });
  videoEl.addEventListener('timeupdate', () => {
    document.getElementById('rev-time').textContent = formatTime(videoEl.currentTime);
    refreshPlayingHighlight();
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
    state.results = await api('/review/results');
    renderResultsDropdown();
    document.getElementById('rev-filter').addEventListener('change', renderResultsDropdown);
    document.querySelectorAll('.rev-kind-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        state.kindFilter = btn.dataset.kind;
        renderResultsDropdown();
      });
    });
  } catch (e) {
    showToast(`Failed to load results: ${e.message}`, 'error');
  }
}

function renderResultsDropdown() {
  const sel = document.getElementById('rev-results');
  const filter = document.getElementById('rev-filter')?.value || 'all';

  // Clear except the placeholder (first <option>)
  while (sel.options.length > 1) sel.remove(1);

  const annotatedNames = new Set(
    state.results.filter(r => r.source === 'annotation').map(r => r.name)
  );

  // Apply the split/quality filter once; the kind filter and tab counts both
  // run off this filtered set so the badges show only entries the dropdown
  // could currently produce.
  const passesSplit = (r) => {
    const isVal = r.subset === 'validation';
    const isTrain = r.subset === 'training';
    const isFailing = typeof r.map === 'number' && r.map < 0.3;
    const isPredictOnly = r.source === 'tad-prediction' && !annotatedNames.has(r.name);
    if (filter === 'val' && !isVal) return false;
    if (filter === 'train' && !isTrain) return false;
    if (filter === 'predict-only' && !isPredictOnly) return false;
    if (filter === 'failing' && !isFailing) return false;
    if (filter === 'val-failing' && !(isVal && isFailing)) return false;
    return true;
  };
  const counts = { all: 0, broadcast: 0, sideline: 0 };
  state.results.forEach(r => {
    if (!passesSplit(r)) return;
    counts.all++;
    if (r.kind === 'broadcast') counts.broadcast++;
    else if (r.kind === 'sideline') counts.sideline++;
  });
  document.querySelectorAll('.rev-kind-tab').forEach(btn => {
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

  let kept = 0;
  state.results.forEach(r => {
    if (!passesSplit(r)) return;
    if (state.kindFilter !== 'all' && r.kind !== state.kindFilter) return;

    const opt = document.createElement('option');
    opt.value = `${r.source}::${r.name}`;
    const srcTag = r.source === 'annotation' ? '\u2705' : '\ud83e\udd16';
    const valTag = r.subset === 'validation' ? ' [VAL]' : '';
    const kindTag = r.kind === 'sideline' ? ' [SIDE]' : '';
    // mAP describes the model's prediction vs ground truth — only meaningful
    // on the prediction entry. Annotation entries are the GT itself.
    const mapTag = (r.source === 'tad-prediction' && typeof r.map === 'number')
      ? ` (mAP=${(r.map * 100).toFixed(0)}%)` : '';
    opt.textContent = `${srcTag}${valTag}${kindTag} ${r.name}${mapTag}`;
    sel.appendChild(opt);
    kept++;
  });

  // Placeholder reflects filter state
  sel.options[0].textContent = kept ? `Select result file... (${kept})` : 'No matches';
}

async function loadFile() {
  const raw = document.getElementById('rev-results').value;
  if (!raw) return;
  const sep = raw.indexOf('::');
  const source = sep >= 0 ? raw.slice(0, sep) : '';
  const name = sep >= 0 ? raw.slice(sep + 2) : raw;

  try {
    const qs = source ? `?source=${encodeURIComponent(source)}` : '';
    const data = await api(`/review/results/${encodeURIComponent(name)}${qs}`);
    const videoPath = data.video || data.source_video || data.metadata?.video || '';
    state.videoName = videoPath;
    if (videoPath) {
      videoEl.pause();
      videoEl.removeAttribute('src');
      videoEl.load();
      videoEl.src = `/api/review/video/${encodeURIComponent(videoPath)}`;
      videoEl.load();
    }

    state.annotations = (data.results || []).map(r => ({
      start: r.start ?? r.start_time ?? r.segment?.[0] ?? 0,
      end: r.end ?? r.end_time ?? r.segment?.[1] ?? 0,
      label: r.label || 'rally',
      score: r.confidence ?? r.score ?? null,
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
  document.getElementById('rev-mark-info').classList.remove('hidden');
  document.getElementById('rev-mark-time').textContent = formatTime(_markStart);
}

function markEnd() {
  if (_markStart == null) return;
  const end = videoEl.currentTime;
  if (end <= _markStart) return showToast('End must be after start', 'warning');
  state.annotations.push({ start: _markStart, end, label: 'rally' });
  state.annotations.sort((a, b) => a.start - b.start);
  _markStart = null;
  document.getElementById('rev-mark-info').classList.add('hidden');
  renderAnnotations();
}

function addAnnotation(label) {
  if (_markStart == null) return showToast('Mark start first with [', 'warning');
  const end = videoEl.currentTime;
  if (end <= _markStart) return showToast('End must be after start', 'warning');
  state.annotations.push({ start: _markStart, end, label });
  state.annotations.sort((a, b) => a.start - b.start);
  _markStart = null;
  document.getElementById('rev-mark-info').classList.add('hidden');
  renderAnnotations();
}

// Highlight rows around the playhead so the user can find "where am I now" in
// a long list. Three states:
//   inside a rally  → orange left border on that row
//   between rallies → orange marker on prev rally + orange marker on next rally
//   before/after all → only one marker on the adjacent rally
function refreshPlayingHighlight({ scroll = true } = {}) {
  if (!videoEl) return;
  const t = videoEl.currentTime;

  let inside = -1, prev = -1, next = -1;
  for (let i = 0; i < state.annotations.length; i++) {
    const a = state.annotations[i];
    if (t >= a.start && t < a.end) { inside = i; prev = -1; next = -1; break; }
    if (a.end <= t) prev = i;
    if (a.start > t && next === -1) next = i;
  }

  if (inside === _highlight.inside && prev === _highlight.prev && next === _highlight.next) return;

  const listEl = document.getElementById('rev-list');
  if (!listEl) { _highlight = { inside, prev, next }; return; }

  const STYLE = {
    inside: 'inset 3px 0 0 #F97316',
    prev:   'inset 3px 0 0 rgba(249, 115, 22, 0.4)',
    next:   'inset 3px 0 0 rgba(249, 115, 22, 0.4)',
  };
  const applyStyle = (idx, kind) => {
    if (idx < 0) return;
    const row = listEl.querySelector(`.rev-item[data-idx="${idx}"]`);
    if (!row) return;
    if (kind) {
      row.style.boxShadow = STYLE[kind];
      row.dataset.playing = '1';
    } else {
      row.style.boxShadow = '';
      delete row.dataset.playing;
    }
  };

  applyStyle(_highlight.inside, null);
  applyStyle(_highlight.prev, null);
  applyStyle(_highlight.next, null);
  applyStyle(inside, 'inside');
  applyStyle(prev, 'prev');
  applyStyle(next, 'next');

  if (scroll) {
    const target = inside >= 0 ? inside : (next >= 0 ? next : prev);
    if (target >= 0) {
      const row = listEl.querySelector(`.rev-item[data-idx="${target}"]`);
      row?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  _highlight = { inside, prev, next };
}

function renderAnnotations() {
  const el = document.getElementById('rev-list');
  document.getElementById('rev-count').textContent = `(${state.annotations.length})`;
  // DOM was replaced; force the highlight to re-apply on the new nodes.
  _highlight = { inside: -1, prev: -1, next: -1 };

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
    const scoreColor = a.score > 0.7 ? 'text-emerald-400' : a.score > 0.4 ? 'text-amber-400' : 'text-text-muted';
    const scoreBg = a.score > 0.7 ? 'bg-emerald-500/10 ring-emerald-500/20' : a.score > 0.4 ? 'bg-amber-500/10 ring-amber-500/20' : 'bg-white/5 ring-white/10';

    return `
      <div class="rev-item flex items-center gap-2.5 px-3 py-2.5 rounded-xl border ${rowCls} cursor-pointer transition-all duration-200 group" data-idx="${i}">
        <span class="text-[10px] font-heading text-text-muted/60 w-4 text-right select-none">${i + 1}</span>
        <span class="rev-label ${labelCls} px-2.5 py-0.5 rounded-full text-[11px] font-medium select-none">${a.label}</span>
        <div class="flex items-center gap-1.5 ml-auto">
          <input type="text" value="${formatTime(a.start)}" class="rev-start bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading focus:border-primary-light outline-none transition-colors duration-200 tabular-nums" data-idx="${i}">
          <span class="text-text-muted/40 text-[10px]">\u2192</span>
          <input type="text" value="${formatTime(a.end)}" class="rev-end bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading focus:border-primary-light outline-none transition-colors duration-200 tabular-nums" data-idx="${i}">
        </div>
        <span class="text-[10px] text-text-muted font-heading tabular-nums bg-surface-200/40 px-1.5 py-0.5 rounded">${durationSec}s</span>
        ${a.score != null ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-heading font-medium tabular-nums ring-1 ${scoreColor} ${scoreBg}">${(a.score * 100).toFixed(0)}%</span>` : ''}
        <button class="rev-preview text-primary-light hover:text-white cursor-pointer transition-colors duration-200" data-idx="${i}" title="Jump to end">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
        </button>
        <button class="rev-delete text-red-400/60 hover:text-red-400 cursor-pointer opacity-0 group-hover:opacity-100 transition-all duration-200" data-idx="${i}" title="Delete">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>`;
  }).join('');

  // Click whole row -> jump to rally start and play
  el.querySelectorAll('.rev-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (e.target.closest('.rev-start, .rev-end, .rev-preview, .rev-delete')) return;
      const idx = parseInt(item.dataset.idx);
      _selectedIdx = idx;
      videoEl.currentTime = state.annotations[idx].start;
      videoEl.play();
      renderAnnotations();
    });
  });
  // Edit start time
  el.querySelectorAll('.rev-start').forEach(inp => {
    inp.addEventListener('change', (e) => {
      state.annotations[parseInt(e.target.dataset.idx)].start = parseTime(e.target.value);
      renderAnnotations();
    });
  });
  // Edit end time
  el.querySelectorAll('.rev-end').forEach(inp => {
    inp.addEventListener('change', (e) => {
      state.annotations[parseInt(e.target.dataset.idx)].end = parseTime(e.target.value);
      renderAnnotations();
    });
  });
  el.querySelectorAll('.rev-preview').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(e.currentTarget.dataset.idx);
      _selectedIdx = idx;
      const a = state.annotations[idx];
      videoEl.currentTime = Math.max(a.start, a.end - 5);
      videoEl.play();
      renderAnnotations();
    });
  });
  el.querySelectorAll('.rev-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(e.currentTarget.dataset.idx);
      state.annotations.splice(idx, 1);
      if (_selectedIdx === idx) _selectedIdx = -1;
      else if (_selectedIdx > idx) _selectedIdx--;
      renderAnnotations();
    });
  });

  refreshPlayingHighlight({ scroll: false });
}

async function saveAnnotations() {
  if (!state.videoName) return showToast('No video loaded', 'warning');
  try {
    await api('/review/annotations', {
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

// -- Timeline --
function resizeTimeline() {
  if (!timelineCanvas) return;
  const rect = timelineCanvas.getBoundingClientRect();
  if (rect.width === 0) return;
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
