/**
 * Action Label page - rally-scoped frame/point action annotation.
 */
import {
  api, API, SSEClient, pageHeader, card, sectionTitle, btnSmall,
  showToast, showConfirm, emptyState, escapeHtml, selectCls, inputCls, kbdHint, renderJobProgress,
} from '../shared.js';

const COLORS = {
  serve: '#38BDF8',
  receive: '#22C55E',
  set: '#A78BFA',
  spike: '#F97316',
  block: '#EF4444',
  score: '#FBBF24',
};
const MAX_VIDEO_OPTIONS = 80;

let videos = [];
let labels = ['serve', 'receive', 'set', 'spike', 'block', 'score'];
let selectedLabel = 'serve';
let selectedVideo = '';
let kindFilter = 'all';
let progressFilter = 'all';
let videoSearch = '';
let videoDropdownOpen = false;
let activeVideoOption = 0;
let pointMode = false;
let spotInfo = { available: false, checkpoints: [], default_checkpoint: '' };
let spotJob = null;
let spotClient = null;
let state = {
  video: '',
  duration: 0,
  fps: 30,
  numFrames: 0,
  rallies: [],
  events: [],
  dirty: false,
};
let videoEl = null;
let overlayEl = null;
let selectedIdx = -1;
let selectedRallyId = 'all';
let tickTimer = null;
let lastOverlayFrame = -1;
let dragPoint = null;
let suppressOverlayClick = false;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-5">
      ${pageHeader('Action Label', 'Add action points on top of rally annotations', `
        <select id="act-kind" class="${selectCls}" title="Filter by cut kind">
          <option value="all">All kinds</option>
          <option value="broadcast">Broadcast only</option>
          <option value="sideline">Sideline only</option>
        </select>
        <select id="act-progress" class="${selectCls}" title="Filter by action label status">
          <option value="all">All</option>
          <option value="unlabeled">Unlabeled</option>
          <option value="labeled">Labeled</option>
        </select>
        <div class="relative w-[18rem]" id="act-video-combo-wrap">
          <input id="act-video-combo" class="${selectCls} w-full pr-8 truncate" role="combobox" aria-expanded="false" aria-controls="act-video-options" placeholder="Select cut video..." autocomplete="off">
          <button id="act-video-clear" type="button" class="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary text-xs px-1" title="Clear video search">x</button>
          <div id="act-video-options" class="hidden absolute z-40 left-0 right-0 top-full mt-1 max-h-72 overflow-auto rounded-xl border border-border bg-surface-100 shadow-2xl shadow-black/30 p-1"></div>
        </div>
        ${btnSmall('Load', 'id="act-load" title="Load selected video"', 'primary')}
        ${btnSmall('Export JSON', 'id="act-export" title="Export saved action annotations"')}
      `)}

      <div class="flex flex-col lg:flex-row gap-5">
        <div class="flex-1 min-w-0 space-y-4">
          ${card(`
            <div class="space-y-4">
              <div id="act-video-wrap" class="relative bg-black rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-lg shadow-black/30">
                <video id="act-player" class="w-full max-h-[45vh]" playsinline preload="metadata"></video>
                <div id="act-overlay" class="absolute inset-0 pointer-events-none"></div>
              </div>

              <div class="space-y-2">
                <div id="act-timeline" class="relative h-12 rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-inner cursor-pointer" title="Click to seek" style="background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)">
                  <div id="act-timeline-playhead" class="absolute top-0 bottom-0 w-px bg-accent z-20 pointer-events-none" style="left:0%"></div>
                  <div id="act-timeline-markers" class="absolute inset-0"></div>
                </div>
                <div class="flex items-center justify-between px-0.5 gap-3">
                  <span id="act-time" class="text-sm font-heading text-text-primary tabular-nums bg-surface-200/50 px-2.5 py-1 rounded-lg border border-border">00:00</span>
                  <div class="flex flex-wrap items-center justify-end gap-2">
                    ${btnSmall('<svg id="act-play-icon" class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>', 'id="act-play-toggle" title="Play / pause"')}
                    ${btnSmall('<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/></svg>', 'id="act-prev-frame" title="Previous frame"')}
                    ${btnSmall('<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>', 'id="act-next-frame" title="Next frame"')}
                    ${btnSmall('Review mode', 'id="act-point-mode" title="Review mode: clicking the video will not add points"')}
                    ${btnSmall('Add center', 'id="act-add-center"', 'primary')}
                  </div>
                </div>
              </div>

              <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-text-muted font-heading tabular-nums">
                <span id="act-video-count">Loading videos...</span>
                <span id="act-export-summary"></span>
                <span id="act-meta"></span>
              </div>
            </div>
          `)}
        </div>

        <div class="lg:w-[420px] lg:flex-shrink-0">
          ${card(`
            <div class="space-y-4">
              ${sectionTitle(
                'Action Labels',
                '',
                `${btnSmall('Save', 'id="act-save"', 'success')} ${btnSmall('Clear', 'id="act-clear"', 'danger')}`,
              )}
              <div class="flex items-center justify-between gap-2 text-[11px] text-text-muted">
                <span id="act-rally-now" class="font-heading tabular-nums">No rally selected</span>
                <span id="act-dirty"></span>
              </div>
              <div id="act-labels" class="grid grid-cols-3 gap-2"></div>
              <div class="h-px bg-border"></div>
              <div class="space-y-2">
                <div class="flex items-center gap-2">
                  <select id="act-rally-scope" class="${selectCls} flex-1 text-xs" title="Choose rally scope">
                    <option value="all">All rallies</option>
                  </select>
                  ${btnSmall('Prev', 'id="act-rally-prev" title="Previous rally"')}
                  ${btnSmall('Next', 'id="act-rally-next" title="Next rally"')}
                </div>
                <div id="act-rally-summary" class="rounded-lg border border-border bg-surface-100/45 px-3 py-2 text-xs text-text-secondary font-heading tabular-nums">No video loaded</div>
              </div>
              <div class="h-px bg-border"></div>
              ${sectionTitle(
                'Events <span id="act-count" class="text-text-muted font-normal">(0)</span>',
                '',
                `${btnSmall('Sort', 'id="act-sort"')}`,
              )}
              <div id="act-events" class="space-y-1.5 max-h-[55vh] overflow-auto pr-1 scrollbar-thin"></div>
            </div>
          `)}
        </div>
      </div>

      ${kbdHint([['1-6', 'label'], ['Space', 'play/pause'], ['Left/Right', '1 frame'], ['Shift+Left/Right', '10 frames'], ['Enter', 'add center'], ['P', 'point mode'], ['Del', 'remove']])}

      <details id="act-spot-tools" class="rounded-xl border border-border bg-surface-100/40 px-3 py-2">
        <summary class="cursor-pointer list-none text-xs font-heading text-text-secondary hover:text-text-primary select-none">
          <span class="inline-flex flex-wrap items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-primary-light/80"></span>
            SPOT pre-label
            <span class="text-text-muted font-normal">checkpoint, score threshold, GPU handoff</span>
          </span>
        </summary>
        <div class="mt-3 grid grid-cols-1 md:grid-cols-[minmax(16rem,1fr)_7rem_7rem_auto_auto] gap-2.5 items-end">
          <label class="space-y-1.5 min-w-0">
            <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Checkpoint</span>
            <select id="act-spot-checkpoint" class="${selectCls} w-full">
              <option value="">Loading SPOT...</option>
            </select>
          </label>
          <label class="space-y-1.5 min-w-0">
            <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Min score</span>
            <input id="act-spot-score" type="number" min="0" max="1" step="0.05" value="0.15" class="${inputCls} w-full">
          </label>
          <label class="space-y-1.5 min-w-0">
            <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Batch</span>
            <input id="act-spot-batch" type="number" min="1" max="32" step="1" value="8" class="${inputCls} w-full">
          </label>
          <label class="flex items-center gap-2 text-xs text-text-secondary pb-3 cursor-pointer">
            <input id="act-spot-stop-vllm" type="checkbox" class="accent-primary w-3.5 h-3.5">
            Stop vLLM
          </label>
          ${btnSmall('Run SPOT', 'id="act-spot-run" title="Run ~/yp-spot model and overwrite saved action labels"', 'primary')}
        </div>
        <div id="act-spot-progress" class="hidden mt-4 pt-4 border-t border-border"></div>
      </details>
    </div>`;

  bindEvents();
  loadInitial();
  activate();
  window.removeEventListener('beforeunload', onBeforeUnload);
  window.addEventListener('beforeunload', onBeforeUnload);
}

export function activate() {
  document.addEventListener('keydown', onKeydown);
  tickTimer = setInterval(refreshPlayhead, 100);
  refreshPlayhead();
}

export function deactivate() {
  document.removeEventListener('keydown', onKeydown);
  if (tickTimer) clearInterval(tickTimer);
  tickTimer = null;
  spotClient?.stop();
  spotClient = null;
  videoEl?.pause();
}

function bindEvents() {
  videoEl = document.getElementById('act-player');
  overlayEl = document.getElementById('act-overlay');

  const combo = document.getElementById('act-video-combo');
  combo.addEventListener('focus', () => {
    videoDropdownOpen = true;
    renderVideoOptions();
  });
  combo.addEventListener('click', () => {
    videoDropdownOpen = true;
    renderVideoOptions();
  });
  combo.addEventListener('input', (e) => {
    videoSearch = e.target.value;
    selectedVideo = '';
    activeVideoOption = 0;
    videoDropdownOpen = true;
    renderVideoOptions();
  });
  combo.addEventListener('keydown', (e) => {
    const matches = filteredVideos();
    const maxActive = Math.max(0, Math.min(matches.length, MAX_VIDEO_OPTIONS) - 1);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      videoDropdownOpen = true;
      activeVideoOption = Math.min(activeVideoOption + 1, maxActive);
      renderVideoOptions();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      videoDropdownOpen = true;
      activeVideoOption = Math.max(0, activeVideoOption - 1);
      renderVideoOptions();
    } else if (e.key === 'Escape') {
      closeVideoDropdown();
    } else if (e.key === 'Tab') {
      closeVideoDropdown();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (!selectedVideo && matches[activeVideoOption]) {
        selectVideo(matches[activeVideoOption].name);
        return;
      }
      loadSelectedVideo();
    }
  });
  document.getElementById('act-video-clear').addEventListener('click', () => {
    videoSearch = '';
    selectedVideo = '';
    activeVideoOption = 0;
    combo.value = '';
    videoDropdownOpen = true;
    combo.focus();
    renderVideoOptions();
  });
  document.getElementById('act-video-options').addEventListener('mousedown', (e) => {
    e.preventDefault();
    const opt = e.target.closest('[data-video-name]');
    if (!opt) return;
    selectVideo(opt.dataset.videoName);
  });
  document.removeEventListener('mousedown', onDocumentMouseDown);
  document.addEventListener('mousedown', onDocumentMouseDown);
  document.getElementById('act-rally-scope').addEventListener('change', (e) => selectRally(e.target.value));
  document.getElementById('act-rally-prev').addEventListener('click', () => stepRally(-1));
  document.getElementById('act-rally-next').addEventListener('click', () => stepRally(1));
  document.getElementById('act-timeline').addEventListener('click', onTimelineClick);
  document.getElementById('act-kind').addEventListener('change', (e) => {
    kindFilter = e.target.value;
    activeVideoOption = 0;
    renderVideoOptions();
  });
  document.getElementById('act-progress').addEventListener('change', (e) => {
    progressFilter = e.target.value;
    activeVideoOption = 0;
    renderVideoOptions();
  });
  document.getElementById('act-load').addEventListener('click', loadSelectedVideo);
  document.getElementById('act-save').addEventListener('click', save);
  document.getElementById('act-export').addEventListener('click', exportDataset);
  document.getElementById('act-spot-run').addEventListener('click', startSpotPrelabel);
  document.getElementById('act-point-mode').addEventListener('click', togglePointMode);
  document.getElementById('act-add-center').addEventListener('click', () => addEvent(0.5, 0.5));
  document.getElementById('act-play-toggle').addEventListener('click', togglePlayback);
  document.getElementById('act-prev-frame').addEventListener('click', () => stepFrame(-1));
  document.getElementById('act-next-frame').addEventListener('click', () => stepFrame(1));
  document.getElementById('act-sort').addEventListener('click', () => {
    if (!state.video || state.events.length < 2) return;
    sortEvents();
    markDirty();
    renderEvents();
  });
  document.getElementById('act-clear').addEventListener('click', clearEvents);

  videoEl.addEventListener('loadedmetadata', () => {
    refreshPlayhead();
    updatePlaybackButton();
  });
  videoEl.addEventListener('timeupdate', refreshPlayhead);
  videoEl.addEventListener('play', updatePlaybackButton);
  videoEl.addEventListener('pause', updatePlaybackButton);
  videoEl.addEventListener('ended', updatePlaybackButton);
  videoEl.addEventListener('click', (e) => {
    if (!state.video || !pointMode) return;
    const rect = videoEl.getBoundingClientRect();
    const x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const y = clamp((e.clientY - rect.top) / rect.height, 0, 1);
    addEvent(x, y);
  });

  document.getElementById('act-events').addEventListener('click', onEventClick);
  document.getElementById('act-events').addEventListener('change', onEventChange);
  document.getElementById('act-events').addEventListener('input', onEventChange);
  updatePointModeButton();
  renderRallies();
  renderEvents();
}

async function loadInitial() {
  try {
    const [videoList, labelData, spotData] = await Promise.all([
      api(API.actionAnnotate.videos),
      api(API.actionAnnotate.labels),
      api(API.actionAnnotate.spot).catch(e => ({ available: false, error: e.message, checkpoints: [] })),
    ]);
    videos = videoList;
    labels = labelData.labels || labels;
    spotInfo = spotData;
    selectedLabel = labels[0] || selectedLabel;
    renderLabels();
    renderVideoOptions();
    renderSpotControls();
  } catch (e) {
    showToast(`Failed to load action annotator: ${e.message}`, 'error');
  }
}

function renderVideoOptions() {
  const combo = document.getElementById('act-video-combo');
  const list = document.getElementById('act-video-options');
  const matches = filteredVideos();
  if (activeVideoOption >= Math.min(matches.length, MAX_VIDEO_OPTIONS)) {
    activeVideoOption = Math.max(0, Math.min(matches.length, MAX_VIDEO_OPTIONS) - 1);
  }

  if (combo && document.activeElement !== combo && selectedVideo && combo.value !== selectedVideo) {
    combo.value = selectedVideo;
  }
  if (combo) combo.setAttribute('aria-expanded', String(videoDropdownOpen));

  if (list) {
    list.classList.toggle('hidden', !videoDropdownOpen);
    if (!matches.length) {
      list.innerHTML = '<div class="px-3 py-2 text-xs text-text-muted">No videos match</div>';
    } else {
      const visible = matches.slice(0, MAX_VIDEO_OPTIONS);
      list.innerHTML = visible.map((v, idx) => {
        const active = idx === activeVideoOption;
        const selected = v.name === selectedVideo;
        const status = v.has_action_annotation ? `${v.event_count} events` : 'unlabeled';
        const kind = v.kind === 'sideline' ? 'SIDE' : 'CAST';
        return `<button type="button" data-video-name="${escapeHtml(v.name)}"
          class="w-full text-left px-3 py-2 rounded-lg text-xs transition-colors ${active || selected ? 'bg-primary/10 text-text-primary' : 'text-text-secondary hover:bg-white/[0.06] hover:text-text-primary'}">
          <span class="flex items-center gap-2 min-w-0">
            <span class="shrink-0 w-10 text-[10px] font-heading text-text-muted">${kind}</span>
            <span class="truncate flex-1">${escapeHtml(v.name)}</span>
            <span class="shrink-0 text-[10px] font-heading text-text-muted">${escapeHtml(status)}</span>
          </span>
        </button>`;
      }).join('') + (matches.length > MAX_VIDEO_OPTIONS
        ? `<div class="px-3 py-2 text-[11px] text-text-muted">Showing ${MAX_VIDEO_OPTIONS} of ${matches.length}. Keep typing to narrow results.</div>`
        : '');
    }
  }

  const countEl = document.getElementById('act-video-count');
  if (countEl) countEl.textContent = `${matches.length} shown / ${videos.length} total`;
  renderExportSummary();
}

function filteredVideos() {
  const needle = videoSearch.trim().toLowerCase();
  return videos.filter(v => {
    if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
    if (progressFilter === 'unlabeled' && v.has_action_annotation) return false;
    if (progressFilter === 'labeled' && !v.has_action_annotation) return false;
    return !needle || v.name.toLowerCase().includes(needle);
  });
}

function exactVideoMatch(value) {
  const needle = value.trim().toLowerCase();
  if (!needle) return '';
  const exact = videos.find(v => v.name.toLowerCase() === needle);
  if (exact) return exact.name;
  const matches = filteredVideos();
  return matches.length === 1 ? matches[0].name : '';
}

function selectVideo(name) {
  selectedVideo = name;
  videoSearch = name;
  const combo = document.getElementById('act-video-combo');
  if (combo) combo.value = name;
  closeVideoDropdown();
  renderVideoOptions();
}

function closeVideoDropdown() {
  videoDropdownOpen = false;
  const combo = document.getElementById('act-video-combo');
  if (combo) combo.setAttribute('aria-expanded', 'false');
  document.getElementById('act-video-options')?.classList.add('hidden');
}

function onDocumentMouseDown(e) {
  if (e.target.closest('#act-video-combo-wrap')) return;
  closeVideoDropdown();
}

function renderExportSummary() {
  const stats = datasetStats();
  const summary = document.getElementById('act-export-summary');
  const btn = document.getElementById('act-export');
  if (summary) {
    summary.textContent = `Saved: ${stats.videos} videos / ${stats.events} events`;
  }
  if (btn) {
    btn.disabled = stats.events === 0;
    btn.title = stats.events === 0
      ? 'No saved action annotations to export'
      : 'Export saved action annotations';
  }
}

function datasetStats() {
  return videos.reduce((acc, v) => {
    if (v.has_action_annotation) acc.videos += 1;
    acc.events += Math.max(0, Number(v.event_count) || 0);
    return acc;
  }, { videos: 0, events: 0 });
}

function exportDataset() {
  const stats = datasetStats();
  if (stats.events === 0) {
    showToast('No saved action annotations to export yet', 'warning');
    return;
  }
  window.location.href = `/api${API.actionAnnotate.export}`;
}

function renderSpotControls() {
  const sel = document.getElementById('act-spot-checkpoint');
  const btn = document.getElementById('act-spot-run');
  if (!sel || !btn) return;

  const checkpoints = spotInfo.checkpoints || [];
  sel.innerHTML = '';
  if (!spotInfo.available) {
    sel.innerHTML = `<option value="">SPOT unavailable: ${escapeHtml(spotInfo.error || '~/yp-spot not ready')}</option>`;
    btn.disabled = true;
    return;
  }
  if (!checkpoints.length) {
    sel.innerHTML = '<option value="">No SPOT checkpoints found</option>';
    btn.disabled = true;
    return;
  }
  for (const cp of checkpoints) {
    const opt = document.createElement('option');
    opt.value = cp.path;
    const details = [];
    if (cp.is_best) {
      details.push(`best${cp.epoch >= 0 ? ` e${cp.epoch}` : ''}`);
    }
    const bestValue = Number(cp.best_value);
    if (cp.best_metric && Number.isFinite(bestValue)) {
      const value = cp.best_metric === 'val_mAP'
        ? `${(bestValue * 100).toFixed(2)}%`
        : bestValue.toFixed(4);
      details.push(`${cp.best_metric} ${value}`);
    }
    details.push(`${cp.size_mb.toFixed(1)} MB`);
    opt.textContent = `${cp.name} (${details.join(', ')})`;
    sel.appendChild(opt);
  }
  sel.value = spotInfo.default_checkpoint || checkpoints[0].path;
  btn.disabled = false;
}

async function startSpotPrelabel() {
  const name = selectedVideo || exactVideoMatch(videoSearch) || filteredVideos()[activeVideoOption]?.name || '';
  if (!name) return showToast('Select a video first', 'warning');
  if (!spotInfo.available) return showToast('SPOT is not available at ~/yp-spot', 'error');

  if (state.dirty) {
    const ok = await showConfirm({
      title: 'Discard unsaved changes?',
      body: 'SPOT pre-label will replace the saved action labels for the selected video.',
      confirmText: 'Discard & Run',
      variant: 'danger',
    });
    if (!ok) return;
    state.dirty = false;
    updateDirtyUi();
  }

  const existing = videos.find(v => v.name === name)?.has_action_annotation;
  if (existing) {
    const ok = await showConfirm({
      title: 'Overwrite saved action labels?',
      body: `${name}\n\nThe SPOT pre-label result will replace the saved action annotation file.`,
      confirmText: 'Overwrite',
      variant: 'warning',
    });
    if (!ok) return;
  }

  let stopVllm = document.getElementById('act-spot-stop-vllm').checked;
  const vllmStatus = await api(API.system.vllmStatus).catch(() => null);
  if (vllmStatus?.status === 'running' && !stopVllm) {
    const ok = await showConfirm({
      title: 'Stop vLLM for SPOT?',
      body:
        'SPOT uses the GPU and may fail if vLLM is holding VRAM.\n\n' +
        'vLLM will be automatically restarted after the job finishes.',
      confirmText: 'Stop & Run',
      cancelText: 'Cancel',
      variant: 'warning',
    });
    if (!ok) return;
    stopVllm = true;
  }

  const btn = document.getElementById('act-spot-run');
  btn.disabled = true;
  document.getElementById('act-spot-tools')?.setAttribute('open', '');
  spotClient?.stop();
  spotClient = null;

  try {
    const job = await api(API.actionAnnotate.prelabel, {
      method: 'POST',
      body: {
        video: name,
        checkpoint: document.getElementById('act-spot-checkpoint').value,
        batch_size: Number(document.getElementById('act-spot-batch').value) || 8,
        num_workers: 4,
        clip_len: 64,
        min_score: Number(document.getElementById('act-spot-score').value) || 0.15,
        overwrite: true,
        stop_vllm: stopVllm,
      },
    });
    spotJob = job;
    renderSpotProgress();

    spotClient = new SSEClient(API.jobs.eventsSSE(job.id), {
      onMessage: async (data) => {
        if (!data.status) return;
        spotJob = data;
        renderSpotProgress();
        if (['completed', 'failed', 'cancelled'].includes(data.status)) {
          spotClient?.stop();
          spotClient = null;
          btn.disabled = false;
          if (data.status === 'completed') {
            showToast(data.message || 'SPOT pre-label complete', 'success');
            videos = await api(API.actionAnnotate.videos);
            renderVideoOptions();
            selectVideo(name);
            state.dirty = false;
            await loadSelectedVideo();
          } else if (data.status === 'failed') {
            showToast(`SPOT failed: ${data.error || 'Unknown error'}`, 'error');
          } else {
            showToast('SPOT pre-label cancelled', 'warning');
          }
        }
      },
      onError: () => { btn.disabled = false; },
    }).start();
  } catch (e) {
    showToast(`Failed to start SPOT: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

function renderSpotProgress() {
  const el = document.getElementById('act-spot-progress');
  if (!el) return;
  el.classList.toggle('hidden', !spotJob);
  if (!spotJob) {
    el.innerHTML = '';
    return;
  }
  el.innerHTML = renderJobProgress(spotJob, { showLogs: true, truncateMsg: false });
}

function renderLabels() {
  const el = document.getElementById('act-labels');
  el.innerHTML = labels.map((label, i) => {
    const active = label === selectedLabel;
    const color = COLORS[label] || '#818CF8';
    return `<button type="button" data-label="${escapeHtml(label)}"
      class="px-3 py-2 rounded-lg border text-xs font-heading font-semibold transition-colors duration-150 ${active ? 'text-white' : 'text-text-secondary hover:text-text-primary'}"
      style="border-color:${active ? color : 'rgba(255,255,255,0.08)'}; background:${active ? `${color}33` : 'rgba(255,255,255,0.035)'}">
      <span class="opacity-60">${i + 1}</span> ${escapeHtml(label)}
    </button>`;
  }).join('');
  el.querySelectorAll('[data-label]').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedLabel = btn.dataset.label;
      renderLabels();
    });
  });
}

function normalizeSelectedRally() {
  if (selectedRallyId === 'all') return;
  if (!state.rallies.some(r => r.id === selectedRallyId)) {
    selectedRallyId = state.rallies[0]?.id || 'all';
  }
}

function currentRally() {
  normalizeSelectedRally();
  if (selectedRallyId === 'all') return null;
  return state.rallies.find(r => r.id === selectedRallyId) || null;
}

function selectedRallyIndex() {
  normalizeSelectedRally();
  return state.rallies.findIndex(r => r.id === selectedRallyId);
}

function eventVisibleInScope(event) {
  if (!event) return false;
  return selectedRallyId === 'all' || event.rally_id === selectedRallyId;
}

function visibleEventEntries() {
  normalizeSelectedRally();
  return state.events
    .map((event, idx) => ({ event, idx }))
    .filter(({ event }) => eventVisibleInScope(event));
}

function rallyEventCount(rallyId) {
  return state.events.filter(e => e.rally_id === rallyId).length;
}

function renderRallies() {
  const scope = document.getElementById('act-rally-scope');
  const summary = document.getElementById('act-rally-summary');
  const prev = document.getElementById('act-rally-prev');
  const next = document.getElementById('act-rally-next');
  if (!scope || !summary) return;

  normalizeSelectedRally();
  const hasVideo = Boolean(state.video);
  const hasRallies = state.rallies.length > 0;
  const outsideCount = state.events.filter(e => !e.rally_id).length;
  const selected = currentRally();
  const selectedIdxForUi = selectedRallyIndex();

  const options = [
    `<option value="all">All rallies (${state.events.length})</option>`,
    ...state.rallies.map((rally, idx) => (
      `<option value="${escapeHtml(rally.id)}">R${idx + 1} · ${formatSeconds(rally.start)}-${formatSeconds(rally.end)} · ${rallyEventCount(rally.id)}</option>`
    )),
  ].join('');
  if (scope.innerHTML !== options) scope.innerHTML = options;
  scope.value = selectedRallyId;
  scope.disabled = !hasVideo;

  if (!hasVideo) {
    summary.textContent = 'No video loaded';
  } else if (!hasRallies) {
    summary.textContent = `${state.events.length} event(s) / no rally annotation`;
  } else {
    if (selected) {
      summary.innerHTML = `
        <span class="text-emerald-300">R${selectedIdxForUi + 1}</span>
        <span class="text-text-muted mx-1">${formatSeconds(selected.start)}-${formatSeconds(selected.end)}</span>
        <span>${rallyEventCount(selected.id)} event(s)</span>`;
    } else {
      summary.innerHTML = `
        <span class="text-text-primary">All rallies</span>
        <span class="text-text-muted mx-1">${state.events.length} event(s)</span>
        <span class="${outsideCount ? 'text-amber-300' : 'text-text-muted'}">${outsideCount} outside</span>`;
    }
  }

  if (prev) prev.disabled = !hasRallies;
  if (next) next.disabled = !hasRallies;
  updateRallyNow();
}

function selectRally(rallyId, { seek = true } = {}) {
  selectedRallyId = rallyId === 'all' ? 'all' : rallyId;
  normalizeSelectedRally();
  if (selectedIdx >= 0 && (!state.events[selectedIdx] || !eventVisibleInScope(state.events[selectedIdx]))) {
    selectedIdx = -1;
  }

  const rally = currentRally();
  if (seek && rally) {
    seekFrame(Math.round(rally.start * state.fps));
  }
  renderEvents();
}

function stepRally(delta) {
  if (!state.rallies.length) return;
  const idx = selectedRallyIndex();
  const nextIdx = idx < 0
    ? (delta > 0 ? 0 : state.rallies.length - 1)
    : clamp(idx + delta, 0, state.rallies.length - 1);
  selectRally(state.rallies[nextIdx].id);
}

function updateRallyNow(frame = currentFrame()) {
  const el = document.getElementById('act-rally-now');
  if (!el) return;
  if (!state.video) {
    el.textContent = 'No video loaded';
    return;
  }

  const selected = currentRally();
  const atFrame = findRallyForFrame(frame);
  const atIdx = atFrame ? rallyIndex(atFrame.id) : -1;
  const frameText = atIdx >= 0 ? `now R${atIdx + 1}` : 'now outside';
  if (selected) {
    const idx = selectedRallyIndex();
    el.textContent = `Scope R${idx + 1} · ${formatSeconds(selected.start)}-${formatSeconds(selected.end)} · ${frameText}`;
  } else {
    el.textContent = `Scope all rallies · ${frameText}`;
  }
}

async function loadSelectedVideo() {
  const name = selectedVideo || exactVideoMatch(videoSearch) || filteredVideos()[activeVideoOption]?.name || '';
  if (!name) return;
  if (state.dirty && name !== state.video) {
    const ok = await confirmDiscardChanges();
    if (!ok) {
      if (state.video) selectVideo(state.video);
      return;
    }
  }
  selectVideo(name);
  const btn = document.getElementById('act-load');
  btn.disabled = true;
  btn.textContent = 'Loading...';
  try {
    const data = await api(API.actionAnnotate.annotation(name));
    const rallies = normalizeRallies(data.rallies || []);
    state = {
      video: data.source_video || data.video,
      duration: data.duration || 0,
      fps: data.fps || 30,
      numFrames: data.num_frames || 0,
      rallies,
      events: normalizeEvents(data.events || [], {
        fps: data.fps || 30,
        numFrames: data.num_frames || 0,
        rallies,
      }),
      dirty: false,
    };
    selectedVideo = name;
    videoSearch = name;
    selectedIdx = -1;
    selectedRallyId = rallies[0]?.id || 'all';
    lastOverlayFrame = -1;
    videoEl.pause();
    videoEl.src = `/api${API.actionAnnotate.video(state.video)}`;
    videoEl.load();
    updatePlaybackButton();
    renderEvents();
    refreshPlayhead();
    showToast(`Loaded ${state.events.length} event(s)`, 'success');
  } catch (e) {
    showToast(`Load failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Load';
  }
}

async function confirmDiscardChanges() {
  return showConfirm({
    title: 'Discard unsaved changes?',
    body: 'The current action labels have not been saved.',
    confirmText: 'Discard',
    variant: 'danger',
  });
}

function onBeforeUnload(e) {
  if (!state.dirty) return;
  e.preventDefault();
  e.returnValue = '';
}

function markDirty() {
  state.dirty = true;
  lastOverlayFrame = -1;
  updateDirtyUi();
}

function updateDirtyUi() {
  const dirty = document.getElementById('act-dirty');
  const saveBtn = document.getElementById('act-save');
  if (dirty) dirty.textContent = state.dirty ? 'Unsaved changes' : '';
  if (saveBtn) saveBtn.textContent = state.dirty ? 'Save *' : 'Save';
}

function normalizeRallies(rallies) {
  return (rallies || []).map((r, idx) => ({
    id: r.id || `rally_${idx + 1}`,
    start: Number(r.start) || 0,
    end: Number(r.end) || 0,
    label: r.label || 'rally',
  })).sort((a, b) => a.start - b.start || a.end - b.end || a.id.localeCompare(b.id));
}

function normalizeEvents(events, context = state) {
  return events.map(e => ({
    id: e.id || makeClientId('act'),
    rally_id: e.rally_id || null,
    frame: Math.max(0, Math.round(Number(e.frame) || 0)),
    time: Number(e.time) || null,
    relative_frame: Number.isInteger(e.relative_frame) ? e.relative_frame : null,
    label: labels.includes(e.label) ? e.label : labels[0],
    xy: [
      clamp(Number(e.xy?.[0] ?? e.x ?? 0.5), 0, 1),
      clamp(Number(e.xy?.[1] ?? e.y ?? 0.5), 0, 1),
    ],
  })).map(e => withRallyFields(e, context))
    .sort((a, b) => a.frame - b.frame || a.label.localeCompare(b.label) || a.id.localeCompare(b.id));
}

function addEvent(x, y) {
  if (!state.video) return showToast('Load a video first', 'warning');
  const frame = frameForNewEvent();
  if (frame !== currentFrame()) seekFrame(frame);
  const event = withRallyFields({
    id: makeClientId('act'),
    frame,
    label: selectedLabel,
    xy: [round4(x), round4(y)],
  });
  state.events.push(event);
  sortEvents();
  selectedIdx = state.events.indexOf(event);
  markDirty();
  renderEvents();
}

function frameForNewEvent() {
  const frame = currentFrame();
  const rally = currentRally();
  if (!rally) return frame;

  const startFrame = Math.max(0, Math.round(rally.start * state.fps));
  const endFrame = Math.max(startFrame, Math.ceil(rally.end * state.fps) - 1);
  return clamp(frame, startFrame, Math.min(endFrame, Math.max(0, state.numFrames - 1)));
}

function sortEvents() {
  const selected = selectedIdx >= 0 ? state.events[selectedIdx] : null;
  state.events.sort((a, b) => a.frame - b.frame || a.label.localeCompare(b.label) || a.id.localeCompare(b.id));
  if (selected) selectedIdx = state.events.indexOf(selected);
}

async function clearEvents() {
  if (!state.events.length) return;
  const ok = await showConfirm({
    title: 'Clear action events?',
    body: `This will remove ${state.events.length} event(s) from this view. The file on disk is not touched until Save.`,
    confirmText: 'Clear',
    variant: 'danger',
  });
  if (!ok) return;
  state.events = [];
  selectedIdx = -1;
  markDirty();
  renderEvents();
}

async function save() {
  if (!state.video) return showToast('No video loaded', 'warning');
  const btn = document.getElementById('act-save');
  btn.disabled = true;
  btn.textContent = 'Saving...';
  try {
    await api(API.actionAnnotate.annotations, {
      method: 'POST',
      body: {
        video: state.video,
        fps: state.fps,
        num_frames: state.numFrames,
        events: state.events,
      },
    });
    state.dirty = false;
    renderEvents();
    videos = await api(API.actionAnnotate.videos);
    renderVideoOptions();
    showToast('Action annotations saved', 'success');
  } catch (e) {
    showToast(`Save failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    document.getElementById('act-save').textContent = state.dirty ? 'Save *' : 'Save';
  }
}

function renderEvents() {
  renderRallies();
  if (selectedIdx >= 0 && (!state.events[selectedIdx] || !eventVisibleInScope(state.events[selectedIdx]))) {
    selectedIdx = -1;
  }
  const visibleEntries = visibleEventEntries();
  const visibleCount = visibleEntries.length;
  document.getElementById('act-count').textContent = selectedRallyId === 'all'
    ? `(${state.events.length})`
    : `(${visibleCount}/${state.events.length})`;
  updateDirtyUi();

  const el = document.getElementById('act-events');
  if (!state.events.length) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>',
      'No action events',
      '',
    );
    renderOverlay();
    renderTimeline();
    return;
  }
  if (!visibleEntries.length) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3"/></svg>',
      'No events in this rally',
      '',
    );
    renderOverlay();
    renderTimeline();
    return;
  }

  el.innerHTML = `
    <div class="grid grid-cols-[1.25rem_5.8rem_4rem_3.8rem_3.4rem_2.8rem] gap-1.5 min-w-[22rem] px-3 text-[10px] uppercase tracking-widest text-text-muted font-semibold">
      <span>#</span>
      <span>Label</span>
      <span>Frame</span>
      <span>Time</span>
      <span>Rally</span>
      <span></span>
    </div>
    ${visibleEntries.map(({ event, idx }, rowIdx) => eventRow(event, idx, rowIdx + 1)).join('')}`;
  renderOverlay();
  renderTimeline();
}

function eventRow(event, idx, rowNumber = idx + 1) {
  const selected = idx === selectedIdx;
  const color = COLORS[event.label] || '#818CF8';
  const labelOptions = labels.map(label => `<option value="${escapeHtml(label)}" ${label === event.label ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('');
  const rallyText = rallyLabel(event);
  const rallyTitle = event.rally_id
    ? `${event.rally_id}${Number.isInteger(event.relative_frame) ? ` / +${event.relative_frame}f` : ''}`
    : 'Outside rally annotations';
  return `
    <div class="act-event grid grid-cols-[1.25rem_5.8rem_4rem_3.8rem_3.4rem_2.8rem] items-center gap-1.5 min-w-[22rem] px-3 py-2 rounded-xl border cursor-pointer transition-colors duration-150 ${selected ? 'bg-primary/10 border-primary/[0.35]' : 'bg-white/[0.035] border-border hover:bg-white/[0.06]'}" data-idx="${idx}" title="${escapeHtml(event.id || '')}">
      <span class="text-right text-[10px] font-heading text-text-muted/70">${rowNumber}</span>
      <span class="flex items-center gap-1.5 min-w-0">
        <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background:${color}"></span>
        <select class="min-w-0 w-full bg-surface-100 border border-border rounded-lg px-1.5 py-1 text-xs text-text-primary" data-field="label" data-idx="${idx}">${labelOptions}</select>
      </span>
      <input class="bg-transparent border-b border-white/10 text-text-primary text-[11px] w-full text-center font-heading tabular-nums focus:border-primary-light outline-none" data-field="frame" data-idx="${idx}" value="${event.frame}">
      <span class="text-[10px] text-text-muted font-heading tabular-nums text-center">${formatSeconds(event.frame / state.fps)}</span>
      <span class="text-[10px] ${event.rally_id ? 'text-emerald-300/90 bg-emerald-500/10 border-emerald-500/20' : 'text-text-muted bg-white/[0.035] border-white/10'} border rounded-md px-1.5 py-0.5 text-center font-heading tabular-nums" title="${escapeHtml(rallyTitle)}">${escapeHtml(rallyText)}</span>
      <span class="flex items-center justify-end gap-1.5">
        <button class="text-primary-light hover:text-white cursor-pointer" data-action="jump" title="Jump to event">
          <svg class="w-4 h-4 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 12h14m0 0l-5-5m5 5l-5 5"/></svg>
        </button>
        <button class="text-red-400/60 hover:text-red-400 cursor-pointer" data-action="delete" title="Delete">
          <svg class="w-3.5 h-3.5 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </span>
    </div>`;
}

function onEventClick(e) {
  const row = e.target.closest('.act-event');
  if (!row) return;
  const idx = Number(row.dataset.idx);
  if (e.target.closest('[data-action="delete"]')) {
    state.events.splice(idx, 1);
    selectedIdx = -1;
    markDirty();
    renderEvents();
    return;
  }
  selectedIdx = idx;
  if (['INPUT', 'SELECT'].includes(e.target.tagName)) {
    renderOverlay();
    return;
  }
  jumpToEvent(idx);
}

function onEventChange(e) {
  const idx = Number(e.target.dataset.idx);
  const field = e.target.dataset.field;
  if (!Number.isInteger(idx) || !field || !state.events[idx]) return;
  const event = state.events[idx];
  if (field === 'label') event.label = e.target.value;
  else if (field === 'frame') {
    event.frame = clamp(Math.round(Number(e.target.value) || 0), 0, Math.max(0, state.numFrames - 1));
    Object.assign(event, withRallyFields(event));
  }
  markDirty();
  renderOverlay();
  if (e.type === 'change') renderEvents();
}

function renderOverlay() {
  if (!overlayEl) return;
  const frame = currentFrame();
  overlayEl.innerHTML = state.events.map((event, idx) => ({ event, idx }))
    .filter(({ event }) => eventVisibleInScope(event) && Math.abs(event.frame - frame) <= 2)
    .map(({ event, idx }) => {
    const color = COLORS[event.label] || '#818CF8';
    const active = idx === selectedIdx;
    return `<button type="button" data-idx="${idx}" title="${escapeHtml(event.label)} frame ${event.frame}"
      class="absolute w-5 h-5 -ml-2.5 -mt-2.5 rounded-full pointer-events-auto cursor-grab active:cursor-grabbing touch-none group"
      style="left:${event.xy[0] * 100}%; top:${event.xy[1] * 100}%;">
      <span class="absolute left-1/2 top-1/2 rounded-full border pointer-events-none transition-transform duration-150 -translate-x-1/2 -translate-y-1/2 ${active ? 'w-3.5 h-3.5 border-2 border-white scale-110' : 'w-2.5 h-2.5 border-white/90 group-hover:scale-125'}"
        style="background:${color}; box-shadow:0 0 0 ${active ? 2 : 1}px ${color}55"></span>
    </button>`;
  }).join('');
  overlayEl.querySelectorAll('button[data-idx]').forEach(btn => {
    btn.addEventListener('pointerdown', (e) => startPointDrag(e, btn));
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (suppressOverlayClick) return;
      jumpToEvent(Number(btn.dataset.idx));
    });
  });
}

function startPointDrag(e, btn) {
  if (!state.video) return;
  e.preventDefault();
  e.stopPropagation();

  const idx = Number(btn.dataset.idx);
  if (!Number.isInteger(idx) || !state.events[idx]) return;
  selectedIdx = idx;
  videoEl.pause();
  updatePlaybackButton();
  if (moveEventToCurrentFrame(idx)) {
    markDirty();
  }
  renderTimeline();

  dragPoint = {
    idx,
    btn,
    startX: e.clientX,
    startY: e.clientY,
    moved: false,
  };
  btn.setPointerCapture?.(e.pointerId);
  btn.classList.add('scale-125');
  document.addEventListener('pointermove', onPointDragMove);
  document.addEventListener('pointerup', onPointDragEnd, { once: true });
  document.addEventListener('pointercancel', onPointDragEnd, { once: true });
}

function moveEventToCurrentFrame(idx) {
  const event = state.events[idx];
  if (!event) return false;
  const frame = currentFrame();
  if (event.frame === frame) return false;
  Object.assign(event, withRallyFields({ ...event, frame }));
  return true;
}

function onPointDragMove(e) {
  if (!dragPoint || !state.events[dragPoint.idx]) return;
  const moved = Math.abs(e.clientX - dragPoint.startX) + Math.abs(e.clientY - dragPoint.startY) > 2;
  dragPoint.moved = dragPoint.moved || moved;

  const [x, y] = clientToVideoPoint(e.clientX, e.clientY);
  const event = state.events[dragPoint.idx];
  event.xy = [x, y];
  dragPoint.btn.style.left = `${x * 100}%`;
  dragPoint.btn.style.top = `${y * 100}%`;
  markDirty();
}

function onPointDragEnd() {
  if (!dragPoint) return;
  document.removeEventListener('pointermove', onPointDragMove);
  document.removeEventListener('pointerup', onPointDragEnd);
  document.removeEventListener('pointercancel', onPointDragEnd);

  if (dragPoint.moved) {
    suppressOverlayClick = true;
    setTimeout(() => { suppressOverlayClick = false; }, 0);
  }
  dragPoint = null;
  renderEvents();
}

function clientToVideoPoint(clientX, clientY) {
  const rect = videoEl.getBoundingClientRect();
  return [
    round4(clamp((clientX - rect.left) / rect.width, 0, 1)),
    round4(clamp((clientY - rect.top) / rect.height, 0, 1)),
  ];
}

function renderTimeline() {
  const markers = document.getElementById('act-timeline-markers');
  if (!markers) return;
  if (!state.video || !state.numFrames) {
    markers.innerHTML = '';
    updateTimelinePlayhead();
    return;
  }
  const maxFrame = Math.max(1, state.numFrames - 1);
  const rallyBands = state.rallies.map((rally, idx) => {
    const startPct = clamp((rally.start * state.fps) / maxFrame, 0, 1) * 100;
    const endPct = clamp((rally.end * state.fps) / maxFrame, 0, 1) * 100;
    const active = rally.id === selectedRallyId;
    return `<div class="absolute top-0 bottom-0 rounded-sm ${active ? 'bg-emerald-400/[0.18] border-x border-emerald-300/50' : 'bg-emerald-500/[0.08] border-x border-emerald-400/[0.15]'}"
      title="R${idx + 1} ${formatSeconds(rally.start)}-${formatSeconds(rally.end)}"
      style="left:${startPct}%; width:${Math.max(0.2, endPct - startPct)}%"></div>`;
  }).join('');
  const eventButtons = visibleEventEntries().map(({ event, idx }) => {
    const pct = clamp(event.frame / maxFrame, 0, 1) * 100;
    const color = COLORS[event.label] || '#818CF8';
    const active = idx === selectedIdx;
    return `<button type="button" data-idx="${idx}" title="${escapeHtml(event.label)} frame ${event.frame}"
      class="absolute top-1/2 -translate-y-1/2 rounded-full border border-black/50 transition-transform ${active ? 'w-3 h-5 -ml-1.5 scale-110' : 'w-1.5 h-4 -ml-px hover:scale-125'}"
      style="left:${pct}%; background:${color}"></button>`;
  }).join('');
  markers.innerHTML = rallyBands + eventButtons;
  markers.querySelectorAll('button[data-idx]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      jumpToEvent(Number(btn.dataset.idx));
    });
  });
  updateTimelinePlayhead();
}

function updateTimelinePlayhead() {
  const playhead = document.getElementById('act-timeline-playhead');
  if (!playhead) return;
  const maxFrame = Math.max(1, state.numFrames - 1);
  const pct = state.video ? clamp(currentFrame() / maxFrame, 0, 1) * 100 : 0;
  playhead.style.left = `${pct}%`;
}

function onTimelineClick(e) {
  if (!state.video || !state.numFrames) return;
  if (e.target.closest('button[data-idx]')) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const pct = clamp((e.clientX - rect.left) / rect.width, 0, 1);
  seekFrame(Math.round(pct * Math.max(0, state.numFrames - 1)));
}

function refreshPlayhead() {
  if (!videoEl) return;
  const t = videoEl.currentTime || 0;
  const frame = currentFrame();
  document.getElementById('act-time').textContent = `${formatSeconds(t)} / f${frame}`;
  document.getElementById('act-meta').textContent = state.video
    ? `${state.fps.toFixed(3)} fps · ${state.numFrames} frames`
    : '';
  if (frame !== lastOverlayFrame) {
    lastOverlayFrame = frame;
    renderOverlay();
  }
  updateRallyNow(frame);
  updateTimelinePlayhead();
}

function togglePlayback() {
  if (!videoEl?.src) return;
  if (videoEl.paused) {
    videoEl.play().catch(e => showToast(`Play failed: ${e.message}`, 'error'));
  } else {
    videoEl.pause();
  }
}

function updatePlaybackButton() {
  const btn = document.getElementById('act-play-toggle');
  const icon = document.getElementById('act-play-icon');
  if (!btn || !icon) return;
  const playing = Boolean(videoEl?.src) && !videoEl.paused && !videoEl.ended;
  btn.title = playing ? 'Pause' : 'Play';
  btn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  icon.innerHTML = playing
    ? '<path d="M7 5h4v14H7zM13 5h4v14h-4z"/>'
    : '<path d="M8 5v14l11-7z"/>';
}

function currentFrame() {
  const maxFrame = Math.max(0, state.numFrames - 1);
  return clamp(Math.round((videoEl?.currentTime || 0) * state.fps), 0, maxFrame);
}

function seekFrame(frame) {
  if (!videoEl || !state.fps) return;
  videoEl.currentTime = Math.max(0, frame / state.fps);
  refreshPlayhead();
}

function jumpToEvent(idx) {
  const event = state.events[idx];
  if (!event) return;
  selectedIdx = idx;
  videoEl?.pause();
  seekFrame(event.frame);
  updatePlaybackButton();
  renderEvents();
}

function stepFrame(delta) {
  if (!state.video || !state.fps) return;
  videoEl.pause();
  seekFrame(currentFrame() + delta);
}

function togglePointMode() {
  pointMode = !pointMode;
  updatePointModeButton();
}

function updatePointModeButton() {
  const btn = document.getElementById('act-point-mode');
  if (!btn) return;
  btn.textContent = pointMode ? 'Point mode' : 'Review mode';
  btn.title = pointMode
    ? 'Point mode: click the video to add the selected action label'
    : 'Review mode: clicking the video will not add points';
  btn.setAttribute('aria-pressed', String(pointMode));
  btn.classList.toggle('ring-2', pointMode);
  btn.classList.toggle('ring-emerald-400/40', pointMode);
  btn.style.background = pointMode ? 'rgba(34, 197, 94, 0.18)' : '';
  btn.style.borderColor = pointMode ? 'rgba(34, 197, 94, 0.35)' : '';
  btn.style.color = pointMode ? '#86EFAC' : '';
}

function onKeydown(e) {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
  if (e.key >= '1' && e.key <= '6') {
    const label = labels[Number(e.key) - 1];
    if (label) {
      selectedLabel = label;
      renderLabels();
    }
    return;
  }
  if (e.key === ' ') {
    e.preventDefault();
    togglePlayback();
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    stepFrame(e.shiftKey ? -10 : -1);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    stepFrame(e.shiftKey ? 10 : 1);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    addEvent(0.5, 0.5);
  } else if (e.key.toLowerCase() === 'p') {
    e.preventDefault();
    togglePointMode();
  } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIdx >= 0) {
    state.events.splice(selectedIdx, 1);
    selectedIdx = -1;
    markDirty();
    renderEvents();
  }
}

function formatSeconds(seconds) {
  if (!Number.isFinite(seconds)) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds - m * 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function makeClientId(prefix) {
  if (globalThis.crypto?.randomUUID) return `${prefix}_${globalThis.crypto.randomUUID().replaceAll('-', '').slice(0, 16)}`;
  return `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function findRallyForFrame(frame, context = state) {
  const fps = Number(context.fps) || 30;
  const time = frame / fps;
  return (context.rallies || []).find(r => time >= r.start && time < r.end) || null;
}

function rallyIndex(id) {
  if (!id) return -1;
  return (state.rallies || []).findIndex(r => r.id === id);
}

function rallyLabel(event) {
  const idx = rallyIndex(event.rally_id);
  return idx >= 0 ? `R${idx + 1}` : 'out';
}

function withRallyFields(event, context = state) {
  const fps = Number(context.fps) || 30;
  const frame = Math.max(0, Math.round(Number(event.frame) || 0));
  const time = frame / fps;
  const rally = findRallyForFrame(frame, context);
  return {
    ...event,
    frame,
    time: round4(time),
    rally_id: rally?.id || null,
    relative_frame: rally ? Math.max(0, Math.round((time - rally.start) * fps)) : null,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round4(value) {
  return Math.round(value * 10000) / 10000;
}
