/**
 * Action Label page - rally-scoped frame/point action annotation.
 */
import {
  api, API, SSEClient, pageHeader, card, sectionTitle, btnSmall,
  showToast, showConfirm, emptyState, escapeHtml, selectCls, inputCls, renderJobProgress,
} from '../shared.js';

const COLORS = {
  serve: '#38BDF8',
  receive: '#22C55E',
  set: '#A78BFA',
  spike: '#F97316',
  block: '#EF4444',
  score: '#FBBF24',
};
const OUTSIDE_GROUP_ID = '__outside_actions__';
const WAVEFORM_POINTS_PER_SECOND = 120;
const WAVEFORM_MIN_POINTS = 2400;
const WAVEFORM_MAX_POINTS = 96000;
const WAVEFORM_SCALE_PERCENTILE = 0.98;
const WAVEFORM_SCALE_HEADROOM = 1.35;
const WAVEFORM_MIN_SCALE = 0.02;
const WAVEFORM_VERTICAL_FILL = 0.40;
const WAVEFORM_PEAK_GAIN = 0.55;
const WAVEFORM_RMS_GAIN = 1.15;

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
let waveform = { video: '', loading: false, error: '', hasAudio: false, duration: 0, peaks: [], rms: [] };
let waveformRequestId = 0;
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
let videoWrapEl = null;
let overlayEl = null;
let overlayResizeObserver = null;
let selectedIdx = -1;
let selectedRallyId = 'all';
let timelineScope = 'rally';
let expandedRallyIds = new Set();
let tickTimer = null;
let lastOverlayFrame = -1;
// Frame-clock state model (three interacting variables):
//   lockedFrame          — when paused/stepping, the exact frame we are pinned
//                          to. While set, it is the source of truth for the
//                          current frame and overrides presentedMediaTime;
//                          cleared (null) whenever the video is playing.
//   presentedMediaTime   — mediaTime of the last frame the compositor actually
//                          presented, reported by requestVideoFrameCallback;
//                          used for the readback when not locked.
//   frameClockGeneration — monotonically bumped on every seek/reset so that a
//                          still-pending requestVideoFrameCallback from before
//                          the seek can detect it is stale and bail out.
let presentedMediaTime = null;
let videoFrameCallbackId = null;
let frameClockGeneration = 0;
let lockedFrame = null;
let dragPoint = null;
let suppressOverlayClick = false;

function actionKbdHint() {
  const rows = [
    [['1-6', 'label'], ['Space', 'play/pause'], ['Left/Right', '1 frame'], ['Shift+Left/Right', '10 frames'], ['Enter', 'add center'], ['Right click', 'hidden'], ['Del', 'remove']],
    [['P', 'point mode'], ['O', 'timeline']],
  ];
  const renderShortcut = ([key, label]) => (
    `<span class="inline-flex items-center gap-1">
      <kbd class="px-1.5 py-0.5 rounded bg-surface-200 border border-border text-[10px] font-heading text-text-secondary">${key}</kbd>
      ${label}
    </span>`
  );
  return `<div class="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-x-3 gap-y-1 text-[11px] text-text-muted px-1">
    <span class="font-heading text-text-secondary">Keys:</span>
    <div class="flex flex-wrap items-center gap-x-4 gap-y-1">${rows[0].map(renderShortcut).join('')}</div>
    <span aria-hidden="true"></span>
    <div class="flex flex-wrap items-center gap-x-4 gap-y-1">${rows[1].map(renderShortcut).join('')}</div>
  </div>`;
}

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-5">
      ${pageHeader('Action Label', 'Add action points on top of rally annotations')}

      ${card(`
        <div class="space-y-3">
          <div class="grid grid-cols-1 lg:grid-cols-[8.5rem_8.5rem_minmax(20rem,1fr)_auto_auto] gap-3 items-end">
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Kind</span>
              <select id="act-kind" class="${selectCls} w-full" title="Filter by cut kind">
                <option value="all">All kinds</option>
                <option value="broadcast">Broadcast only</option>
                <option value="sideline">Sideline only</option>
              </select>
            </label>
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Status</span>
              <select id="act-progress" class="${selectCls} w-full" title="Filter by action label status">
                <option value="all">All</option>
                <option value="unlabeled">Unlabeled</option>
                <option value="pre-labeled">Pre-Labeled</option>
                <option value="labeled">Labeled</option>
              </select>
            </label>
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Video</span>
              <div class="relative" id="act-video-combo-wrap">
                <input id="act-video-combo" class="${selectCls} w-full pr-8 font-mono text-xs" role="combobox" aria-expanded="false" aria-controls="act-video-options" placeholder="Type to search full filename..." autocomplete="off">
                <button id="act-video-clear" type="button" class="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary text-xs px-1" title="Clear video search">x</button>
                <div id="act-video-options" class="hidden absolute z-[90] left-0 right-0 top-full mt-1 max-h-[24rem] overflow-auto rounded-xl border border-border bg-surface-100 shadow-2xl shadow-black/30 p-1"></div>
              </div>
            </label>
            ${btnSmall('Load', 'id="act-load" title="Load selected video"', 'primary')}
            ${btnSmall('Export JSONL', 'id="act-export" title="Export saved action annotations"')}
          </div>
          <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-text-muted font-heading tabular-nums">
            <span id="act-video-count">Loading videos...</span>
            <span id="act-export-summary"></span>
          </div>
        </div>
      `, 'relative z-40 overflow-visible')}

      <div class="relative z-0 flex flex-col lg:flex-row gap-5">
        <div class="flex-1 min-w-0 space-y-4">
          ${card(`
            <div class="space-y-4">
              <div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <h3 class="text-sm font-heading font-semibold text-text-primary">Action Labels</h3>
                <div id="act-labels" class="grid grid-cols-3 sm:grid-cols-6 gap-2 sm:min-w-[29rem]"></div>
              </div>

              <div id="act-video-wrap" class="relative bg-black rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-lg shadow-black/30">
                <video id="act-player" class="block w-full max-h-[45vh] object-contain bg-black mx-auto" playsinline preload="metadata"></video>
                <div id="act-overlay" class="absolute inset-0 pointer-events-none"></div>
              </div>

              <div class="space-y-2">
                <div id="act-timeline" class="relative h-12 rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-inner cursor-pointer" title="Click to seek" style="background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)">
                  <div id="act-timeline-playhead" class="absolute top-0 bottom-0 w-px bg-accent z-20 pointer-events-none" style="left:0%"></div>
                  <div id="act-timeline-markers" class="absolute inset-0"></div>
                </div>
                <div id="act-waveform-wrap" class="relative h-14 rounded-xl overflow-hidden ring-1 ring-white/[0.06] bg-surface-100/45 cursor-pointer touch-manipulation" title="Click to seek" aria-label="Audio waveform">
                  <canvas id="act-waveform" class="absolute inset-0 w-full h-full"></canvas>
                  <div id="act-waveform-playhead" class="absolute top-0 bottom-0 w-px bg-accent/80 z-10 pointer-events-none" style="left:0%"></div>
                  <div id="act-waveform-status" class="absolute left-2 top-1.5 text-[10px] text-text-muted font-heading pointer-events-none"></div>
                </div>
                <div class="flex flex-wrap items-center justify-between px-0.5 gap-3">
                  <div class="flex flex-wrap items-center gap-2 min-w-0">
                    <span id="act-time" class="shrink-0 text-sm font-heading text-text-primary tabular-nums bg-surface-200/50 px-2.5 py-1 rounded-lg border border-border">00:00</span>
                    ${btnSmall('Current rally', 'id="act-timeline-scope" title="Timeline range"')}
                    <span id="act-timeline-range" class="text-[11px] text-text-muted font-heading tabular-nums truncate max-w-[13rem]"></span>
                  </div>
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
                <span id="act-meta"></span>
              </div>
            </div>
          `)}

          ${actionKbdHint()}
        </div>

        <div class="lg:w-[420px] lg:flex-shrink-0 min-w-0">
          ${card(`
            <div class="space-y-3">
              ${sectionTitle(
                'Rallies',
                '<span id="act-count" class="text-sm">(0)</span>',
                `${btnSmall('Save<span id="act-save-star" class="invisible absolute right-2 top-1/2 -translate-y-1/2">*</span>', 'id="act-save" style="position:relative;min-width:4.5rem" title="Save action labels"', 'success')} ${btnSmall('Clear', 'id="act-clear" style="min-width:4.5rem"', 'danger')}`,
              )}
              <div class="space-y-2">
                <div class="flex items-center gap-2">
                  <select id="act-rally-scope" class="${selectCls} flex-1 text-xs" title="Choose rally">
                    <option value="all">All rallies</option>
                  </select>
                  ${btnSmall('Prev', 'id="act-rally-prev" title="Previous rally"')}
                  ${btnSmall('Next', 'id="act-rally-next" title="Next rally"')}
                </div>
                <div class="min-h-4 text-right text-[11px] text-text-muted">
                  <span id="act-dirty"></span>
                </div>
              </div>
              <div class="h-px bg-border"></div>
              <div id="act-events" class="space-y-1.5 max-h-[65vh] lg:max-h-[calc(100vh-16rem)] overflow-y-auto overflow-x-hidden pr-1 scrollbar-thin"></div>
            </div>
          `)}
        </div>
      </div>

      <details id="act-spot-tools" class="rounded-xl border border-border bg-surface-100/40 px-3 py-2">
        <summary class="cursor-pointer list-none text-xs font-heading text-text-secondary hover:text-text-primary select-none">
          <span class="inline-flex flex-wrap items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-primary-light/80"></span>
            SPOT pre-label
            <span class="text-text-muted font-normal">checkpoint, score threshold, GPU handoff</span>
          </span>
        </summary>
        <div class="mt-3 space-y-2.5">
          <div class="grid grid-cols-1 md:grid-cols-[minmax(16rem,1fr)_7rem_7rem_auto_auto] gap-2.5 items-end">
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
              <input id="act-spot-batch" type="number" min="1" max="128" step="1" value="64" class="${inputCls} w-full">
            </label>
            <label class="flex items-center gap-2 text-xs text-text-secondary pb-3 cursor-pointer">
              <input id="act-spot-stop-vllm" type="checkbox" class="accent-primary w-3.5 h-3.5">
              Stop vLLM
            </label>
            ${btnSmall('Run SPOT', 'id="act-spot-run" title="Run ~/yp-spot model and create an action pre-label"', 'primary')}
          </div>
          <div class="grid grid-cols-2 md:grid-cols-5 gap-2.5 items-end">
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Decoder</span>
              <select id="act-spot-decoder" class="${selectCls} w-full">
                <option value="opencv" selected>OpenCV</option>
                <option value="nvdec">NVDEC (GPU)</option>
              </select>
            </label>
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Producers</span>
              <input id="act-spot-producers" type="number" min="1" max="8" step="1" value="2" class="${inputCls} w-full">
            </label>
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Threads</span>
              <input id="act-spot-threads" type="number" min="1" max="8" step="1" value="1" class="${inputCls} w-full">
            </label>
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Prefetch</span>
              <input id="act-spot-prefetch" type="number" min="1" max="8" step="1" value="2" class="${inputCls} w-full">
            </label>
            <label class="space-y-1.5 min-w-0">
              <span class="text-[10px] uppercase tracking-widest text-text-muted font-semibold">Chunk</span>
              <input id="act-spot-chunk" type="number" min="1" max="512" step="16" value="256" class="${inputCls} w-full">
            </label>
          </div>
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
  startVideoFrameClock();
  refreshPlayhead();
}

export function deactivate() {
  document.removeEventListener('keydown', onKeydown);
  if (tickTimer) clearInterval(tickTimer);
  tickTimer = null;
  overlayResizeObserver?.disconnect();
  overlayResizeObserver = null;
  window.removeEventListener('resize', onVideoGeometryChange);
  spotClient?.stop();
  spotClient = null;
  stopVideoFrameClock();
  videoEl?.pause();
}

function bindEvents() {
  videoWrapEl = document.getElementById('act-video-wrap');
  videoEl = document.getElementById('act-player');
  overlayEl = document.getElementById('act-overlay');

  const combo = document.getElementById('act-video-combo');
  const openVideoDropdown = () => {
    // Clear the leftover filename so the dropdown shows every video again
    // (otherwise the previously-loaded name filters the list down to one).
    if (videoSearch) {
      videoSearch = '';
      combo.value = '';
      activeVideoOption = 0;
    }
    videoDropdownOpen = true;
    renderVideoOptions();
  };
  combo.addEventListener('focus', openVideoDropdown);
  combo.addEventListener('click', openVideoDropdown);
  combo.addEventListener('input', (e) => {
    videoSearch = e.target.value;
    selectedVideo = '';
    activeVideoOption = 0;
    videoDropdownOpen = true;
    renderVideoOptions();
  });
  combo.addEventListener('keydown', (e) => {
    const matches = filteredVideos();
    const maxActive = Math.max(0, matches.length - 1);
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
  document.getElementById('act-timeline-scope').addEventListener('click', toggleTimelineScope);
  document.getElementById('act-timeline').addEventListener('click', onTimelineClick);
  document.getElementById('act-waveform-wrap').addEventListener('click', onTimelineClick);
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
  document.getElementById('act-clear').addEventListener('click', clearEvents);

  videoEl.addEventListener('loadedmetadata', () => {
    lockedFrame = null;
    stopVideoFrameClock();
    startVideoFrameClock();
    syncOverlayGeometry();
    refreshPlayhead();
    updatePlaybackButton();
  });
  videoEl.addEventListener('loadeddata', onVideoGeometryChange);
  videoEl.addEventListener('seeked', () => {
    presentedMediaTime = lockedFrame !== null && state.fps
      ? lockedFrame / state.fps
      : videoEl.currentTime || 0;
    startVideoFrameClock();
    onVideoGeometryChange();
  });
  videoEl.addEventListener('timeupdate', onVideoTimeUpdate);
  videoEl.addEventListener('play', () => {
    lockedFrame = null;
    presentedMediaTime = videoEl.currentTime || presentedMediaTime;
    restartVideoFrameClock();
    updatePlaybackButton();
  });
  videoEl.addEventListener('pause', updatePlaybackButton);
  videoEl.addEventListener('ended', updatePlaybackButton);
  videoEl.addEventListener('click', (e) => {
    if (!state.video || !pointMode) return;
    const point = clientToVideoPoint(e.clientX, e.clientY);
    if (!point) return;
    const [x, y] = point;
    addEvent(x, y);
  });
  videoWrapEl.addEventListener('contextmenu', onVideoContextMenu);

  overlayResizeObserver?.disconnect();
  if (window.ResizeObserver && videoWrapEl) {
    overlayResizeObserver = new ResizeObserver(onVideoGeometryChange);
    overlayResizeObserver.observe(videoWrapEl);
    overlayResizeObserver.observe(videoEl);
  }
  window.removeEventListener('resize', onVideoGeometryChange);
  window.addEventListener('resize', onVideoGeometryChange);

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
  if (activeVideoOption >= matches.length) {
    activeVideoOption = Math.max(0, matches.length - 1);
  }

  if (combo && document.activeElement !== combo && selectedVideo && combo.value !== selectedVideo) {
    combo.value = selectedVideo;
  }
  if (combo) {
    combo.setAttribute('aria-expanded', String(videoDropdownOpen));
    combo.title = combo.value || 'Type to search full filename';
  }

  if (list) {
    list.classList.toggle('hidden', !videoDropdownOpen);
    if (!matches.length) {
      list.innerHTML = '<div class="px-3 py-2 text-xs text-text-muted">No videos match</div>';
    } else {
      list.innerHTML = matches.map((v, idx) => {
        const active = idx === activeVideoOption;
        const selected = v.name === selectedVideo;
        const status = actionStatusText(v);
        const kind = v.kind === 'sideline' ? 'SIDE' : 'CAST';
        const rallyTag = rallySourceTag(v);
        const rallyTitle = rallySourceTitle(v);
        const actionTag = actionStatusTag(v);
        return `<button type="button" data-video-name="${escapeHtml(v.name)}"
          class="w-full text-left px-3 py-2.5 rounded-lg text-xs transition-colors ${active || selected ? 'bg-primary/10 text-text-primary' : 'text-text-secondary hover:bg-white/[0.06] hover:text-text-primary'}">
          <span class="flex items-center gap-2 min-w-0">
            <span class="shrink-0 inline-flex h-5 min-w-9 items-center justify-center px-1.5 rounded bg-white/5 text-[10px] leading-none font-heading text-text-muted">${kind}</span>
            <span class="shrink-0 inline-flex w-5 h-5 items-center justify-center" title="${escapeHtml(rallyTitle)}">${rallyTag}</span>
            ${actionTag}
            <span class="min-w-0 flex-1 whitespace-normal break-all leading-5 font-mono">${escapeHtml(v.name)}</span>
            <span class="shrink-0 inline-flex h-5 items-center px-1.5 rounded bg-white/5 text-[10px] leading-none font-heading text-text-muted">${escapeHtml(status)}</span>
          </span>
        </button>`;
      }).join('');
    }
  }

  const countEl = document.getElementById('act-video-count');
  if (countEl) {
    const labeled = videos.filter(hasFinalActionAnnotation).length;
    countEl.textContent = `${matches.length} shown / ${videos.length} total · ${labeled} action labeled`;
  }
  renderExportSummary();
}

function actionStatusText(video) {
  const count = Math.max(0, Number(video?.event_count) || 0);
  if (hasFinalActionAnnotation(video)) return `${count} labeled`;
  if (hasActionPrelabel(video)) return `${count} pre-label`;
  return 'unlabeled';
}

function actionStatusTag(video) {
  const reviewed = hasFinalActionAnnotation(video);
  const hasPrelabel = hasActionPrelabel(video);
  const count = Math.max(0, Number(video?.event_count) || 0);
  const title = reviewed
    ? `Action labeled: ${count} event(s)`
    : hasPrelabel
      ? `SPOT pre-label only: ${count} event(s)`
      : 'Action not labeled';
  if (reviewed) {
    return `<span class="shrink-0 inline-flex w-5 h-5 items-center justify-center rounded-full bg-emerald-500/16 text-emerald-300 ring-1 ring-emerald-400/30 leading-none" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}">
      <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"></path>
      </svg>
    </span>`;
  }
  if (hasPrelabel) {
    return `<span class="shrink-0 inline-flex w-5 h-5 items-center justify-center rounded-full bg-amber-500/12 text-amber-300 ring-1 ring-amber-400/30 font-heading text-[10px] leading-none" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}">P</span>`;
  }
  return `<span class="shrink-0 inline-flex w-5 h-5 items-center justify-center rounded-full ring-1 ring-white/10 leading-none" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}">
    <span class="w-2 h-2 rounded-full border border-text-muted/60"></span>
  </span>`;
}

function hasActiveActionAnnotation(video) {
  return Boolean(video?.has_action_annotation || video?.has_action_final_annotation || video?.has_action_pre_annotation);
}

function hasFinalActionAnnotation(video) {
  return Boolean(video?.action_reviewed);
}

function hasActionPrelabel(video) {
  return hasActiveActionAnnotation(video) && !hasFinalActionAnnotation(video);
}

function rallySourceTag(video) {
  const sources = video?.rally_sources || [];
  if (sources.includes('annotation')) return '✅';
  if (sources.includes('pre-annotation')) return '⚡';
  if (sources.includes('tad-prediction')) return '🤖';
  return '—';
}

function rallySourceTitle(video) {
  const sources = video?.rally_sources || [];
  if (sources.includes('annotation')) return 'Rally annotation reviewed';
  if (sources.includes('pre-annotation')) return 'Rally pre-label';
  if (sources.includes('tad-prediction')) return 'Raw TAD prediction (lower quality)';
  return 'No rally annotation';
}

function filteredVideos() {
  const needle = videoSearch.trim().toLowerCase();
  return videos.filter(v => {
    if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
    if (progressFilter === 'unlabeled' && hasActiveActionAnnotation(v)) return false;
    if (progressFilter === 'pre-labeled' && !hasActionPrelabel(v)) return false;
    if (progressFilter === 'labeled' && !hasFinalActionAnnotation(v)) return false;
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
  if (combo) {
    combo.setAttribute('aria-expanded', 'false');
    // Restore the loaded video's name if the box was cleared without picking a new one.
    if (selectedVideo && combo.value !== selectedVideo) {
      combo.value = selectedVideo;
      videoSearch = selectedVideo;
    }
  }
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
    if (hasFinalActionAnnotation(v)) {
      acc.videos += 1;
      acc.events += Math.max(0, Number(v.event_count) || 0);
    }
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
      body: 'SPOT pre-label will discard unsaved changes in this view.',
      confirmText: 'Discard & Run',
      variant: 'danger',
    });
    if (!ok) return;
    state.dirty = false;
    updateDirtyUi();
  }

  const existing = hasActiveActionAnnotation(videos.find(v => v.name === name));
  if (existing) {
    const ok = await showConfirm({
      title: 'Overwrite action pre-label?',
      body: `${name}\n\nThis will replace the active action pre-label. If a saved action label exists, it will be replaced by the new pre-label.`,
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
    const decodeProducers = Number(document.getElementById('act-spot-producers').value) || 2;
    const job = await api(API.actionAnnotate.prelabel, {
      method: 'POST',
      body: {
        video: name,
        checkpoint: document.getElementById('act-spot-checkpoint').value,
        batch_size: Number(document.getElementById('act-spot-batch').value) || 64,
        num_workers: decodeProducers,
        clip_len: 64,
        decoder: document.getElementById('act-spot-decoder').value,
        decode_producers: decodeProducers,
        decoder_threads: Number(document.getElementById('act-spot-threads').value) || 1,
        prefetch_factor: Number(document.getElementById('act-spot-prefetch').value) || 2,
        decode_chunk_frames: Number(document.getElementById('act-spot-chunk').value) || 256,
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
  if (!state.rallies.some(r => r.rally_id === selectedRallyId)) {
    selectedRallyId = state.rallies[0]?.rally_id || 'all';
  }
}

function currentRally() {
  normalizeSelectedRally();
  if (selectedRallyId === 'all') return null;
  return state.rallies.find(r => r.rally_id === selectedRallyId) || null;
}

function selectedRallyIndex() {
  normalizeSelectedRally();
  return state.rallies.findIndex(r => r.rally_id === selectedRallyId);
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

function eventEntriesForRally(rallyId) {
  return state.events
    .map((event, idx) => ({ event, idx }))
    .filter(({ event }) => event.rally_id === rallyId);
}

function outsideEventEntries() {
  return state.events
    .map((event, idx) => ({ event, idx }))
    .filter(({ event }) => !event.rally_id);
}

function setExpandedRally(rallyId) {
  const key = rallyId === OUTSIDE_GROUP_ID ? OUTSIDE_GROUP_ID : normalizeRallyId(rallyId);
  expandedRallyIds = key ? new Set([key]) : new Set();
}

function renderRallies() {
  const scope = document.getElementById('act-rally-scope');
  const prev = document.getElementById('act-rally-prev');
  const next = document.getElementById('act-rally-next');
  if (!scope) return;

  normalizeSelectedRally();
  const hasVideo = Boolean(state.video);
  const hasRallies = state.rallies.length > 0;

  const options = [
    `<option value="all">All rallies (${state.events.length})</option>`,
    ...state.rallies.map((rally, idx) => (
      `<option value="${rally.rally_id}">R${idx + 1} · ${formatSeconds(rally.start)}-${formatSeconds(rally.end)} · ${rallyEventCount(rally.rally_id)}</option>`
    )),
  ].join('');
  if (scope.innerHTML !== options) scope.innerHTML = options;
  scope.value = selectedRallyId === 'all' ? 'all' : String(selectedRallyId);
  scope.disabled = !hasVideo;

  if (prev) prev.disabled = !hasRallies;
  if (next) next.disabled = !hasRallies;
}

function selectRally(rallyId, { seek = true, expand = true } = {}) {
  selectedRallyId = rallyId === 'all' ? 'all' : normalizeRallyId(rallyId);
  normalizeSelectedRally();
  if (expand) setExpandedRally(selectedRallyId !== 'all' ? selectedRallyId : null);
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
  selectRally(state.rallies[nextIdx].rally_id);
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
    selectedRallyId = rallies[0]?.rally_id || 'all';
    expandedRallyIds = new Set(rallies[0]?.rally_id ? [rallies[0].rally_id] : []);
    lastOverlayFrame = -1;
    videoEl.pause();
    stopVideoFrameClock();
    videoEl.src = `/api${API.actionAnnotate.video(state.video)}`;
    videoEl.load();
    startVideoFrameClock();
    updatePlaybackButton();
    renderEvents();
    loadWaveform(state.video);
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
  const star = document.getElementById('act-save-star');
  if (dirty) dirty.textContent = state.dirty ? 'Unsaved changes' : '';
  if (star) star.classList.toggle('invisible', !state.dirty);
}

function normalizeRallies(rallies) {
  return (rallies || []).map((r, idx) => ({
    rally_id: normalizeRallyId(r.rally_id) || idx + 1,
    start: Number(r.start) || 0,
    end: Number(r.end) || 0,
    label: r.label || 'rally',
  })).sort((a, b) => a.start - b.start || a.end - b.end || a.rally_id - b.rally_id);
}

function normalizeEvents(events, context = state) {
  return events.map(e => ({
    id: e.id || makeClientId('act'),
    rally_id: normalizeRallyId(e.rally_id),
    frame: Math.max(0, Math.round(Number(e.frame) || 0)),
    time: Number(e.time) || null,
    relative_frame: Number.isInteger(e.relative_frame) ? e.relative_frame : null,
    label: labels.includes(e.label) ? e.label : labels[0],
    xy: [
      clamp(Number(e.xy?.[0] ?? e.x ?? 0.5), 0, 1),
      clamp(Number(e.xy?.[1] ?? e.y ?? 0.5), 0, 1),
    ],
    visible: parseEventVisible(e.visible),
  })).map(e => withRallyFields(e, context))
    .sort((a, b) => a.frame - b.frame || a.label.localeCompare(b.label) || a.id.localeCompare(b.id));
}

function parseEventVisible(value) {
  if (typeof value === 'string') {
    return !['0', 'false', 'no', 'off'].includes(value.trim().toLowerCase());
  }
  return value !== false;
}

function addEvent(x, y, { visible = true } = {}) {
  if (!state.video) return showToast('Load a video first', 'warning');
  const frame = frameForNewEvent();
  if (frame !== currentFrame()) seekFrame(frame);
  const event = withRallyFields({
    id: makeClientId('act'),
    frame,
    label: selectedLabel,
    xy: [round4(x), round4(y)],
    visible: Boolean(visible),
  });
  state.events.push(event);
  sortEvents();
  selectedIdx = state.events.indexOf(event);
  if (event.rally_id) {
    selectedRallyId = event.rally_id;
    setExpandedRally(event.rally_id);
  } else {
    setExpandedRally(OUTSIDE_GROUP_ID);
  }
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
  setExpandedRally(selectedRallyId !== 'all' ? selectedRallyId : null);
  markDirty();
  renderEvents();
}

async function save() {
  if (!state.video) return showToast('No video loaded', 'warning');
  const btn = document.getElementById('act-save');
  btn.disabled = true;
  try {
    const result = await api(API.actionAnnotate.annotations, {
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
    markVideoActionReviewed(state.video, Number(result?.count) || state.events.length);
    renderVideoOptions();
    showToast('Action annotations saved', 'success');
  } catch (e) {
    showToast(`Save failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    updateDirtyUi();
  }
}

function markVideoActionReviewed(videoName, eventCount) {
  const item = videos.find(v => v.name === videoName);
  if (!item) return;
  item.has_action_annotation = true;
  item.has_action_final_annotation = true;
  item.action_annotation_source = 'action-annotations';
  item.action_reviewed = true;
  item.event_count = Math.max(0, Number(eventCount) || 0);
}

function renderEvents() {
  renderRallies();
  if (selectedIdx >= 0 && (!state.events[selectedIdx] || !eventVisibleInScope(state.events[selectedIdx]))) {
    selectedIdx = -1;
  }
  document.getElementById('act-count').textContent = state.rallies.length
    ? `(${state.rallies.length} rally · ${state.events.length} action)`
    : `(${state.events.length} action)`;
  updateDirtyUi();

  const el = document.getElementById('act-events');
  if (!state.video) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.55 2.28a1 1 0 010 1.44L15 16m-6 0l-4.55-2.28a1 1 0 010-1.44L9 10"/></svg>',
      'No video loaded',
      '',
    );
    renderOverlay();
    renderTimeline();
    renderWaveform();
    return;
  }

  const outsideEntries = outsideEventEntries();
  if (!state.rallies.length && !outsideEntries.length) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>',
      'No rally annotations',
      '',
    );
    renderOverlay();
    renderTimeline();
    renderWaveform();
    return;
  }

  const rallyRows = state.rallies.map((rally, idx) => rallyRow(rally, idx)).join('');
  const outsideRow = outsideEntries.length ? outsideActionsRow(outsideEntries) : '';
  el.innerHTML = rallyRows + outsideRow;
  renderOverlay();
  renderTimeline();
  renderWaveform();
}

function rallyRow(rally, idx) {
  const selected = rally.rally_id === selectedRallyId;
  const expanded = expandedRallyIds.has(rally.rally_id);
  const entries = eventEntriesForRally(rally.rally_id);
  const rowCls = selected
    ? 'border-emerald-500/40 bg-emerald-500/[0.08]'
    : 'border-emerald-500/15 bg-emerald-500/[0.04] hover:bg-emerald-500/[0.08]';
  const durationSec = Math.max(0, rally.end - rally.start).toFixed(1);
  return `
    <div class="space-y-1.5">
      <div class="act-rally-row flex items-center gap-2.5 px-3 py-2.5 rounded-xl border ${rowCls} cursor-pointer transition-all duration-200 group" data-rally-id="${rally.rally_id}">
        <span class="text-[10px] font-heading text-text-muted/60 w-4 text-right select-none">${idx + 1}</span>
        <button class="flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/25 hover:bg-emerald-500/30 cursor-pointer transition-colors duration-200" data-action="toggle-rally" aria-expanded="${expanded}" title="${expanded ? 'Hide action labels' : 'Show action labels'}">
          <svg class="w-3 h-3 pointer-events-none transition-transform duration-150 ${expanded ? 'rotate-90' : ''}" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>
          <span class="pointer-events-none">actions</span>
          <span class="pointer-events-none text-emerald-200/70">${entries.length}</span>
        </button>
        <div class="flex items-center gap-1.5 ml-auto">
          <span class="bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading tabular-nums">${formatSeconds(rally.start)}</span>
          <span class="text-text-muted/40 text-[10px]">→</span>
          <span class="bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading tabular-nums">${formatSeconds(rally.end)}</span>
        </div>
        <span class="text-[10px] text-text-muted font-heading tabular-nums bg-surface-200/40 px-1.5 py-0.5 rounded">${durationSec}s</span>
        <button class="text-primary-light hover:text-white cursor-pointer transition-colors duration-200" data-action="preview-rally" title="Jump to rally end">
          <svg class="w-4 h-4 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
        </button>
      </div>
      ${expanded ? actionPanel(entries, 'No actions in this rally') : ''}
    </div>`;
}

function outsideActionsRow(entries) {
  const expanded = expandedRallyIds.has(OUTSIDE_GROUP_ID);
  return `
    <div class="space-y-1.5">
      <div class="act-rally-row flex items-center gap-2.5 px-3 py-2.5 rounded-xl border border-amber-500/20 bg-amber-500/[0.04] hover:bg-amber-500/[0.08] cursor-pointer transition-all duration-200" data-rally-id="${OUTSIDE_GROUP_ID}">
        <span class="text-[10px] font-heading text-text-muted/60 w-4 text-right select-none">out</span>
        <button class="flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/25 hover:bg-amber-500/25 cursor-pointer transition-colors duration-200" data-action="toggle-rally" aria-expanded="${expanded}" title="${expanded ? 'Hide outside actions' : 'Show outside actions'}">
          <svg class="w-3 h-3 pointer-events-none transition-transform duration-150 ${expanded ? 'rotate-90' : ''}" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>
          <span class="pointer-events-none">outside</span>
          <span class="pointer-events-none text-amber-100/70">${entries.length}</span>
        </button>
        <span class="ml-auto text-[11px] text-text-muted font-heading">outside rally</span>
      </div>
      ${expanded ? actionPanel(entries, 'No outside actions') : ''}
    </div>`;
}

function actionPanel(entries, emptyText) {
  return `
    <div class="ml-6 rounded-xl border border-white/[0.06] bg-surface-100/35 p-2 space-y-1.5">
      ${entries.length ? `
        <div class="grid grid-cols-[1rem_minmax(5.2rem,1fr)_3.8rem_2.8rem_2.35rem] gap-1.5 px-2 text-[10px] uppercase tracking-widest text-text-muted font-semibold">
          <span class="text-right">#</span>
          <span class="pl-4">Label</span>
          <span class="text-center">Frame</span>
          <span class="text-center">Time</span>
          <span></span>
        </div>
        ${entries.map(({ event, idx }, rowIdx) => eventRow(event, idx, rowIdx + 1)).join('')}
      ` : `<div class="px-3 py-2 text-xs text-text-muted">${emptyText}</div>`}
    </div>`;
}

function eventRow(event, idx, rowNumber = idx + 1) {
  const selected = idx === selectedIdx;
  const color = COLORS[event.label] || '#818CF8';
  const visible = event.visible !== false;
  const labelOptions = labels.map(label => `<option value="${escapeHtml(label)}" ${label === event.label ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('');
  const dotStyle = visible
    ? `background:${color}`
    : `background:transparent; border-color:${color}`;
  const rowTitle = visible
    ? (event.id || '')
    : `${event.id ? `${event.id} · ` : ''}non-visible`;
  return `
    <div class="act-event grid grid-cols-[1rem_minmax(5.2rem,1fr)_3.8rem_2.8rem_2.35rem] items-center gap-1.5 px-2 py-1.5 rounded-lg border cursor-pointer transition-colors duration-150 ${selected ? 'bg-primary/10 border-primary/[0.35]' : 'bg-white/[0.035] border-border hover:bg-white/[0.06]'}" data-idx="${idx}" title="${escapeHtml(rowTitle)}">
      <span class="text-right text-[10px] font-heading text-text-muted/70">${rowNumber}</span>
      <span class="flex items-center gap-1.5 min-w-0">
        <button class="w-2.5 h-2.5 rounded-full flex-shrink-0 cursor-pointer ${visible ? '' : 'border'}" data-action="toggle-visible" title="${visible ? 'Visible point — click to mark non-visible' : 'Non-visible event — click to mark visible'}" style="${dotStyle}"></button>
        <select class="min-w-0 w-full bg-surface-100 border border-border rounded-lg px-1.5 py-1 text-xs text-text-primary" data-field="label" data-idx="${idx}">${labelOptions}</select>
      </span>
      <input class="w-full bg-transparent border-b border-white/10 text-text-primary text-[11px] text-center font-heading tabular-nums focus:border-primary-light outline-none" data-field="frame" data-idx="${idx}" value="${event.frame}">
      <span class="text-[10px] text-text-muted font-heading tabular-nums text-center">${formatSeconds(event.frame / state.fps)}</span>
      <span class="flex items-center justify-end gap-1">
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
  if (row) {
    const idx = Number(row.dataset.idx);
    if (e.target.closest('[data-action="delete"]')) {
      state.events.splice(idx, 1);
      selectedIdx = -1;
      markDirty();
      renderEvents();
      return;
    }
    if (e.target.closest('[data-action="toggle-visible"]')) {
      const event = state.events[idx];
      if (event) {
        event.visible = event.visible === false;
        markDirty();
        renderOverlay();
        renderEvents();
      }
      return;
    }
    const editingField = ['INPUT', 'SELECT'].includes(e.target.tagName);
    jumpToEvent(idx, { renderList: !editingField });
    return;
  }

  const rallyRowEl = e.target.closest('.act-rally-row');
  if (!rallyRowEl) return;
  const rallyId = rallyRowEl.dataset.rallyId;
  const action = e.target.closest('[data-action]')?.dataset.action || '';
  if (action === 'toggle-rally') {
    toggleRallyExpanded(rallyId);
    selectRallyFromRow(rallyId, { seek: true, expand: false });
  } else if (action === 'preview-rally') {
    jumpToRallyEnd(rallyId);
  } else {
    selectRallyFromRow(rallyId, { seek: true });
  }
}

function toggleRallyExpanded(rallyId) {
  const key = rallyId === OUTSIDE_GROUP_ID ? OUTSIDE_GROUP_ID : normalizeRallyId(rallyId);
  if (!key) return;
  if (expandedRallyIds.has(key)) {
    setExpandedRally(null);
  } else {
    setExpandedRally(key);
  }
}

function selectRallyFromRow(rallyId, { seek = true, expand = true } = {}) {
  if (rallyId === OUTSIDE_GROUP_ID) {
    selectedRallyId = 'all';
    selectedIdx = -1;
    if (expand) setExpandedRally(OUTSIDE_GROUP_ID);
    if (seek) {
      const firstOutside = outsideEventEntries()[0]?.event;
      if (firstOutside) seekFrame(firstOutside.frame);
    }
    renderEvents();
    return;
  }
  selectRally(rallyId, { seek, expand });
}

function jumpToRallyEnd(rallyId) {
  const parsedRallyId = normalizeRallyId(rallyId);
  const rally = state.rallies.find(r => r.rally_id === parsedRallyId);
  if (!rally) return;
  selectedRallyId = rally.rally_id;
  setExpandedRally(rally.rally_id);
  selectedIdx = -1;
  videoEl?.pause();
  seekFrame(Math.max(0, Math.ceil(rally.end * state.fps) - 1));
  updatePlaybackButton();
  renderEvents();
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
    if (event.rally_id) {
      selectedRallyId = event.rally_id;
      setExpandedRally(event.rally_id);
    } else {
      setExpandedRally(OUTSIDE_GROUP_ID);
    }
  }
  markDirty();
  renderOverlay();
  if (e.type === 'change') renderEvents();
}

function onVideoContextMenu(e) {
  e.preventDefault();
  if (!state.video || e.target.closest('button[data-idx]')) return;
  const mediaRect = videoMediaRect();
  if (
    !mediaRect
    || e.clientX < mediaRect.left
    || e.clientX > mediaRect.right
    || e.clientY < mediaRect.top
    || e.clientY > mediaRect.bottom
  ) return;
  const point = clientToVideoPoint(e.clientX, e.clientY);
  if (!point) return;
  const [x, y] = point;
  addEvent(x, y, { visible: false });
}

function onVideoGeometryChange() {
  syncOverlayGeometry();
  refreshPlayhead();
  renderOverlay();
  renderWaveform();
}

function videoMediaRect() {
  if (!videoEl) return null;
  const rect = videoEl.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;

  const videoWidth = videoEl.videoWidth || 0;
  const videoHeight = videoEl.videoHeight || 0;
  if (!videoWidth || !videoHeight) return rect;

  const mediaRatio = videoWidth / videoHeight;
  const boxRatio = rect.width / rect.height;
  let width = rect.width;
  let height = rect.height;
  let left = rect.left;
  let top = rect.top;

  if (boxRatio > mediaRatio) {
    width = rect.height * mediaRatio;
    left += (rect.width - width) / 2;
  } else if (boxRatio < mediaRatio) {
    height = rect.width / mediaRatio;
    top += (rect.height - height) / 2;
  }

  return {
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
  };
}

function syncOverlayGeometry() {
  if (!overlayEl || !videoWrapEl || !videoEl) return null;
  const wrapRect = videoWrapEl.getBoundingClientRect();
  const mediaRect = videoMediaRect();
  if (!mediaRect || !wrapRect.width || !wrapRect.height) return null;

  overlayEl.style.left = `${mediaRect.left - wrapRect.left}px`;
  overlayEl.style.top = `${mediaRect.top - wrapRect.top}px`;
  overlayEl.style.width = `${mediaRect.width}px`;
  overlayEl.style.height = `${mediaRect.height}px`;
  overlayEl.style.right = 'auto';
  overlayEl.style.bottom = 'auto';

  return {
    width: mediaRect.width,
    height: mediaRect.height,
  };
}

function renderOverlay() {
  if (!overlayEl) return;
  syncOverlayGeometry();
  const frame = currentFrame();
  overlayEl.innerHTML = state.events.map((event, idx) => ({ event, idx }))
    .filter(({ event }) => event.visible !== false && eventVisibleInScope(event) && Math.abs(event.frame - frame) <= 2)
    .map(({ event, idx }) => {
    const color = COLORS[event.label] || '#818CF8';
    const exactFrame = event.frame === frame;
    const eventFrameRing = exactFrame
      ? `<span class="absolute left-1/2 top-1/2 w-5 h-5 rounded-full border-2 border-white/90 pointer-events-none -translate-x-1/2 -translate-y-1/2"
          style="box-shadow:0 0 0 1px ${color}88"></span>`
      : '';
    return `<button type="button" data-idx="${idx}" title="${escapeHtml(event.label)} frame ${event.frame}"
      class="absolute w-6 h-6 -ml-3 -mt-3 rounded-full pointer-events-auto cursor-grab active:cursor-grabbing touch-none group"
      style="left:${event.xy[0] * 100}%; top:${event.xy[1] * 100}%;">
      ${eventFrameRing}
      <span class="absolute left-1/2 top-1/2 w-2 h-2 rounded-full border border-white/85 pointer-events-none -translate-x-1/2 -translate-y-1/2"
        style="background:${color}; box-shadow:0 0 0 1px ${color}55"></span>
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

  const point = clientToVideoPoint(e.clientX, e.clientY);
  if (!point) return;
  const [x, y] = point;
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
  const rect = videoMediaRect();
  if (!rect || !rect.width || !rect.height) return null;
  return [
    round4(clamp((clientX - rect.left) / rect.width, 0, 1)),
    round4(clamp((clientY - rect.top) / rect.height, 0, 1)),
  ];
}

function timelineRange() {
  const fps = Number(state.fps) || 30;
  const maxFrame = Math.max(0, state.numFrames - 1);
  const rally = currentRally();
  if (timelineScope === 'rally' && rally) {
    const startFrame = clamp(Math.round(rally.start * fps), 0, maxFrame);
    const endFrame = clamp(Math.max(startFrame, Math.ceil(rally.end * fps) - 1), 0, maxFrame);
    return {
      scope: 'rally',
      startFrame,
      endFrame,
      startTime: rally.start,
      endTime: rally.end,
      rally,
    };
  }
  return {
    scope: 'video',
    startFrame: 0,
    endFrame: maxFrame,
    startTime: 0,
    endTime: state.duration || (fps ? maxFrame / fps : 0),
    rally: null,
  };
}

function frameToTimelinePct(frame, range = timelineRange()) {
  const width = Math.max(1, range.endFrame - range.startFrame);
  return clamp((frame - range.startFrame) / width, 0, 1) * 100;
}

function timelinePctToFrame(pct, range = timelineRange()) {
  const width = Math.max(0, range.endFrame - range.startFrame);
  return Math.round(range.startFrame + clamp(pct, 0, 1) * width);
}

function eventInTimelineRange(event, range = timelineRange()) {
  return event.frame >= range.startFrame && event.frame <= range.endFrame;
}

function toggleTimelineScope() {
  if (!state.video || !state.numFrames) return;
  if (timelineScope === 'rally') {
    timelineScope = 'video';
  } else if (currentRally()) {
    timelineScope = 'rally';
  }
  renderTimeline();
  renderWaveform();
  refreshPlayhead();
}

function updateTimelineScopeUi(range = timelineRange()) {
  const btn = document.getElementById('act-timeline-scope');
  if (btn) {
    const hasSelectedRally = Boolean(currentRally());
    const effectiveScope = hasSelectedRally ? timelineScope : 'video';
    btn.textContent = effectiveScope === 'rally' ? 'Current rally' : 'Full video';
    btn.style.minWidth = '7rem';
    btn.title = effectiveScope === 'rally'
      ? 'Timeline range: current rally'
      : 'Timeline range: full video';
    btn.setAttribute('aria-pressed', String(effectiveScope === 'rally'));
    btn.disabled = !state.video || !state.numFrames || !hasSelectedRally;
    btn.classList.toggle('ring-2', effectiveScope === 'rally');
    btn.classList.toggle('ring-emerald-400/40', effectiveScope === 'rally');
    btn.style.background = effectiveScope === 'rally' ? 'rgba(34, 197, 94, 0.14)' : '';
    btn.style.borderColor = effectiveScope === 'rally' ? 'rgba(34, 197, 94, 0.32)' : '';
    btn.style.color = effectiveScope === 'rally' ? '#86EFAC' : '';
  }

  const label = document.getElementById('act-timeline-range');
  if (!label) return;
  if (!state.video || !state.numFrames) {
    label.textContent = '';
    return;
  }
  if (range.scope === 'rally' && range.rally) {
    const idx = selectedRallyIndex();
    label.textContent = `R${idx + 1} ${formatSeconds(range.startTime)}-${formatSeconds(range.endTime)}`;
  } else {
    label.textContent = `Video ${formatSeconds(range.startTime)}-${formatSeconds(range.endTime)}`;
  }
}

async function loadWaveform(videoName) {
  const requestId = ++waveformRequestId;
  waveform = { video: videoName, loading: true, error: '', hasAudio: false, duration: 0, peaks: [], rms: [] };
  renderWaveform();
  try {
    const points = waveformPointCount(state.duration);
    const data = await api(`${API.actionAnnotate.waveform(videoName)}?points=${points}`);
    if (requestId !== waveformRequestId || videoName !== state.video) return;
    const peaks = Array.isArray(data.peaks) ? data.peaks.map(v => clamp(Number(v) || 0, 0, 1)) : [];
    const rms = Array.isArray(data.rms) && data.rms.length === peaks.length
      ? data.rms.map(v => clamp(Number(v) || 0, 0, 1))
      : peaks;
    waveform = {
      video: videoName,
      loading: false,
      error: '',
      hasAudio: Boolean(data.has_audio),
      duration: Number(data.duration) || state.duration || 0,
      peaks,
      rms,
    };
  } catch (e) {
    if (requestId !== waveformRequestId || videoName !== state.video) return;
    waveform = { video: videoName, loading: false, error: e.message, hasAudio: false, duration: 0, peaks: [], rms: [] };
  }
  renderWaveform();
}

function waveformPointCount(durationSeconds) {
  const duration = Math.max(0, Number(durationSeconds) || 0);
  const points = Math.ceil(duration * WAVEFORM_POINTS_PER_SECOND);
  return clamp(points || WAVEFORM_MIN_POINTS, WAVEFORM_MIN_POINTS, WAVEFORM_MAX_POINTS);
}

function renderWaveform() {
  const canvas = document.getElementById('act-waveform');
  const status = document.getElementById('act-waveform-status');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.floor(width * dpr));
  const pixelHeight = Math.max(1, Math.floor(height * dpr));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, 'rgba(56, 189, 248, 0.12)');
  gradient.addColorStop(1, 'rgba(249, 115, 22, 0.06)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  if (status) {
    if (!state.video) status.textContent = '';
    else if (waveform.loading) status.textContent = 'Loading audio...';
    else if (waveform.error) status.textContent = 'Audio unavailable';
    else if (!waveform.hasAudio || !waveform.peaks.length) status.textContent = 'No audio';
    else status.textContent = 'Audio';
  }
  if (!state.video || waveform.loading || !waveform.hasAudio || !waveform.peaks.length) {
    updateTimelinePlayhead();
    return;
  }

  const range = timelineRange();
  const duration = waveform.duration || state.duration || 1;
  const startIndex = clamp(Math.floor((range.startTime / duration) * waveform.peaks.length), 0, waveform.peaks.length - 1);
  const endIndex = clamp(Math.ceil((range.endTime / duration) * waveform.peaks.length), startIndex + 1, waveform.peaks.length);
  const visiblePeaks = waveform.peaks.slice(startIndex, endIndex);
  const visibleRms = waveform.rms.length ? waveform.rms.slice(startIndex, endIndex) : visiblePeaks;
  const scaleAmp = waveformScaleAmp(visibleRms);
  const center = height / 2;
  const usableHeight = Math.max(8, height - 10);
  const halfHeight = usableHeight * WAVEFORM_VERTICAL_FILL;

  const valueAtPixel = (values, x) => {
    if (!values.length) return 0;
    if (values.length < width * 1.5) {
      const pos = (x / Math.max(1, width - 1)) * Math.max(0, values.length - 1);
      const left = Math.floor(pos);
      const right = Math.min(values.length - 1, left + 1);
      const mix = pos - left;
      return (values[left] || 0) * (1 - mix) + (values[right] || 0) * mix;
    }

    const from = Math.floor((x / width) * values.length);
    const to = Math.max(from + 1, Math.ceil(((x + 1) / width) * values.length));
    let value = 0;
    for (let i = from; i < to; i += 1) {
      value = Math.max(value, values[i] || 0);
    }
    return value;
  };

  const compressedAmp = (value, multiplier = 1) => Math.sqrt(clamp((value * multiplier) / scaleAmp, 0, 1));

  ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
  ctx.beginPath();
  ctx.moveTo(0, center);
  for (let x = 0; x < width; x += 1) {
    const amp = compressedAmp(valueAtPixel(visiblePeaks, x), WAVEFORM_PEAK_GAIN) * halfHeight;
    ctx.lineTo(x, center - amp);
  }
  for (let x = width - 1; x >= 0; x -= 1) {
    const amp = compressedAmp(valueAtPixel(visiblePeaks, x), WAVEFORM_PEAK_GAIN) * halfHeight;
    ctx.lineTo(x, center + amp);
  }
  ctx.closePath();
  ctx.fill();

  const rmsGradient = ctx.createLinearGradient(0, 0, width, 0);
  rmsGradient.addColorStop(0, 'rgba(56, 189, 248, 0.72)');
  rmsGradient.addColorStop(0.58, 'rgba(129, 140, 248, 0.76)');
  rmsGradient.addColorStop(1, 'rgba(249, 115, 22, 0.68)');
  ctx.fillStyle = rmsGradient;
  ctx.beginPath();
  ctx.moveTo(0, center);
  for (let x = 0; x < width; x += 1) {
    const amp = compressedAmp(valueAtPixel(visibleRms, x), WAVEFORM_RMS_GAIN) * halfHeight;
    ctx.lineTo(x, center - amp);
  }
  for (let x = width - 1; x >= 0; x -= 1) {
    const amp = compressedAmp(valueAtPixel(visibleRms, x), WAVEFORM_RMS_GAIN) * halfHeight;
    ctx.lineTo(x, center + amp);
  }
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, center);
  ctx.lineTo(width, center);
  ctx.stroke();
  updateTimelinePlayhead();
}

function waveformScaleAmp(values) {
  const active = values.filter(v => v > 0.001).sort((a, b) => a - b);
  if (!active.length) return 0.03;
  const idx = clamp(Math.floor((active.length - 1) * WAVEFORM_SCALE_PERCENTILE), 0, active.length - 1);
  return Math.max(WAVEFORM_MIN_SCALE, active[idx] * WAVEFORM_SCALE_HEADROOM);
}

function renderTimeline() {
  const markers = document.getElementById('act-timeline-markers');
  if (!markers) return;
  const range = timelineRange();
  updateTimelineScopeUi(range);
  if (!state.video || !state.numFrames) {
    markers.innerHTML = '';
    updateTimelinePlayhead();
    return;
  }
  const rallyBands = range.scope === 'rally' && range.rally
    ? `<div class="absolute top-0 bottom-0 rounded-sm bg-emerald-400/[0.18] border-x border-emerald-300/50"
        title="R${selectedRallyIndex() + 1} ${formatSeconds(range.startTime)}-${formatSeconds(range.endTime)}"
        style="left:0%; width:100%"></div>`
    : state.rallies.map((rally, idx) => {
      const startFrame = Math.round(rally.start * state.fps);
      const endFrame = Math.round(rally.end * state.fps);
      const startPct = frameToTimelinePct(startFrame, range);
      const endPct = frameToTimelinePct(endFrame, range);
      const active = rally.rally_id === selectedRallyId;
      return `<div class="absolute top-0 bottom-0 rounded-sm ${active ? 'bg-emerald-400/[0.18] border-x border-emerald-300/50' : 'bg-emerald-500/[0.08] border-x border-emerald-400/[0.15]'}"
        title="R${idx + 1} ${formatSeconds(rally.start)}-${formatSeconds(rally.end)}"
        style="left:${startPct}%; width:${Math.max(0.2, endPct - startPct)}%"></div>`;
    }).join('');
  const eventButtons = visibleEventEntries().filter(({ event }) => eventInTimelineRange(event, range)).map(({ event, idx }) => {
    const pct = frameToTimelinePct(event.frame, range);
    const color = COLORS[event.label] || '#818CF8';
    const active = idx === selectedIdx;
    const visible = event.visible !== false;
    return `<button type="button" data-idx="${idx}" title="${escapeHtml(event.label)} frame ${event.frame}"
      class="absolute top-1/2 -translate-y-1/2 rounded-full border border-black/50 transition-transform ${active ? 'w-3 h-5 -ml-1.5 scale-110' : 'w-1.5 h-4 -ml-px hover:scale-125'}"
      style="left:${pct}%; background:${visible ? color : 'transparent'}; border-color:${visible ? 'rgba(0,0,0,0.5)' : color}"></button>`;
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
  const pct = state.video ? frameToTimelinePct(currentFrame(), timelineRange()) : 0;
  const timelinePlayhead = document.getElementById('act-timeline-playhead');
  const waveformPlayhead = document.getElementById('act-waveform-playhead');
  if (timelinePlayhead) timelinePlayhead.style.left = `${pct}%`;
  if (waveformPlayhead) waveformPlayhead.style.left = `${pct}%`;
}

function hasVideoFrameClock() {
  return Boolean(videoEl?.requestVideoFrameCallback);
}

function currentMediaTime() {
  if (lockedFrame !== null && state.fps) {
    return lockedFrame / state.fps;
  }
  const t = hasVideoFrameClock() && Number.isFinite(presentedMediaTime)
    ? presentedMediaTime
    : (videoEl?.currentTime || 0);
  if (state.duration > 0) return clamp(t, 0, state.duration);
  return Math.max(0, t);
}

function invalidateVideoFrameClock({ clearPresented = false } = {}) {
  frameClockGeneration += 1;
  if (videoFrameCallbackId !== null && videoEl?.cancelVideoFrameCallback) {
    videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
  }
  videoFrameCallbackId = null;
  if (clearPresented) presentedMediaTime = null;
}

function restartVideoFrameClock({ clearPresented = false } = {}) {
  invalidateVideoFrameClock({ clearPresented });
  startVideoFrameClock();
}

function startVideoFrameClock() {
  if (!hasVideoFrameClock()) {
    presentedMediaTime = null;
    return;
  }
  if (videoFrameCallbackId !== null) return;
  const callbackGeneration = frameClockGeneration;
  let callbackId = null;
  callbackId = videoEl.requestVideoFrameCallback((_now, metadata) => {
    if (videoFrameCallbackId === callbackId) {
      videoFrameCallbackId = null;
    }
    if (callbackGeneration !== frameClockGeneration) {
      if (videoFrameCallbackId === null) startVideoFrameClock();
      return;
    }
    if (!videoEl.paused) {
      lockedFrame = null;
    }
    if (Number.isFinite(metadata?.mediaTime)) {
      if (lockedFrame === null || !videoEl.paused) {
        presentedMediaTime = metadata.mediaTime;
      }
    }
    refreshPlayhead();
    startVideoFrameClock();
  });
  videoFrameCallbackId = callbackId;
}

function stopVideoFrameClock() {
  invalidateVideoFrameClock({ clearPresented: true });
  lockedFrame = null;
}

function onVideoTimeUpdate() {
  if (videoEl && !videoEl.paused) {
    lockedFrame = null;
  }
  refreshPlayhead();
}

function onTimelineClick(e) {
  if (!state.video || !state.numFrames) return;
  if (e.target.closest('button[data-idx]')) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const pct = clamp((e.clientX - rect.left) / rect.width, 0, 1);
  seekFrame(timelinePctToFrame(pct, timelineRange()));
}

function refreshPlayhead() {
  if (!videoEl) return;
  const t = currentMediaTime();
  const frame = currentFrame();
  document.getElementById('act-time').textContent = `${formatSeconds(t)} / f${frame}`;
  document.getElementById('act-meta').textContent = state.video
    ? `${state.fps.toFixed(3)} fps · ${state.numFrames} frames`
    : '';
  if (frame !== lastOverlayFrame) {
    lastOverlayFrame = frame;
    renderOverlay();
  }
  updateTimelinePlayhead();
  autoPauseAtRallyEnd(t);
}

function autoPauseAtRallyEnd(t) {
  if (!videoEl || videoEl.paused || selectedRallyId === 'all') return;
  const rally = currentRally();
  if (!rally || !Number.isFinite(rally.end)) return;
  if (t < rally.end) return;

  const endTime = Math.min(rally.end, videoEl.duration || rally.end);
  presentedMediaTime = endTime;
  videoEl.pause();
  videoEl.currentTime = endTime;
  startVideoFrameClock();
  updatePlaybackButton();
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
  if (lockedFrame !== null) return clampFrame(lockedFrame);
  return clampFrame(Math.round(currentMediaTime() * state.fps));
}

function seekFrame(frame) {
  if (!videoEl || !state.fps) return;
  const targetFrame = clampFrame(frame);
  // Seek to the middle of the frame's display interval, not its leading edge.
  // currentTime = N/fps lands exactly on the N-1 / N boundary, which different
  // decoders resolve to either side (off-by-one vs the 0-based ffmpeg cache).
  // (N + 0.5)/fps lands unambiguously inside frame N on every browser.
  const seekTime = (targetFrame + 0.5) / state.fps;
  const targetTime = state.duration > 0
    ? clamp(seekTime, 0, state.duration)
    : Math.max(0, seekTime);
  lockedFrame = targetFrame;
  // Report the frame's nominal time (N/fps) for the rounding-based readback,
  // even though we seek to the interval midpoint above.
  presentedMediaTime = targetFrame / state.fps;
  invalidateVideoFrameClock();
  videoEl.currentTime = targetTime;
  startVideoFrameClock();
  refreshPlayhead();
  requestAnimationFrame(refreshPlayhead);
}

function jumpToEvent(idx, { renderList = true } = {}) {
  const event = state.events[idx];
  if (!event) return;
  selectedIdx = idx;
  if (event.rally_id) {
    selectedRallyId = event.rally_id;
    setExpandedRally(event.rally_id);
  } else {
    selectedRallyId = 'all';
    setExpandedRally(OUTSIDE_GROUP_ID);
  }
  videoEl?.pause();
  seekFrame(event.frame);
  updatePlaybackButton();
  if (renderList) {
    renderEvents();
  } else {
    renderOverlay();
    renderTimeline();
  }
}

function stepFrame(delta) {
  if (!state.video || !state.fps) return;
  videoEl.pause();
  const baseFrame = lockedFrame !== null ? lockedFrame : currentFrame();
  seekFrame(baseFrame + delta);
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
  } else if (e.key.toLowerCase() === 'o') {
    e.preventDefault();
    toggleTimelineScope();
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

function normalizeRallyId(value) {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

function findRallyForFrame(frame, context = state) {
  const fps = Number(context.fps) || 30;
  const time = frame / fps;
  return (context.rallies || []).find(r => time >= r.start && time < r.end) || null;
}

function rallyIndex(id) {
  const parsedId = normalizeRallyId(id);
  if (!parsedId) return -1;
  return (state.rallies || []).findIndex(r => r.rally_id === parsedId);
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
    rally_id: rally?.rally_id || null,
    relative_frame: rally ? Math.max(0, Math.round((time - rally.start) * fps)) : null,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round4(value) {
  return Math.round(value * 10000) / 10000;
}

function clampFrame(frame) {
  const maxFrame = Math.max(0, state.numFrames - 1);
  const n = Math.round(Number(frame));
  return clamp(Number.isFinite(n) ? n : 0, 0, maxFrame);
}
