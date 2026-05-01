/**
 * Cut page — Video segment cutter.
 */
import { api, API, formatTimePrecise, card, pageHeader, sectionTitle, btnPrimary, btnDanger, btnSmall, selectCls, inputCls, showToast, showConfirm, emptyState, kbdHint } from '../shared.js';

let state = { videos: [], segments: [], markStart: null, kind: 'broadcast' };
let videoEl = null;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Cut', 'Mark and export video segments')}

      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            <label class="text-sm font-medium text-text-secondary flex-shrink-0">Video</label>
            <select id="cut-video-select" class="flex-1 ${selectCls}">
              <option value="">Select a video...</option>
            </select>
          </div>
          <div class="relative bg-black rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-lg shadow-black/20">
            <video id="cut-player" class="w-full max-h-[50vh]" controls></video>
          </div>
          <div class="flex items-center justify-between pt-1">
            <span id="cut-time" class="text-sm font-heading text-text-primary tabular-nums bg-surface-100 border border-border px-3 py-1.5 rounded-lg">00:00.000</span>
            <div class="flex items-center gap-2">
              ${btnSmall('Mark Start [', 'id="cut-mark-start"', 'primary')}
              ${btnSmall('Mark End ]', 'id="cut-mark-end"', 'primary')}
            </div>
          </div>
        </div>
      `)}

      ${card(`
        <div class="space-y-4">
          ${sectionTitle(
            'Segments',
            '',
            `<div class="flex items-center gap-2">
              <div class="inline-flex rounded-lg border border-border bg-surface-100 p-0.5" role="radiogroup" aria-label="Cut destination">
                <button type="button" data-kind="broadcast" class="cut-kind-btn px-2.5 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="true">Broadcast</button>
                <button type="button" data-kind="sideline" class="cut-kind-btn px-2.5 py-1 text-xs font-heading rounded-md transition-colors duration-150" aria-pressed="false">Sideline</button>
              </div>
              ${btnPrimary('Export All', 'id="cut-export"')}
            </div>`
          )}

          <div id="cut-mark-info" class="hidden">
            <div class="flex items-center gap-3 p-3 rounded-xl bg-primary/10 border border-primary/20">
              <span class="w-1.5 h-1.5 rounded-full bg-primary-light animate-pulse"></span>
              <span class="text-xs text-primary-light">Start at <strong id="cut-mark-time" class="font-heading"></strong> — press ] to set end</span>
            </div>
          </div>

          <div id="cut-segments" class="space-y-1.5">
            ${emptyState(
              '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M14.121 14.121A3 3 0 109.879 9.879m4.242 4.242L9.879 9.879"/></svg>',
              'No segments yet',
              'Use [ and ] keys to mark start/end points'
            )}
          </div>
        </div>
      `)}

      ${card(`
        <div class="flex items-center justify-between gap-4">
          <div class="min-w-0">
            <p class="text-sm font-heading font-medium text-text-primary">Delete source video</p>
            <p class="text-[11px] text-text-muted mt-0.5">Permanently removes the raw file from local storage. Cut segments are kept.</p>
          </div>
          ${btnDanger('Delete Video', 'id="cut-delete-video"')}
        </div>
      `)}

      ${kbdHint([['[  ]', 'mark'], ['\u2190 \u2192', 'skip 5s']])}
    </div>`;

  videoEl = document.getElementById('cut-player');
  loadVideos();
  bindEvents();
  activate();
}

export function activate() {
  document.addEventListener('keydown', handleKeydown);
}

export function deactivate() {
  document.removeEventListener('keydown', handleKeydown);
  if (videoEl && !videoEl.paused) videoEl.pause();
}

function bindEvents() {
  document.getElementById('cut-video-select').addEventListener('change', onVideoChange);
  document.getElementById('cut-mark-start').addEventListener('click', markStart);
  document.getElementById('cut-mark-end').addEventListener('click', markEnd);
  document.getElementById('cut-export').addEventListener('click', exportAll);
  document.getElementById('cut-delete-video').addEventListener('click', () => deleteSourceVideo());
  document.querySelectorAll('.cut-kind-btn').forEach(btn => {
    btn.addEventListener('click', () => setKind(btn.dataset.kind));
  });
  setKind(state.kind);  // initial styling

  videoEl.addEventListener('timeupdate', () => {
    document.getElementById('cut-time').textContent = formatTimePrecise(videoEl.currentTime);
  });
}

function setKind(kind) {
  state.kind = kind;
  document.querySelectorAll('.cut-kind-btn').forEach(btn => {
    const active = btn.dataset.kind === kind;
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    btn.classList.toggle('bg-primary', active);
    btn.classList.toggle('text-white', active);
    btn.classList.toggle('text-text-secondary', !active);
    btn.classList.toggle('hover:bg-white/[0.04]', !active);
  });
}

async function deleteSourceVideo({ skipConfirm = false, contextMessage = '' } = {}) {
  const name = document.getElementById('cut-video-select').value;
  if (!name) {
    showToast('No video selected', 'warning');
    return false;
  }

  if (!skipConfirm) {
    const ok = await showConfirm({
      title: 'Delete source video?',
      body: `${contextMessage ? contextMessage + '\n\n' : ''}This permanently removes the raw file from local storage:\n${name}\n\nCut segments already exported are not affected.`,
      confirmText: 'Delete',
      cancelText: 'Keep',
      variant: 'danger',
    });
    if (!ok) return false;
  }

  try {
    await api(API.cut.video(name), { method: 'DELETE' });
    showToast(`Deleted ${name}`, 'success');

    // Clear player + remove from dropdown
    videoEl.pause();
    videoEl.removeAttribute('src');
    videoEl.load();
    const sel = document.getElementById('cut-video-select');
    const opt = sel.querySelector(`option[value="${CSS.escape(name)}"]`);
    if (opt) opt.remove();
    sel.value = '';
    state.videos = state.videos.filter(v => v !== name);
    state.segments = [];
    state.markStart = null;
    document.getElementById('cut-mark-info').classList.add('hidden');
    renderSegments();
    return true;
  } catch (e) {
    showToast(`Delete failed: ${e.message}`, 'error');
    return false;
  }
}

function handleKeydown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  switch (e.key) {
    case ' ': e.preventDefault(); videoEl.paused ? videoEl.play() : videoEl.pause(); break;
    case '[': e.preventDefault(); markStart(); break;
    case ']': e.preventDefault(); markEnd(); break;
    case 'ArrowLeft': e.preventDefault(); videoEl.currentTime = Math.max(0, videoEl.currentTime - 5); break;
    case 'ArrowRight': e.preventDefault(); videoEl.currentTime += 5; break;
  }
}

async function loadVideos() {
  try {
    const videos = await api(API.cut.videos);
    state.videos = videos;
    const sel = document.getElementById('cut-video-select');
    videos.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    });
  } catch (e) {
    showToast(`Failed to load videos: ${e.message}`, 'error');
  }
}

function onVideoChange(e) {
  const name = e.target.value;
  if (!name) return;
  state.segments = [];
  state.markStart = null;
  document.getElementById('cut-mark-info').classList.add('hidden');
  renderSegments();
  videoEl.src = `/api/cut/video/${encodeURIComponent(name)}`;
  videoEl.load();
}

function markStart() {
  if (!videoEl.src) return;
  state.markStart = videoEl.currentTime;
  document.getElementById('cut-mark-info').classList.remove('hidden');
  document.getElementById('cut-mark-time').textContent = formatTimePrecise(state.markStart);
}

function markEnd() {
  if (state.markStart == null || !videoEl.src) return;
  const end = videoEl.currentTime;
  if (end <= state.markStart) return showToast('End must be after start', 'warning');

  // _auto flags this segment as eligible for renumbering. As long as the user
  // doesn't edit the name input, renumberAutoNames() keeps it in sync with
  // the segment count: bare stem when there's exactly one set, `_set1/_set2/…`
  // once a second segment is added (and back to bare if it gets deleted).
  state.segments.push({
    name: '',  // filled in below by renumberAutoNames
    start: state.markStart,
    end: end,
    _auto: true,
  });
  renumberAutoNames();

  state.markStart = null;
  document.getElementById('cut-mark-info').classList.add('hidden');
  renderSegments();
}

function videoStem() {
  return (document.getElementById('cut-video-select').value || '').replace(/\.[^.]+$/, '');
}

function renumberAutoNames() {
  const stem = videoStem();
  const total = state.segments.length;
  state.segments.forEach((s, i) => {
    if (s._auto) {
      s.name = total === 1 ? stem : `${stem}_set${i + 1}`;
    }
  });
}

function renderSegments() {
  const el = document.getElementById('cut-segments');
  if (state.segments.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M14.121 14.121A3 3 0 109.879 9.879m4.242 4.242L9.879 9.879"/></svg>',
      'No segments yet',
      'Use [ and ] keys to mark start/end points'
    );
    return;
  }

  el.innerHTML = state.segments.map((s, i) => `
    <div class="flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-white/[0.03] border border-white/5 group hover:bg-white/[0.06] hover:border-white/[0.08] transition-all duration-200">
      <span class="w-6 h-6 rounded-md bg-surface-200 border border-border flex items-center justify-center text-[10px] font-heading text-text-muted flex-shrink-0">${i + 1}</span>
      <input type="text" value="${s.name}" data-idx="${i}" class="seg-name ${inputCls} !bg-transparent !border-transparent !px-0 !py-0 !rounded-none !ring-0 focus:!border-b focus:!border-primary-light flex-1 min-w-0 font-heading text-sm transition-colors duration-200">
      <div class="flex items-center gap-2 ml-auto">
        <span class="text-xs text-text-muted font-heading tabular-nums">${formatTimePrecise(s.start)}</span>
        <span class="text-text-muted/40">&rarr;</span>
        <span class="text-xs text-text-muted font-heading tabular-nums">${formatTimePrecise(s.end)}</span>
        <span class="text-[10px] text-text-muted bg-surface-200 border border-border px-1.5 py-0.5 rounded font-heading">${(s.end - s.start).toFixed(1)}s</span>
      </div>
      <div class="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity duration-200 ml-2">
        <button data-idx="${i}" class="seg-preview text-xs text-primary-light cursor-pointer hover:underline transition-colors duration-200">Preview</button>
        <button data-idx="${i}" class="seg-delete text-xs text-red-400 cursor-pointer hover:underline transition-colors duration-200">Delete</button>
      </div>
    </div>
  `).join('');

  el.querySelectorAll('.seg-name').forEach(inp => {
    inp.addEventListener('change', (e) => {
      const seg = state.segments[parseInt(e.target.dataset.idx)];
      seg.name = e.target.value;
      // User intent: opt out of auto-renumbering for this row.
      seg._auto = false;
    });
  });
  el.querySelectorAll('.seg-preview').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const s = state.segments[parseInt(e.target.dataset.idx)];
      videoEl.currentTime = s.start;
      videoEl.play();
    });
  });
  el.querySelectorAll('.seg-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      state.segments.splice(parseInt(e.target.dataset.idx), 1);
      renumberAutoNames();
      renderSegments();
    });
  });
}

async function exportAll() {
  if (state.segments.length === 0) return showToast('No segments to export', 'warning');
  const source = document.getElementById('cut-video-select').value;
  if (!source) return showToast('No video selected', 'warning');

  const btn = document.getElementById('cut-export');
  btn.disabled = true;
  btn.textContent = 'Exporting...';

  try {
    const res = await api(API.cut.export, {
      method: 'POST',
      body: { source, segments: state.segments, kind: state.kind },
    });
    showToast(`Exported ${res.success.length} segments${res.failed.length ? `, ${res.failed.length} failed` : ''}`, res.failed.length ? 'warning' : 'success');

    // Offer to delete the source video only if everything exported cleanly
    if (res.failed.length === 0 && res.success.length > 0) {
      await deleteSourceVideo({
        contextMessage: `All ${res.success.length} segments exported successfully.`,
      });
    }
  } catch (e) {
    showToast(`Export failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Export All';
  }
}
