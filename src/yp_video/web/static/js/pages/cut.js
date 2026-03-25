/**
 * Cut page — Video segment cutter.
 */
import { api, formatTimePrecise, card, pageHeader, sectionTitle, btnPrimary, btnSmall, selectCls, inputCls, showToast, emptyState, kbdHint } from '../shared.js';

let state = { videos: [], segments: [], markStart: null };
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
          ${sectionTitle('Segments', '', btnPrimary('Export All', 'id="cut-export"'))}

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

  videoEl.addEventListener('timeupdate', () => {
    document.getElementById('cut-time').textContent = formatTimePrecise(videoEl.currentTime);
  });
}

function handleKeydown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  switch (e.key) {
    case '[': e.preventDefault(); markStart(); break;
    case ']': e.preventDefault(); markEnd(); break;
    case 'ArrowLeft': e.preventDefault(); videoEl.currentTime = Math.max(0, videoEl.currentTime - 5); break;
    case 'ArrowRight': e.preventDefault(); videoEl.currentTime += 5; break;
  }
}

async function loadVideos() {
  try {
    const videos = await api('/cut/videos');
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

  const idx = state.segments.length + 1;
  state.segments.push({
    name: `set${idx}`,
    start: state.markStart,
    end: end,
  });

  state.markStart = null;
  document.getElementById('cut-mark-info').classList.add('hidden');
  renderSegments();
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

  const videoStem = (document.getElementById('cut-video-select').value || '').replace(/\.[^.]+$/, '');

  el.innerHTML = state.segments.map((s, i) => `
    <div class="flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-white/[0.03] border border-white/5 group hover:bg-white/[0.06] hover:border-white/[0.08] transition-all duration-200">
      <span class="w-6 h-6 rounded-md bg-surface-200 border border-border flex items-center justify-center text-[10px] font-heading text-text-muted flex-shrink-0">${i + 1}</span>
      <input type="text" value="${videoStem}_${s.name}" data-idx="${i}" class="seg-name ${inputCls} !bg-transparent !border-transparent !px-0 !py-0 !rounded-none !ring-0 focus:!border-b focus:!border-primary-light flex-1 min-w-0 font-heading text-sm transition-colors duration-200">
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
      state.segments[parseInt(e.target.dataset.idx)].name = e.target.value;
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
    const res = await api('/cut/export', {
      method: 'POST',
      body: { source, segments: state.segments },
    });
    showToast(`Exported ${res.success.length} segments${res.failed.length ? `, ${res.failed.length} failed` : ''}`, res.failed.length ? 'warning' : 'success');
  } catch (e) {
    showToast(`Export failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Export All';
  }
}
