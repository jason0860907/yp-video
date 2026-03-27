/**
 * Download page — YouTube playlist batch download.
 */
import { api, SSEClient, formatBytes, formatSpeed, formatDuration, card, pageHeader, sectionTitle, btnPrimary, btnDanger, btnSmall, selectInput, showToast, emptyState, inputCls, createProgressBar, createStatusBadge, kbdHint } from '../shared.js';

let sseClient = null;
let state = { videos: [], sessionId: null, downloading: false };

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Download', 'Download videos from YouTube playlists')}

      ${card(`
        <div class="space-y-4">
          ${sectionTitle('Playlist URL')}
          <div class="flex gap-3">
            <input id="dl-url" type="text" placeholder="https://www.youtube.com/playlist?list=..."
              class="flex-1 ${inputCls}">
            ${btnPrimary('Fetch Playlist', 'id="dl-fetch"')}
          </div>
          <div id="dl-status" class="text-sm text-text-muted hidden"></div>
        </div>
      `)}

      <div id="dl-playlist" class="hidden space-y-6">
        ${card(`
          <div class="space-y-4">
            ${sectionTitle(
              '<span id="dl-title"></span>',
              '',
              `<div class="flex items-center gap-3">
                <label class="text-[11px] text-text-muted uppercase tracking-wider">Quality</label>
                ${selectInput('dl-quality', [
                  { value: 'best', label: 'Best' },
                  { value: '1080', label: '1080p' },
                  { value: '720', label: '720p' },
                  { value: '480', label: '480p' },
                ])}
              </div>`
            )}
            <div class="flex items-center gap-2 text-xs">
              ${btnSmall('Select All', 'id="dl-select-all"', 'primary')}
              ${btnSmall('Deselect All', 'id="dl-deselect-all"')}
              <span id="dl-count" class="ml-auto text-text-muted font-heading tabular-nums"></span>
            </div>
            <div id="dl-videos" class="space-y-0.5 max-h-[28rem] overflow-y-auto pr-1"></div>
            <div class="flex items-center gap-3 pt-2 border-t border-border">
              <div id="dl-action-btns"></div>
            </div>
          </div>
        `)}
      </div>

      ${kbdHint([['Enter', 'fetch playlist']])}
    </div>`;

  document.getElementById('dl-fetch').addEventListener('click', fetchPlaylist);
  document.getElementById('dl-url').addEventListener('keydown', e => { if (e.key === 'Enter') fetchPlaylist(); });
  document.getElementById('dl-select-all').addEventListener('click', () => toggleAll(true));
  document.getElementById('dl-deselect-all').addEventListener('click', () => toggleAll(false));

}

export function activate() {
  // Reconnect SSE if connection dropped during a long download
  if (state.sessionId && (!sseClient || !sseClient.source)) {
    reconnectSSE(state.sessionId);
  }
}
export function deactivate() {}

async function fetchPlaylist() {
  const url = document.getElementById('dl-url').value.trim();
  if (!url) return;

  const btn = document.getElementById('dl-fetch');
  const status = document.getElementById('dl-status');
  btn.disabled = true;
  btn.textContent = 'Fetching...';
  status.classList.remove('hidden');
  status.textContent = 'Fetching playlist info...';

  try {
    const data = await api(`/download/playlist?url=${encodeURIComponent(url)}`);
    state.videos = data.videos.map(v => ({ ...v, selected: true, status: 'pending', progress: null }));

    document.getElementById('dl-title').textContent = data.title;
    document.getElementById('dl-playlist').classList.remove('hidden');
    status.classList.add('hidden');
    renderVideoList();
    renderActionBtns();
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    status.className = 'text-sm text-red-400';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Fetch Playlist';
  }
}

function renderVideoList() {
  const el = document.getElementById('dl-videos');
  const selected = state.videos.filter(v => v.selected).length;
  document.getElementById('dl-count').textContent = `${selected} / ${state.videos.length} selected`;

  el.innerHTML = state.videos.map((v, i) => {
    let progressHtml = '';
    if (v.progress) {
      const p = v.progress;
      const pct = p.percent ? `${p.percent.toFixed(1)}%` : '';
      const size = p.downloaded && p.total ? `${formatBytes(p.downloaded)}/${formatBytes(p.total)}` : '';
      const speed = p.speed ? formatSpeed(p.speed) : '';
      const eta = p.eta ? `ETA ${p.eta}s` : '';
      progressHtml = `
        <div class="mt-2.5">
          ${createProgressBar(p.percent / 100, v.status === 'completed' ? 'success' : 'primary')}
          <div class="flex gap-3 mt-1.5 text-[11px] text-text-muted font-heading tabular-nums">${[pct, size, speed, eta].filter(Boolean).join(' <span class="text-white/10">\u00b7</span> ')}</div>
        </div>`;
    }

    return `
      <div class="group flex items-start gap-3 p-3 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200 ${v.status === 'completed' ? 'opacity-50' : ''}">
        <input type="checkbox" data-idx="${i}" class="dl-check mt-0.5 cursor-pointer accent-primary w-3.5 h-3.5" ${v.selected ? 'checked' : ''} ${state.downloading ? 'disabled' : ''}>
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2">
            <p class="text-sm text-text-primary truncate group-hover:text-white transition-colors duration-200">${v.title}</p>
          </div>
          <div class="flex items-center gap-2.5 mt-1">
            ${v.duration ? `<span class="text-[11px] text-text-muted font-heading tabular-nums">${formatDuration(v.duration)}</span>` : ''}
            ${createStatusBadge(v.status)}
          </div>
          ${progressHtml}
        </div>
      </div>`;
  }).join('');

  el.querySelectorAll('.dl-check').forEach(cb => {
    cb.addEventListener('change', (e) => {
      state.videos[parseInt(e.target.dataset.idx)].selected = e.target.checked;
      renderVideoList();
    });
  });
}

function renderActionBtns() {
  const el = document.getElementById('dl-action-btns');
  if (state.downloading) {
    el.innerHTML = btnDanger('Cancel', 'id="dl-cancel"');
    document.getElementById('dl-cancel').addEventListener('click', cancelDownload);
  } else {
    el.innerHTML = btnPrimary('Download Selected', 'id="dl-start"');
    document.getElementById('dl-start').addEventListener('click', startDownload);
  }
}

function reconnectSSE(sessionId) {
  sseClient?.stop();
  sseClient = new SSEClient(`/api/download/${sessionId}/progress`, {
    onMessage: (data) => {
      if (data.video_id) {
        const v = state.videos.find(x => x.id === data.video_id);
        if (v) {
          v.status = data.status || v.status;
          v.progress = data;
        }
      }
      if (data.status === 'complete') {
        state.downloading = false;
        sseClient?.stop();
        showToast('Download complete!', 'success');
        renderActionBtns();
      }
      renderVideoList();
    },
    onError: () => {
      state.downloading = false;
      renderActionBtns();
    },
  }).start();
}

async function startDownload() {
  const selected = state.videos.filter(v => v.selected && v.status !== 'completed');
  if (selected.length === 0) return showToast('No videos selected', 'warning');

  const quality = document.getElementById('dl-quality').value;
  state.downloading = true;
  renderActionBtns();

  try {
    const res = await api('/download/start', {
      method: 'POST',
      body: { videos: selected.map(v => ({ id: v.id, title: v.title, duration: v.duration, url: v.url })), quality },
    });

    state.sessionId = res.session_id;
    reconnectSSE(res.session_id);
  } catch (e) {
    state.downloading = false;
    showToast(`Download failed: ${e.message}`, 'error');
    renderActionBtns();
  }
}

async function cancelDownload() {
  if (!state.sessionId) return;
  try {
    await api(`/download/${state.sessionId}/cancel`, { method: 'POST' });
    sseClient?.stop();
    state.downloading = false;
    showToast('Download cancelled', 'warning');
    renderActionBtns();
  } catch (e) {
    showToast(`Cancel failed: ${e.message}`, 'error');
  }
}

function toggleAll(val) {
  state.videos.forEach(v => v.selected = val);
  renderVideoList();
}
