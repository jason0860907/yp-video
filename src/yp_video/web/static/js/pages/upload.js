/**
 * Upload page — R2 cloud storage sync.
 */
import { api, SSEClient, formatBytes, card, pageHeader, sectionTitle, btnPrimary, btnSecondary, btnSmall, btnDanger, createProgressBar, showToast, emptyState } from '../shared.js';

let sseClients = [];

const CATEGORIES = [
  { key: 'videos', label: 'Videos', localOnly: true },
  { key: 'cuts', label: 'Cuts' },
  { key: 'seg-annotations', label: 'Seg Annotations' },
  { key: 'rally-pre-annotations', label: 'VLM-Predictions' },
  { key: 'tad-predictions', label: 'TAD-Predictions' },
  { key: 'tad-features', label: 'TAD-Features' },
  { key: 'tad-checkpoints', label: 'TAD-Checkpoints' },
  { key: 'rally-annotations', label: 'Annotations' },
  { key: 'rally_clips', label: 'Rally Clips' },
];

function isLocalOnly() {
  return CATEGORIES.find(c => c.key === state.category)?.localOnly || false;
}

let state = {
  configured: false,
  bucket: null,
  category: 'cuts',
  mode: 'upload', // 'upload' or 'download'
  files: [],
  jobs: [],
};

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Cloud Storage', 'Sync files with Cloudflare R2')}

      <!-- R2 Status -->
      <div id="upl-status"></div>

      <!-- Mode Toggle -->
      ${card(`
        <div class="flex items-center gap-3">
          <button id="upl-mode-upload" class="upl-mode-btn px-4 py-2 rounded-xl text-sm font-medium cursor-pointer transition-all duration-200">Upload</button>
          <button id="upl-mode-download" class="upl-mode-btn px-4 py-2 rounded-xl text-sm font-medium cursor-pointer transition-all duration-200">Download from R2</button>
        </div>
      `)}

      <!-- Category Tabs -->
      ${card(`
        <div class="space-y-4">
          ${sectionTitle('Category')}
          <div id="upl-tabs" class="flex flex-wrap gap-2"></div>
        </div>
      `)}

      <!-- File List -->
      ${card(`
        <div class="space-y-4">
          <div id="upl-files-header"></div>
          <div id="upl-files" class="space-y-0.5 max-h-[32rem] overflow-y-auto pr-1"></div>
          <div id="upl-actions" class="flex items-center gap-3 pt-2 border-t border-border"></div>
        </div>
      `)}

      <!-- Progress -->
      <div id="upl-progress" class="hidden">
        ${card(`
          <div class="space-y-3">
            <div class="flex items-center gap-2.5 mb-3">
              <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse-dot"></span>
              <h3 class="text-sm font-heading font-semibold text-text-primary">Progress</h3>
            </div>
            <div id="upl-jobs-progress" class="space-y-3"></div>
          </div>
        `)}
      </div>
    </div>`;

  bindEvents();
  loadStatus();
}

export function activate() { loadStatus(); }

export function deactivate() {
  sseClients.forEach(c => c.stop());
  sseClients = [];
}

function bindEvents() {
  document.getElementById('upl-mode-upload').addEventListener('click', () => setMode('upload'));
  document.getElementById('upl-mode-download').addEventListener('click', () => setMode('download'));
}

function setMode(mode) {
  state.mode = mode;
  renderModeTabs();
  loadFiles();
}

function updateModeVisibility() {
  const modeCard = document.getElementById('upl-mode-upload')?.closest('.glass-card');
  if (modeCard) {
    modeCard.style.display = isLocalOnly() ? 'none' : '';
  }
}

function renderModeTabs() {
  document.querySelectorAll('.upl-mode-btn').forEach(btn => {
    const isActive = btn.id === `upl-mode-${state.mode}`;
    btn.className = `upl-mode-btn px-4 py-2 rounded-xl text-sm font-medium cursor-pointer transition-all duration-200 ${
      isActive
        ? 'bg-primary/15 text-primary-light border border-primary/20'
        : 'bg-white/[0.04] text-text-muted border border-transparent hover:bg-white/[0.08] hover:text-text-secondary'
    }`;
  });
}

async function loadStatus() {
  try {
    const res = await api('/upload/status');
    state.configured = res.configured;
    state.bucket = res.bucket;
    renderStatus();
    renderModeTabs();
    renderCategoryTabs();
    if (state.configured) loadFiles();
  } catch (e) {
    showToast(`Failed to check R2 status: ${e.message}`, 'error');
  }
}

function renderStatus() {
  const el = document.getElementById('upl-status');
  if (state.configured) {
    el.innerHTML = card(`
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-xl bg-emerald-500/[0.08] border border-emerald-500/15 flex items-center justify-center">
          <svg class="w-4.5 h-4.5 text-emerald-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15a4.5 4.5 0 004.5 4.5H18a3.75 3.75 0 001.332-7.257 3 3 0 00-3.758-3.848 5.25 5.25 0 00-10.233 2.33A4.502 4.502 0 002.25 15z"/></svg>
        </div>
        <div>
          <h3 class="text-sm font-heading font-semibold text-text-primary">R2 Connected</h3>
          <span class="text-[11px] text-text-muted">Bucket: ${state.bucket}</span>
        </div>
        <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20 ml-auto">
          <span class="w-1.5 h-1.5 rounded-full bg-current"></span>configured
        </span>
      </div>
    `);
  } else {
    el.innerHTML = card(`
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-xl bg-amber-500/[0.08] border border-amber-500/15 flex items-center justify-center">
          <svg class="w-4.5 h-4.5 text-amber-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/></svg>
        </div>
        <div>
          <h3 class="text-sm font-heading font-semibold text-text-primary">R2 Not Configured</h3>
          <span class="text-[11px] text-text-muted">Fill in <code class="bg-surface-200 px-1.5 py-0.5 rounded text-text-secondary">r2.env</code> with your Cloudflare R2 credentials</span>
        </div>
      </div>
    `);
  }
}

function renderCategoryTabs() {
  const el = document.getElementById('upl-tabs');
  el.innerHTML = CATEGORIES.map(cat => {
    const isActive = cat.key === state.category;
    return `<button class="upl-cat-btn px-3.5 py-2 rounded-xl text-xs font-medium cursor-pointer transition-all duration-200 ${
      isActive
        ? 'bg-primary/15 text-primary-light border border-primary/20'
        : 'bg-white/[0.04] text-text-muted border border-transparent hover:bg-white/[0.08] hover:text-text-secondary'
    }" data-cat="${cat.key}">${cat.label}</button>`;
  }).join('');

  el.querySelectorAll('.upl-cat-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      state.category = btn.dataset.cat;
      if (isLocalOnly()) state.mode = 'upload';
      renderCategoryTabs();
      renderModeTabs();
      updateModeVisibility();
      loadFiles();
    });
  });
}

async function loadFiles() {
  if (!state.configured) return;

  const el = document.getElementById('upl-files');
  el.innerHTML = '<div class="py-6 text-center text-text-muted text-xs">Loading...</div>';

  try {
    if (state.mode === 'upload') {
      state.files = await api(`/upload/files?category=${state.category}`);
    } else {
      state.files = await api(`/upload/r2-files?category=${state.category}`);
    }
    state.files.forEach(f => f.selected = false);
    renderFiles();
  } catch (e) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
      'Failed to load files',
      e.message
    );
  }
}

function renderFiles() {
  const el = document.getElementById('upl-files');
  const headerEl = document.getElementById('upl-files-header');
  const actionsEl = document.getElementById('upl-actions');

  if (state.files.length === 0) {
    headerEl.innerHTML = sectionTitle('Files');
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 13.5h3.86a2.25 2.25 0 012.012 1.244l.256.512a2.25 2.25 0 002.013 1.244h3.218a2.25 2.25 0 002.013-1.244l.256-.512a2.25 2.25 0 012.013-1.244h3.859M12 3v8.25m0 0l-3-3m3 3l3-3"/></svg>',
      state.mode === 'upload' ? 'No local files' : 'No files on R2',
      `No files found in ${state.category}`
    );
    actionsEl.innerHTML = '';
    return;
  }

  const isUpload = state.mode === 'upload';
  const localOnly = isLocalOnly();
  const syncLabel = isUpload ? 'uploaded' : 'local';
  const unsyncLabel = isUpload ? 'local only' : 'R2 only';

  headerEl.innerHTML = sectionTitle(
    'Files',
    `${state.files.length} files`,
    localOnly
      ? btnSmall('Select All', 'id="upl-select-all"')
      : `${btnSmall('Select All', 'id="upl-select-all"')}
         ${isUpload ? btnSmall('Uploaded', 'id="upl-select-uploaded"', 'success') : ''}
         ${btnSmall('Un-synced', 'id="upl-select-unsynced"', 'primary')}`
  );

  // Pre-count files per group for the group header
  const groupCounts = {};
  for (const f of state.files) {
    if (f.group) groupCounts[f.group] = (groupCounts[f.group] || 0) + 1;
  }

  let currentGroup = null;
  let html = '';

  for (let i = 0; i < state.files.length; i++) {
    const f = state.files[i];
    const isSynced = isUpload ? f.uploaded : f.local;

    // Group header (shown for nested categories like rally_clips, tad-features)
    if (f.group && f.group !== currentGroup) {
      currentGroup = f.group;
      html += `<div class="pt-3 pb-1 px-2.5 text-[11px] font-heading font-medium text-text-muted uppercase tracking-wider flex items-center gap-2">
        <span>${f.group}</span>
        <span class="text-text-muted/70 normal-case tracking-normal tabular-nums">(${groupCounts[f.group]} files)</span>
      </div>`;
    }

    const syncBadge = localOnly ? '' : (isSynced
      ? `<span class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>${syncLabel}</span>`
      : `<span class="inline-flex items-center gap-1.5 text-[11px] text-text-muted bg-white/5 ring-1 ring-white/10 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>${unsyncLabel}</span>`
    );

    html += `
      <div class="group flex items-center gap-3 p-2.5 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200">
        <input type="checkbox" data-idx="${i}" class="upl-check cursor-pointer accent-primary w-3.5 h-3.5" ${f.selected ? 'checked' : ''}>
        <span class="text-sm text-text-primary flex-1 truncate group-hover:text-white transition-colors duration-200">${f.name}</span>
        <span class="text-[11px] text-text-muted tabular-nums">${formatBytes(f.size)}</span>
        ${syncBadge}
      </div>`;
  }

  el.innerHTML = html;

  el.querySelectorAll('.upl-check').forEach(cb => {
    cb.addEventListener('change', (e) => {
      state.files[parseInt(e.target.dataset.idx)].selected = e.target.checked;
      updateSelectedCount();
    });
  });

  // Select buttons
  document.getElementById('upl-select-all').addEventListener('click', () => {
    state.files.forEach(f => f.selected = true);
    renderFiles();
  });
  const uploadedBtn = document.getElementById('upl-select-uploaded');
  if (uploadedBtn) {
    uploadedBtn.addEventListener('click', () => {
      state.files.forEach(f => f.selected = !!f.uploaded);
      renderFiles();
    });
  }
  const unsyncedBtn = document.getElementById('upl-select-unsynced');
  if (unsyncedBtn) {
    unsyncedBtn.addEventListener('click', () => {
      state.files.forEach(f => {
        f.selected = isUpload ? !f.uploaded : !f.local;
      });
      renderFiles();
    });
  }

  // Action buttons
  if (localOnly) {
    actionsEl.innerHTML = `
      ${btnDanger('Delete Local', 'id="upl-delete-local"')}
      <span id="upl-count" class="text-xs text-text-muted font-heading tabular-nums ml-auto"></span>
    `;
    document.getElementById('upl-delete-local').addEventListener('click', deleteLocal);
  } else if (isUpload) {
    actionsEl.innerHTML = `
      ${btnPrimary('Upload Selected', 'id="upl-start"')}
      ${btnDanger('Delete Local', 'id="upl-delete-local"')}
      <span id="upl-count" class="text-xs text-text-muted font-heading tabular-nums ml-auto"></span>
    `;
    document.getElementById('upl-start').addEventListener('click', startUpload);
    document.getElementById('upl-delete-local').addEventListener('click', deleteLocal);
  } else {
    actionsEl.innerHTML = `
      ${btnSecondary('Download Selected', 'id="upl-start-dl"')}
      ${btnDanger('Delete Selected on R2', 'id="upl-delete-r2"')}
      <span id="upl-count" class="text-xs text-text-muted font-heading tabular-nums ml-auto"></span>
    `;
    document.getElementById('upl-start-dl').addEventListener('click', startDownload);
    document.getElementById('upl-delete-r2').addEventListener('click', deleteR2);
  }

  updateSelectedCount();
}

function updateSelectedCount() {
  const count = state.files.filter(f => f.selected).length;
  const totalSize = state.files.filter(f => f.selected).reduce((s, f) => s + f.size, 0);
  const el = document.getElementById('upl-count');
  if (el) {
    el.textContent = count > 0 ? `${count} selected (${formatBytes(totalSize)})` : '';
  }
}

async function startUpload() {
  const selected = state.files.filter(f => f.selected).map(f => f.path);
  if (selected.length === 0) return showToast('No files selected', 'warning');

  const btn = document.getElementById('upl-start');
  btn.disabled = true;

  sseClients.forEach(c => c.stop());
  sseClients = [];

  try {
    const jobs = await api('/upload/start', {
      method: 'POST',
      body: { category: state.category, files: selected },
    });

    state.jobs = jobs.map(j => ({ ...j }));
    document.getElementById('upl-progress').classList.remove('hidden');
    renderJobsProgress();

    trackJobs(jobs, btn, () => loadFiles());
  } catch (e) {
    showToast(`Failed to start upload: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

async function startDownload() {
  const selected = state.files.filter(f => f.selected).map(f => f.path);
  if (selected.length === 0) return showToast('No files selected', 'warning');

  const btn = document.getElementById('upl-start-dl');
  btn.disabled = true;

  sseClients.forEach(c => c.stop());
  sseClients = [];

  try {
    const jobs = await api('/upload/download', {
      method: 'POST',
      body: { category: state.category, files: selected },
    });

    state.jobs = jobs.map(j => ({ ...j }));
    document.getElementById('upl-progress').classList.remove('hidden');
    renderJobsProgress();

    trackJobs(jobs, btn, () => loadFiles());
  } catch (e) {
    showToast(`Failed to start download: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

async function deleteR2() {
  const selected = state.files.filter(f => f.selected);
  if (selected.length === 0) return showToast('No files selected', 'warning');

  const msg = `Delete ${selected.length} selected files from R2?\n\n` +
              `This only removes them from cloud storage — local copies are not touched.`;
  if (!confirm(msg)) return;

  const btn = document.getElementById('upl-delete-r2');
  btn.disabled = true;
  try {
    const res = await api('/upload/delete-r2', {
      method: 'POST',
      body: { category: state.category, files: selected.map(f => f.path) },
    });
    showToast(`Deleted ${res.deleted} files from R2`, 'success');
    loadFiles();
  } catch (e) {
    showToast(`Delete failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

async function deleteLocal() {
  const selected = state.files.filter(f => f.selected);
  if (selected.length === 0) return showToast('No files selected', 'warning');

  const localOnly = isLocalOnly();
  const notUploaded = localOnly ? [] : selected.filter(f => !f.uploaded);

  let msg;
  if (localOnly) {
    msg = `Delete ${selected.length} local files?`;
  } else if (notUploaded.length === 0) {
    msg = `Delete ${selected.length} local files? They are already on R2.`;
  } else {
    msg = `Delete ${selected.length} local files?\n\n` +
          `⚠️ ${notUploaded.length} of them are NOT on R2 and will be lost permanently.`;
  }

  if (!confirm(msg)) return;

  // force=true when user has confirmed despite the R2-safety warning, so the
  // backend doesn't silently skip the not-on-R2 files.
  const force = localOnly || notUploaded.length > 0;

  try {
    const res = await api('/upload/delete-local', {
      method: 'POST',
      body: { category: state.category, files: selected.map(f => f.path), force },
    });
    const count = res.deleted?.length || 0;
    const skipped = res.skipped?.length || 0;
    if (count > 0) showToast(`Deleted ${count} local files${skipped ? `, ${skipped} skipped` : ''}`, 'success');
    else showToast('No files deleted', 'info');
    loadFiles();
  } catch (e) {
    showToast(`Delete failed: ${e.message}`, 'error');
  }
}

function trackJobs(jobs, btn, onDone) {
  let doneCount = 0;
  const total = jobs.length;

  for (const job of jobs) {
    const client = new SSEClient(`/api/jobs/${job.id}/events`, {
      onMessage: (data) => {
        const idx = state.jobs.findIndex(j => j.id === data.id);
        if (idx >= 0) state.jobs[idx] = data;
        renderJobsProgress();

        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
          client.stop();
          doneCount++;
          if (data.status === 'failed') {
            showToast(`${job.name} failed: ${data.error || 'Unknown error'}`, 'error');
          }
          if (doneCount >= total) {
            const failed = state.jobs.filter(j => j.status === 'failed').length;
            if (failed === 0) showToast('All transfers complete!', 'success');
            else showToast(`${total - failed}/${total} completed, ${failed} failed`, 'warning');
            btn.disabled = false;
            onDone();
          }
        }
      },
      onError: () => {
        doneCount++;
        if (doneCount >= total) btn.disabled = false;
      },
    }).start();
    sseClients.push(client);
  }
}

function renderJobsProgress() {
  const el = document.getElementById('upl-jobs-progress');
  if (!el) return;

  el.innerHTML = state.jobs.map(job => {
    const pct = Math.round((job.progress || 0) * 100);
    const isRunning = job.status === 'running';
    const isDone = job.status === 'completed';
    const isFailed = job.status === 'failed';
    const isCancelled = job.status === 'cancelled';

    let statusColor = 'text-text-muted';
    if (isRunning) statusColor = 'text-primary-light';
    else if (isDone) statusColor = 'text-emerald-400';
    else if (isFailed) statusColor = 'text-red-400';
    else if (isCancelled) statusColor = 'text-amber-400';

    return `
      <div class="space-y-1.5">
        <div class="flex items-center justify-between">
          <span class="text-xs text-text-primary font-medium truncate">${job.name}</span>
          <span class="text-[11px] ${statusColor} tabular-nums font-medium">${isDone ? 'done' : isFailed ? 'failed' : isCancelled ? 'cancelled' : pct + '%'}</span>
        </div>
        ${createProgressBar(job.progress)}
        ${job.error ? `<p class="text-[10px] text-red-400/80 truncate">${job.error}</p>` : ''}
      </div>`;
  }).join('');
}
