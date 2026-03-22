/**
 * Jobs page — Background task monitoring + vLLM control.
 */
import { api, card, pageHeader, sectionTitle, btnPrimary, btnSecondary, btnSmall, btnDanger, createProgressBar, createStatusBadge, showToast, emptyState } from '../shared.js';

let pollTimer = null;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-6">
      ${pageHeader('Jobs & System', 'Monitor tasks and control vLLM server')}

      <!-- vLLM Control -->
      ${card(`
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-xl bg-indigo-500/[0.08] border border-indigo-500/15 flex items-center justify-center">
                <svg class="w-4.5 h-4.5 text-indigo-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7m0 0a3 3 0 01-3 3m0 3h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008zm-3 6h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008z"/></svg>
              </div>
              <div>
                <h3 class="text-sm font-heading font-semibold text-text-primary">vLLM Server</h3>
                <span id="jobs-vllm-info" class="text-[11px] text-text-muted leading-tight"></span>
              </div>
              <span id="jobs-vllm-badge" class="ml-1"></span>
            </div>
            <div class="flex gap-2" id="jobs-vllm-btns"></div>
          </div>
        </div>
      `)}

      <!-- Pipeline Stats -->
      <div id="jobs-stats-card"></div>

      <!-- Job List -->
      ${card(`
        <div class="space-y-4">
          ${sectionTitle('Background Jobs', '', btnSmall('Refresh', 'id="jobs-refresh"'))}
          <div id="jobs-list"></div>
        </div>
      `)}
    </div>`;

  document.getElementById('jobs-refresh').addEventListener('click', loadAll);
  loadAll();
  pollTimer = setInterval(loadAll, 5000);
}

export function destroy() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function loadAll() {
  await Promise.all([loadVLLM(), loadStats(), loadJobs()]);
}

async function loadVLLM() {
  try {
    const status = await api('/system/vllm/status');
    document.getElementById('jobs-vllm-badge').innerHTML = createStatusBadge(status.status || 'stopped');
    const btns = document.getElementById('jobs-vllm-btns');
    const info = document.getElementById('jobs-vllm-info');

    if (status.status === 'running') {
      btns.innerHTML = btnDanger('Stop', 'id="jobs-vllm-stop"');
      document.getElementById('jobs-vllm-stop').addEventListener('click', stopVLLM);
      info.textContent = `${status.model || '\u2014'} \u00b7 port ${status.port || '\u2014'}`;
    } else {
      btns.innerHTML = btnPrimary('Start', 'id="jobs-vllm-start"');
      document.getElementById('jobs-vllm-start').addEventListener('click', startVLLM);
      info.textContent = status.model ? `Model: ${status.model}` : 'Server stopped';
    }
  } catch { /* silently fail */ }
}

async function startVLLM() {
  try {
    showToast('Starting vLLM server...', 'info');
    await api('/system/vllm/start', { method: 'POST' });
    showToast('vLLM starting', 'success');
    setTimeout(loadVLLM, 3000);
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
  }
}

async function stopVLLM() {
  try {
    await api('/system/vllm/stop', { method: 'POST' });
    showToast('vLLM stopped', 'success');
    loadVLLM();
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
  }
}

async function loadStats() {
  try {
    const s = await api('/system/stats');
    document.getElementById('jobs-stats-card').innerHTML = card(`
      <div class="space-y-4">
        ${sectionTitle('Pipeline')}
        <div class="grid grid-cols-3 sm:grid-cols-6 gap-2.5">
          ${statBox('Videos', s.videos)}
          ${statBox('Cuts', s.cuts)}
          ${statBox('Detected', s.detections)}
          ${statBox('Pre-Ann', s.pre_annotations)}
          ${statBox('Annotated', s.annotations)}
          ${statBox('Predicted', s.predictions)}
        </div>
      </div>
    `);
  } catch { /* silently fail */ }
}

function statBox(label, value) {
  const hasValue = value > 0;
  return `
    <div class="rounded-xl px-3 py-3.5 text-center border transition-colors duration-200 ${hasValue
      ? 'border-indigo-500/15 bg-indigo-500/[0.04]'
      : 'border-border bg-surface-50/50'}">
      <div class="text-lg font-heading font-bold tabular-nums ${hasValue ? 'text-indigo-400' : 'text-text-muted'}">${value ?? 0}</div>
      <div class="text-[10px] text-text-muted mt-1 uppercase tracking-wider font-medium">${label}</div>
    </div>`;
}

async function loadJobs() {
  try {
    const jobs = await api('/jobs');
    const el = document.getElementById('jobs-list');

    if (!jobs || jobs.length === 0) {
      el.innerHTML = emptyState(
        '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>',
        'No jobs',
        'Jobs appear when you start detection, training, or inference'
      );
      return;
    }

    const sorted = [...jobs].sort((a, b) => {
      if (a.status === 'running' && b.status !== 'running') return -1;
      if (b.status === 'running' && a.status !== 'running') return 1;
      return (b.id || '').localeCompare(a.id || '');
    });

    el.innerHTML = `<div class="space-y-2.5">${sorted.map(job => {
      const isRunning = job.status === 'running';
      return `
        <div class="group p-4 rounded-xl border transition-all duration-200 space-y-3
          ${isRunning
            ? 'bg-indigo-500/[0.03] border-indigo-500/15 hover:border-indigo-500/25 hover:bg-indigo-500/[0.05]'
            : 'bg-surface-50/50 border-border hover:border-border-light hover:bg-white/[0.03]'}">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              ${createStatusBadge(job.status)}
              <span class="text-sm text-text-primary font-heading font-medium">${job.type || 'unknown'}</span>
              <span class="text-[11px] text-text-muted font-mono opacity-60">${job.id?.substring(0, 8) || ''}</span>
            </div>
            ${isRunning ? `<button class="job-cancel text-[11px] text-red-400/80 cursor-pointer hover:text-red-300 font-medium px-2 py-1 rounded-lg hover:bg-red-500/10 transition-all duration-200" data-id="${job.id}">Cancel</button>` : ''}
          </div>
          ${isRunning ? createProgressBar(job.progress) : ''}
          ${job.message ? `<p class="text-[11px] text-text-muted truncate leading-relaxed">${job.message}</p>` : ''}
          ${job.error ? `<p class="text-[11px] text-red-400/80 truncate leading-relaxed">${job.error}</p>` : ''}
        </div>`;
    }).join('')}</div>`;

    el.querySelectorAll('.job-cancel').forEach(btn => {
      btn.addEventListener('click', async () => {
        try {
          await api(`/jobs/${btn.dataset.id}/cancel`, { method: 'POST' });
          showToast('Job cancelled', 'warning');
          loadJobs();
        } catch (e) {
          showToast(`Cancel failed: ${e.message}`, 'error');
        }
      });
    });
  } catch { /* silently fail */ }
}
