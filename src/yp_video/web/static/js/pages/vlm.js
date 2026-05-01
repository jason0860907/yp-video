/**
 * VLM Train page — fine-tune Qwen3.5-VL on 6-second rally / non_rally classification.
 *
 * Mirrors the TAD Train flow but with two steps instead of three:
 *  1. Build manifest (slice cuts into windows, label by IoU with rally annotations)
 *  2. Start LoRA fine-tune
 */
import {
  api,
  API,
  SSEClient,
  card,
  pageHeader,
  stepBadge,
  statCard,
  sectionTitle,
  btnPrimary,
  btnSmall,
  createProgressBar,
  showToast,
  inputCls,
  selectCls,
  escapeHtml,
} from '../shared.js';

let sseClient = null;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-8">
      ${pageHeader('VLM Train', 'Qwen3.5-VL LoRA fine-tune for rally / non_rally clip classification')}

      <div id="vlm-status-card"></div>

      <!-- Step 1: Build manifest -->
      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(1, 'primary')}
            <div>${sectionTitle('Build Manifest', 'Slide windows over cuts/, label by IoU with rally-annotations/')}</div>
          </div>
          <div class="ml-10 grid grid-cols-4 gap-3 items-end">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Window (s)</label>
              <input id="vlm-window" type="number" step="0.5" value="6" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Stride (s)</label>
              <input id="vlm-stride" type="number" step="0.5" value="2" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">IoU threshold</label>
              <input id="vlm-iou" type="number" step="0.05" min="0" max="1" value="0.5" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Train ratio</label>
              <input id="vlm-ratio" type="number" step="0.05" min="0.1" max="0.99" value="0.8" class="w-full ${inputCls}">
            </div>
          </div>
          <div class="ml-10 flex items-center gap-3">
            ${btnSmall('Build', 'id="vlm-build"', 'primary')}
            <span id="vlm-build-status" class="text-xs text-text-muted"></span>
          </div>
        </div>
      `)}

      <!-- Step 2: Train -->
      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(2, 'accent')}
            <div>${sectionTitle('Fine-tune', 'LoRA on Qwen3.5-VL')}</div>
          </div>
          <div class="ml-10 grid grid-cols-4 gap-3">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Model</label>
              <select id="vlm-model" class="w-full ${selectCls}">
                <option value="Qwen/Qwen3.5-0.8B" selected>Qwen3.5-0.8B (multimodal instruct)</option>
                <option value="Qwen/Qwen3.5-0.8B-Base">Qwen3.5-0.8B-Base (foundation)</option>
              </select>
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Epochs</label>
              <input id="vlm-epochs" type="number" value="3" min="1" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Batch size</label>
              <input id="vlm-batch" type="number" value="4" min="1" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Grad accum</label>
              <input id="vlm-accum" type="number" value="4" min="1" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Learning rate</label>
              <input id="vlm-lr" type="number" step="any" value="0.0001" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Frames / clip</label>
              <input id="vlm-frames" type="number" value="8" min="1" max="32" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">LoRA r</label>
              <input id="vlm-lora-r" type="number" value="16" min="1" class="w-full ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">GPU</label>
              <input id="vlm-gpu" type="number" value="0" min="0" max="7" class="w-full ${inputCls}">
            </div>
          </div>
          <div class="ml-10 flex items-center gap-4">
            <label class="flex items-center gap-2 text-[12px] text-text-secondary">
              <input id="vlm-balanced" type="checkbox" checked class="accent-primary w-3.5 h-3.5">
              Balanced sampler (per source)
            </label>
            ${btnPrimary('Start Fine-tune', 'id="vlm-start"')}
          </div>
          <div id="vlm-progress" class="ml-10 hidden space-y-2">
            <div id="vlm-bar"></div>
            <p id="vlm-msg" class="text-xs text-text-muted"></p>
            <div id="vlm-logs" class="max-h-64 overflow-y-auto rounded-lg bg-black/30 border border-white/5 p-3 font-mono text-[11px] text-text-muted leading-relaxed hidden"></div>
          </div>
        </div>
      `)}

      <div id="vlm-performance"></div>
    </div>`;

  loadStatus();
  loadPerformance();
  bindEvents();
}

export function activate() {}
export function deactivate() {
  sseClient?.stop();
  sseClient = null;
}

function bindEvents() {
  document.getElementById('vlm-build').addEventListener('click', buildManifest);
  document.getElementById('vlm-start').addEventListener('click', startTraining);
}

async function loadStatus() {
  try {
    const s = await api(API.vlm.status);
    document.getElementById('vlm-status-card').innerHTML = card(`
      <div class="grid grid-cols-5 gap-3">
        ${statCard('Cuts', s.cuts_count, s.cuts_count > 0)}
        ${statCard('Annotations', s.annotations_count, s.annotations_count > 0)}
        ${statCard('Manifest', s.manifest_exists ? 'ready' : 'missing', s.manifest_exists)}
        ${statCard('Windows', s.n_windows.toLocaleString(), s.n_windows > 0)}
        ${statCard('Rally / Non', `${s.n_rally} / ${s.n_non_rally}`, s.n_rally > 0)}
      </div>
    `);
    if (s.active_train_job) {
      const btn = document.getElementById('vlm-start');
      btn.disabled = true;
      showTrainingUI(s.active_train_job);
      subscribeJob(s.active_train_job.id, btn);
    }
  } catch { /* ignore */ }
}

async function buildManifest() {
  const btn = document.getElementById('vlm-build');
  const status = document.getElementById('vlm-build-status');
  btn.disabled = true;
  status.textContent = 'Building...';
  status.className = 'text-xs text-text-muted';
  try {
    const res = await api(API.vlm.buildManifest, {
      method: 'POST',
      body: {
        window: parseFloat(document.getElementById('vlm-window').value),
        stride: parseFloat(document.getElementById('vlm-stride').value),
        iou_threshold: parseFloat(document.getElementById('vlm-iou').value),
        train_ratio: parseFloat(document.getElementById('vlm-ratio').value),
      },
    });
    status.textContent = `${res.n_windows.toLocaleString()} windows (${res.n_rally} rally / ${res.n_non_rally} non) — train videos ${res.n_train_videos}, val ${res.n_val_videos}`;
    status.className = 'text-xs text-emerald-400';
    showToast('Manifest built', 'success');
    loadStatus();
  } catch (e) {
    status.textContent = e.message;
    status.className = 'text-xs text-red-400';
  } finally {
    btn.disabled = false;
  }
}

function showTrainingUI(data) {
  document.getElementById('vlm-progress').classList.remove('hidden');
  document.getElementById('vlm-bar').innerHTML = createProgressBar(data.progress || 0, 'accent');
  document.getElementById('vlm-msg').textContent = data.message || '';
  const logsEl = document.getElementById('vlm-logs');
  if (data.logs?.length > 0) {
    logsEl.classList.remove('hidden');
    logsEl.innerHTML = data.logs.slice(-200).map(l => `<div>${escapeHtml(l)}</div>`).join('');
    logsEl.scrollTop = logsEl.scrollHeight;
  }
}

function subscribeJob(jobId, btn) {
  sseClient?.stop();
  sseClient = new SSEClient(API.jobs.eventsSSE(jobId), {
    onMessage: (data) => {
      showTrainingUI(data);
      if (data.message?.includes('VLM Eval')) loadPerformance();
      if (data.status === 'completed' || data.status === 'failed') {
        sseClient?.stop();
        btn.disabled = false;
        showToast(
          data.status === 'completed' ? 'VLM training complete!' : `Failed: ${data.error}`,
          data.status === 'completed' ? 'success' : 'error',
        );
        loadStatus();
        loadPerformance();
      }
    },
    onError: () => { btn.disabled = false; },
  }).start();
}

async function startTraining() {
  const btn = document.getElementById('vlm-start');
  btn.disabled = true;
  try {
    const res = await api(API.vlm.start, {
      method: 'POST',
      body: {
        model: document.getElementById('vlm-model').value,
        epochs: parseInt(document.getElementById('vlm-epochs').value),
        batch_size: parseInt(document.getElementById('vlm-batch').value),
        gradient_accumulation: parseInt(document.getElementById('vlm-accum').value),
        lr: parseFloat(document.getElementById('vlm-lr').value),
        n_frames: parseInt(document.getElementById('vlm-frames').value),
        lora_r: parseInt(document.getElementById('vlm-lora-r').value),
        gpu: parseInt(document.getElementById('vlm-gpu').value),
        balanced_sampler: document.getElementById('vlm-balanced').checked,
      },
    });
    showTrainingUI(res);
    subscribeJob(res.id, btn);
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

async function loadPerformance() {
  try {
    const data = await api(API.vlm.performance);
    const el = document.getElementById('vlm-performance');
    if (!data.entries?.length) { el.innerHTML = ''; return; }

    const rows = data.entries.map(e => {
      const acc = (e.accuracy ?? 0) * 100;
      const bySrc = e.by_source || {};
      const srcCells = Object.entries(bySrc).sort()
        .map(([s, v]) => `<td class="px-2 py-1 text-text-secondary tabular-nums">${(v.acc * 100).toFixed(1)}%</td>`).join('');
      return `<tr class="hover:bg-white/[0.02]">
        <td class="px-2 py-1.5 text-text-muted">${e.epoch ?? '-'}</td>
        <td class="px-2 py-1.5 font-heading text-text-primary tabular-nums">${acc.toFixed(1)}%</td>
        ${srcCells}
      </tr>`;
    }).join('');
    const sources = Object.keys(data.entries.at(-1)?.by_source || {}).sort();
    const head = `<tr class="text-text-muted text-[11px] uppercase tracking-wider">
      <th class="px-2 py-1.5 text-left">Epoch</th>
      <th class="px-2 py-1.5 text-left">Accuracy</th>
      ${sources.map(s => `<th class="px-2 py-1.5 text-left">${s}</th>`).join('')}
    </tr>`;
    const meta = data.meta;
    const metaLine = meta ? `<div class="text-[11px] text-text-muted">
      ${meta.model} · LoRA r=${meta.lora?.r} · lr=${meta.opt?.lr} · ${meta.opt?.epochs}ep · ${meta.n_frames} frames · ${meta.n_train_windows.toLocaleString()} train / ${meta.n_val_windows.toLocaleString()} val
    </div>` : '';
    el.innerHTML = card(`
      <div class="space-y-3">
        ${sectionTitle('Performance', data.name || '')}
        ${metaLine}
        <div class="overflow-x-auto">
          <table class="w-full text-xs">
            <thead>${head}</thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    `);
  } catch { /* ignore */ }
}
