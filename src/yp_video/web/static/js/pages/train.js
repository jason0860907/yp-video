/**
 * Train page — Feature extraction + annotation conversion + TAD training.
 */
import { api, SSEClient, card, pageHeader, stepBadge, statCard, sectionTitle, btnPrimary, btnSecondary, createProgressBar, showToast, inputCls, selectCls } from '../shared.js';

let sseClient = null;

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-8">
      ${pageHeader('Train', 'TAD model training pipeline')}

      <div id="train-status-card"></div>

      <!-- Step 1: Convert Annotations -->
      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(1, 'primary')}
            <div>
              ${sectionTitle('Convert Annotations', 'JSONL rally annotations → OpenTAD format')}
            </div>
          </div>
          <div class="ml-10 grid grid-cols-2 gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Source</label>
              <select id="train-source" class="w-full ${selectCls}">
                <option value="rally-annotations">rally-annotations</option>
                <option value="rally-pre-annotations">rally-pre-annotations</option>
              </select>
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Train Ratio</label>
              <input id="train-ratio" type="number" value="0.8" min="0.1" max="0.99" step="0.05" class="w-full ${inputCls}">
            </div>
          </div>
          <div class="ml-10 flex items-center gap-3">
            ${btnSecondary('Convert', 'id="train-convert"')}
            <span id="train-convert-status" class="text-xs text-text-muted"></span>
          </div>
        </div>
      `)}

      <!-- Step 2: Extract Features -->
      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(2, 'primary')}
            <div>
              ${sectionTitle('Extract Features', 'R3D-18 video features for TAD')}
            </div>
          </div>
          <div class="ml-10 flex items-center gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Batch Size</label>
              <input id="train-feat-batch" type="number" value="64" min="1" max="256" class="w-28 ${inputCls}">
            </div>
            ${btnSecondary('Extract', 'id="train-extract"')}
          </div>
          <div id="train-extract-progress" class="ml-10 hidden space-y-2">
            <div id="train-extract-bar"></div>
            <p id="train-extract-msg" class="text-xs text-text-muted"></p>
          </div>
        </div>
      `)}

      <!-- Step 3: Train Model -->
      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(3, 'accent')}
            <div>
              ${sectionTitle('Train Model', 'TAD model training with extracted features')}
            </div>
          </div>
          <div class="ml-10 flex items-center gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">GPU</label>
              <input id="train-gpu" type="number" value="0" min="0" max="7" class="w-20 ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Seed</label>
              <input id="train-seed" type="number" value="42" class="w-20 ${inputCls}">
            </div>
            ${btnPrimary('Start Training', 'id="train-start"')}
          </div>
          <div id="train-progress" class="ml-10 hidden space-y-2">
            <div id="train-bar"></div>
            <p id="train-msg" class="text-xs text-text-muted"></p>
          </div>
        </div>
      `)}

      <div id="train-checkpoints"></div>
    </div>`;

  loadStatus();
  bindEvents();
}

export function activate() {}
export function deactivate() {}

function bindEvents() {
  document.getElementById('train-convert').addEventListener('click', convertAnnotations);
  document.getElementById('train-extract').addEventListener('click', extractFeatures);
  document.getElementById('train-start').addEventListener('click', startTraining);
}

async function loadStatus() {
  try {
    const s = await api('/train/status');
    document.getElementById('train-status-card').innerHTML = card(`
      <div class="grid grid-cols-4 gap-3">
        ${statCard('Cuts', s.cuts_count, s.cuts_count > 0)}
        ${statCard('Features', s.features_count, s.features_count > 0)}
        ${statCard('Annotations', s.annotations_exist ? 'ready' : 'missing', s.annotations_exist)}
        ${statCard('GPU', s.gpu_available ? 'available' : 'busy', s.gpu_available)}
      </div>
    `);

    if (s.checkpoints?.length > 0) {
      document.getElementById('train-checkpoints').innerHTML = card(`
        <div class="space-y-3">
          ${sectionTitle('Checkpoints')}
          <div class="space-y-1.5">
            ${s.checkpoints.map(cp => `
              <div class="flex items-center gap-3 p-2.5 rounded-xl bg-surface-50/50 border border-border text-sm transition-colors duration-150 hover:bg-surface-100/50">
                <svg class="w-4 h-4 text-emerald-400 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4"/></svg>
                <span class="text-text-primary font-heading text-xs truncate">${cp}</span>
              </div>
            `).join('')}
          </div>
        </div>
      `);
    }
  } catch { /* silently fail */ }
}

async function convertAnnotations() {
  const btn = document.getElementById('train-convert');
  const status = document.getElementById('train-convert-status');
  btn.disabled = true;
  try {
    const res = await api('/train/convert-annotations', {
      method: 'POST',
      body: {
        source: document.getElementById('train-source').value,
        train_ratio: parseFloat(document.getElementById('train-ratio').value),
      },
    });
    status.textContent = `${res.video_count} videos converted`;
    status.className = 'text-xs text-emerald-400';
    showToast('Annotations converted!', 'success');
  } catch (e) {
    status.textContent = e.message;
    status.className = 'text-xs text-red-400';
  } finally {
    btn.disabled = false;
  }
}

async function extractFeatures() {
  const btn = document.getElementById('train-extract');
  btn.disabled = true;
  try {
    const res = await api('/train/extract-features', {
      method: 'POST',
      body: { batch_size: parseInt(document.getElementById('train-feat-batch').value) },
    });
    document.getElementById('train-extract-progress').classList.remove('hidden');
    sseClient = new SSEClient(`/api/jobs/${res.id}/events`, {
      onMessage: (data) => {
        document.getElementById('train-extract-bar').innerHTML = createProgressBar(data.progress);
        document.getElementById('train-extract-msg').textContent = data.message || '';
        if (data.status === 'completed' || data.status === 'failed') {
          sseClient?.stop();
          btn.disabled = false;
          showToast(data.status === 'completed' ? 'Features extracted!' : `Failed: ${data.error}`, data.status === 'completed' ? 'success' : 'error');
          loadStatus();
        }
      },
      onError: () => { btn.disabled = false; },
    }).start();
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

async function startTraining() {
  const btn = document.getElementById('train-start');
  btn.disabled = true;
  try {
    const res = await api('/train/start', {
      method: 'POST',
      body: {
        gpu: parseInt(document.getElementById('train-gpu').value),
        seed: parseInt(document.getElementById('train-seed').value),
      },
    });
    document.getElementById('train-progress').classList.remove('hidden');
    sseClient = new SSEClient(`/api/jobs/${res.id}/events`, {
      onMessage: (data) => {
        document.getElementById('train-bar').innerHTML = createProgressBar(data.progress, 'accent');
        document.getElementById('train-msg').textContent = data.message || '';
        if (data.status === 'completed' || data.status === 'failed') {
          sseClient?.stop();
          btn.disabled = false;
          showToast(data.status === 'completed' ? 'Training complete!' : `Failed: ${data.error}`, data.status === 'completed' ? 'success' : 'error');
          loadStatus();
        }
      },
      onError: () => { btn.disabled = false; },
    }).start();
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
    btn.disabled = false;
  }
}
