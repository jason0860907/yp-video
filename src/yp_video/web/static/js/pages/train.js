/**
 * Train page — Feature extraction + annotation conversion + TAD training.
 */
import { api, SSEClient, card, pageHeader, stepBadge, statCard, sectionTitle, btnPrimary, btnSecondary, btnSmall, createProgressBar, showToast, showConfirm, emptyState, inputCls, selectCls } from '../shared.js';

let sseClient = null;
let videos = [];
let convVideos = [];
let selectedModel = 'large';

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-8">
      ${pageHeader('Train', 'TAD model training pipeline')}

      <div id="train-status-card"></div>

      <!-- Step 1: Extract Features -->
      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(1, 'primary')}
            <div>
              ${sectionTitle('Extract Features', 'V-JEPA 2.1 video features for TAD')}
            </div>
          </div>
          <div class="ml-10 space-y-4">
            <div class="space-y-2">
              ${sectionTitle(
                'Videos',
                '',
                btnSmall('Select All', 'id="train-select-all"') + ' ' +
                btnSmall('Deselect All', 'id="train-deselect-all"') + ' ' +
                btnSmall('Unprocessed', 'id="train-select-unprocessed"', 'primary') + ' ' +
                btnSmall('✅ Annotated', 'id="train-select-annotated"') + ' ' +
                btnSmall('⚡ Pre-annotated', 'id="train-select-pre-annotated"')
              )}
              <div id="train-videos" class="space-y-0.5 max-h-72 overflow-y-auto pr-1"></div>
            </div>
            <div class="flex items-end gap-4">
              <div>
                <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Model</label>
                <select id="train-feat-model" class="w-44 ${selectCls}">
                  <option value="base">ViT-B (768d, 80M)</option>
                  <option value="large" selected>ViT-L (1024d, 300M)</option>
                  <option value="giant">ViT-g (1408d, 1B)</option>
                  <option value="gigantic">ViT-G (1664d, 2B)</option>
                </select>
              </div>
              <div>
                <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Batch Size</label>
                <input id="train-feat-batch" type="number" value="32" min="1" max="64" class="w-28 ${inputCls}">
              </div>
              ${btnSecondary('Extract', 'id="train-extract"')}
              <span id="train-extract-count" class="text-xs text-text-muted font-heading tabular-nums ml-auto self-center"></span>
            </div>
          </div>
          <div id="train-extract-progress" class="ml-10 hidden space-y-2">
            <div id="train-extract-bar"></div>
            <p id="train-extract-msg" class="text-xs text-text-muted"></p>
          </div>
        </div>
      `)}

      <!-- Step 2: Convert Annotations -->
      ${card(`
        <div class="space-y-5">
          <div class="flex items-center gap-3">
            ${stepBadge(2, 'primary')}
            <div>
              ${sectionTitle('Convert Annotations', 'JSONL rally annotations → ActionFormer format')}
            </div>
          </div>
          <div class="ml-10 space-y-4">
            <div class="space-y-2">
              ${sectionTitle(
                'Videos',
                '',
                btnSmall('Select All', 'id="conv-select-all"') + ' ' +
                btnSmall('Deselect All', 'id="conv-deselect-all"') + ' ' +
                btnSmall('✅ Annotated', 'id="conv-select-annotated"', 'primary') + ' ' +
                btnSmall('⚡ Pre-annotated', 'id="conv-select-pre-annotated"')
              )}
              <div id="conv-videos" class="space-y-0.5 max-h-72 overflow-y-auto pr-1"></div>
            </div>
            <div class="flex items-end gap-4">
              <div>
                <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Train Ratio</label>
                <input id="train-ratio" type="number" value="0.8" min="0.1" max="0.99" step="0.05" class="w-28 ${inputCls}">
              </div>
              ${btnSecondary('Convert', 'id="train-convert"')}
              <span id="train-convert-status" class="text-xs text-text-muted self-center"></span>
              <span id="train-convert-count" class="text-xs text-text-muted font-heading tabular-nums ml-auto self-center"></span>
            </div>
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
          <div class="ml-10 flex items-end gap-4">
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">GPU</label>
              <input id="train-gpu" type="number" value="0" min="0" max="7" class="w-20 ${inputCls}">
            </div>
            <div>
              <label class="block text-[11px] text-text-muted mb-1.5 uppercase tracking-wider font-medium">Seed</label>
              <input id="train-seed" type="number" value="42" class="w-20 ${inputCls}">
            </div>
            <div>
              <span id="train-model-label" class="text-[11px] text-text-muted">Features: <span class="text-text-primary font-medium">ViT-L</span></span>
            </div>
            ${btnPrimary('Start Training', 'id="train-start"')}
          </div>
          <div id="train-progress" class="ml-10 hidden space-y-2">
            <div id="train-bar"></div>
            <p id="train-msg" class="text-xs text-text-muted"></p>
            <div id="train-logs" class="max-h-64 overflow-y-auto rounded-lg bg-black/30 border border-white/5 p-3 font-mono text-[11px] text-text-muted leading-relaxed hidden"></div>
          </div>
        </div>
      `)}

      <div id="train-performance"></div>
    </div>`;

  loadStatus();
  loadVideos();
  loadPerformance();
  bindEvents();
}

export function activate() {}
export function deactivate() {}

function updateExtractCount() {
  const el = document.getElementById('train-extract-count');
  if (!el) return;
  const sel = videos.filter(v => v.selected).length;
  el.textContent = videos.length ? `${sel} / ${videos.length} selected` : '';
}

function updateConvertCount() {
  const el = document.getElementById('train-convert-count');
  if (!el) return;
  const sel = convVideos.filter(v => v.selected).length;
  el.textContent = convVideos.length ? `${sel} / ${convVideos.length} selected` : '';
}

function bindEvents() {
  document.getElementById('train-convert').addEventListener('click', convertAnnotations);
  document.getElementById('train-extract').addEventListener('click', extractFeatures);
  document.getElementById('train-start').addEventListener('click', startTraining);
  document.getElementById('train-feat-model').addEventListener('change', (e) => {
    selectedModel = e.target.value;
    loadVideos();
    loadStatus();
    const names = { base: 'ViT-B', large: 'ViT-L', giant: 'ViT-g', gigantic: 'ViT-G' };
    document.getElementById('train-model-label').innerHTML =
      `Features: <span class="text-text-primary font-medium">${names[selectedModel] || selectedModel}</span>`;
  });
  document.getElementById('train-select-all').addEventListener('click', () => {
    videos.forEach(v => v.selected = true);
    renderVideos();
  });
  document.getElementById('train-deselect-all').addEventListener('click', () => {
    videos.forEach(v => v.selected = false);
    renderVideos();
  });
  document.getElementById('train-select-unprocessed').addEventListener('click', () => {
    videos.forEach(v => v.selected = !v.has_features);
    renderVideos();
  });
  document.getElementById('train-select-annotated').addEventListener('click', () => {
    videos.forEach(v => v.selected = v.has_annotation);
    renderVideos();
  });
  document.getElementById('train-select-pre-annotated').addEventListener('click', () => {
    videos.forEach(v => v.selected = v.has_pre_annotation);
    renderVideos();
  });
  document.getElementById('conv-select-all').addEventListener('click', () => {
    convVideos.forEach(v => v.selected = true);
    renderConvVideos();
  });
  document.getElementById('conv-deselect-all').addEventListener('click', () => {
    convVideos.forEach(v => v.selected = false);
    renderConvVideos();
  });
  document.getElementById('conv-select-annotated').addEventListener('click', () => {
    convVideos.forEach(v => v.selected = v.has_annotation);
    renderConvVideos();
  });
  document.getElementById('conv-select-pre-annotated').addEventListener('click', () => {
    convVideos.forEach(v => v.selected = v.has_pre_annotation);
    renderConvVideos();
  });
}

async function loadStatus() {
  try {
    const s = await api(`/train/status?model=${selectedModel}`);
    const featB = s.features_by_model?.base ?? 0;
    const featL = s.features_by_model?.large ?? 0;
    document.getElementById('train-status-card').innerHTML = card(`
      <div class="grid grid-cols-5 gap-3">
        ${statCard('Cuts', s.cuts_count, s.cuts_count > 0)}
        ${statCard('Features (B)', featB, featB > 0)}
        ${statCard('Features (L)', featL, featL > 0)}
        ${statCard('Annotations', s.annotations_exist ? 'ready' : 'missing', s.annotations_exist)}
        ${statCard('GPU', s.vllm_running ? 'shared' : 'available', true)}
      </div>
    `);

    // Reconnect to active training job
    if (s.active_train_job) {
      const btn = document.getElementById('train-start');
      btn.disabled = true;
      showTrainingUI(s.active_train_job);
      subscribeTrainingJob(s.active_train_job.id, btn);
    }
  } catch { /* silently fail */ }
}

async function loadVideos() {
  try {
    const list = await api(`/system/videos?model=${selectedModel}`);
    videos = list.map(v => ({ ...v, selected: !v.has_features }));
    convVideos = list.map(v => ({ ...v, selected: v.has_annotation || v.has_pre_annotation }));
    renderVideos();
    renderConvVideos();
  } catch (e) {
    showToast(`Failed to load videos: ${e.message}`, 'error');
  }
}

function videoBadges(v) {
  const b = [];
  if (v.has_annotation) b.push('<span title="Annotated">✅</span>');
  else if (v.has_pre_annotation) b.push('<span title="Pre-annotation">⚡</span>');
  if (v.has_features) b.push('<span class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>features</span>');
  return b.join(' ');
}

function renderVideos() {
  const el = document.getElementById('train-videos');
  if (videos.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>',
      'No cut videos found',
      'Cut some videos first'
    );
    return;
  }

  el.innerHTML = videos.map((v, i) => `
    <div class="group flex items-center gap-3 p-2.5 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200">
      <input type="checkbox" data-idx="${i}" class="train-check cursor-pointer accent-primary w-3.5 h-3.5" ${v.selected ? 'checked' : ''}>
      <span class="text-sm text-text-primary flex-1 truncate group-hover:text-white transition-colors duration-200">${v.name}</span>
      ${videoBadges(v)}
    </div>
  `).join('');

  el.querySelectorAll('.train-check').forEach(cb => {
    cb.addEventListener('change', (e) => {
      videos[parseInt(e.target.dataset.idx)].selected = e.target.checked;
      updateExtractCount();
    });
  });
  updateExtractCount();
}

function renderConvVideos() {
  const el = document.getElementById('conv-videos');
  if (convVideos.length === 0) {
    el.innerHTML = emptyState(
      '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>',
      'No cut videos found',
      'Cut some videos first'
    );
    return;
  }

  el.innerHTML = convVideos.map((v, i) => `
    <div class="group flex items-center gap-3 p-2.5 rounded-xl border border-transparent hover:bg-white/[0.03] hover:border-white/5 transition-all duration-200">
      <input type="checkbox" data-idx="${i}" class="conv-check cursor-pointer accent-primary w-3.5 h-3.5" ${v.selected ? 'checked' : ''}>
      <span class="text-sm text-text-primary flex-1 truncate group-hover:text-white transition-colors duration-200">${v.name}</span>
      ${videoBadges(v)}
    </div>
  `).join('');

  el.querySelectorAll('.conv-check').forEach(cb => {
    cb.addEventListener('change', (e) => {
      convVideos[parseInt(e.target.dataset.idx)].selected = e.target.checked;
      updateConvertCount();
    });
  });
  updateConvertCount();
}

const TIOU_COLORS = {
  '0.30': '#22d3ee', // cyan
  '0.40': '#34d399', // emerald
  '0.50': '#facc15', // yellow
  '0.60': '#fb923c', // orange
  '0.70': '#f87171', // red
};

async function loadPerformance() {
  try {
    const data = await api(`/train/performance?model=${selectedModel}`);
    const el = document.getElementById('train-performance');
    if (!data.entries?.length) {
      el.innerHTML = '';
      return;
    }

    // Extract tIoU keys from first entry that has tiou data
    const sample = data.entries.find(e => e.tiou && Object.keys(e.tiou).length > 0);
    if (!sample) {
      el.innerHTML = '';
      return;
    }
    const tiouKeys = Object.keys(sample.tiou).sort();
    const epochs = data.entries.filter(e => e.tiou).map(e => e.epoch);

    // Find max recall for Y axis
    let maxVal = 0;
    for (const e of data.entries) {
      if (!e.tiou) continue;
      for (const k of tiouKeys) {
        if (e.tiou[k]?.recall > maxVal) maxVal = e.tiou[k].recall;
      }
    }
    maxVal = Math.min(Math.ceil(maxVal * 10) / 10 + 0.1, 1.0);

    // SVG chart dimensions
    const W = 700, H = 260, pad = { t: 20, r: 20, b: 40, l: 50 };
    const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

    const xMin = Math.min(...epochs), xMax = Math.max(...epochs);
    const xRange = xMax - xMin || 1;
    const x = ep => pad.l + ((ep - xMin) / xRange) * cw;
    const y = val => pad.t + (1 - val / maxVal) * ch;

    // Build lines
    let lines = '';
    let dots = '';
    let legend = '';
    for (const tiou of tiouKeys) {
      const color = TIOU_COLORS[tiou] || '#888';
      const points = data.entries.filter(e => e.tiou?.[tiou]).map(e => ({
        x: x(e.epoch), y: y(e.tiou[tiou].recall),
        epoch: e.epoch, recall: e.tiou[tiou].recall,
      }));
      if (points.length === 0) continue;

      const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
      lines += `<path d="${pathD}" fill="none" stroke="${color}" stroke-width="2" opacity="0.85"/>`;
      for (const p of points) {
        dots += `<circle cx="${p.x}" cy="${p.y}" r="3" fill="${color}" opacity="0.9"><title>Epoch ${p.epoch} | tIoU ${tiou} | Recall ${(p.recall * 100).toFixed(1)}%</title></circle>`;
      }
      legend += `<span class="inline-flex items-center gap-1.5 text-[11px]"><span class="w-3 h-0.5 rounded" style="background:${color}"></span><span class="text-text-muted">tIoU ${tiou}</span></span>`;
    }

    // Y axis ticks
    let yAxis = '';
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const val = (maxVal / ySteps) * i;
      const yy = y(val);
      yAxis += `<line x1="${pad.l}" x2="${pad.l + cw}" y1="${yy}" y2="${yy}" stroke="white" stroke-opacity="0.06"/>`;
      yAxis += `<text x="${pad.l - 8}" y="${yy + 3}" text-anchor="end" fill="#888" font-size="10">${(val * 100).toFixed(0)}%</text>`;
    }

    // X axis ticks (show ~6 labels)
    let xAxis = '';
    const step = Math.max(1, Math.floor(epochs.length / 6));
    for (let i = 0; i < epochs.length; i += step) {
      const xx = x(epochs[i]);
      xAxis += `<text x="${xx}" y="${H - 8}" text-anchor="middle" fill="#888" font-size="10">${epochs[i]}</text>`;
    }

    // Right-side info panel
    let rightPanel = '';
    for (const tiou of tiouKeys) {
      const color = TIOU_COLORS[tiou] || '#888';
      const label = parseFloat(tiou);
      let best = 0, bestEp = 0;
      for (const e of data.entries) {
        if (e.tiou?.[tiou]?.recall > best) { best = e.tiou[tiou].recall; bestEp = e.epoch; }
      }
      rightPanel += `
        <div class="flex items-center gap-2 whitespace-nowrap">
          <span class="w-2 h-2 rounded-full flex-shrink-0" style="background:${color}"></span>
          <span class="text-text-muted text-[11px]">tIoU=${label}</span>
          <span class="text-text-primary text-xs font-medium tabular-nums">${(best * 100).toFixed(1)}%</span>
          <span class="text-text-muted text-[10px]">ep${bestEp}</span>
        </div>`;
    }

    el.innerHTML = card(`
      <div class="space-y-4">
        ${sectionTitle('Performance', data.name || '')}
        <div class="grid grid-cols-[3fr_1fr] gap-8 items-center">
          <svg viewBox="0 0 ${W} ${H}" class="w-full" style="max-height:280px">
            ${yAxis}${xAxis}${lines}${dots}
            <text x="${pad.l + cw / 2}" y="${H}" text-anchor="middle" fill="#666" font-size="10">Epoch</text>
            <text x="12" y="${pad.t + ch / 2}" text-anchor="middle" fill="#666" font-size="10" transform="rotate(-90, 12, ${pad.t + ch / 2})">Recall@1x</text>
          </svg>
          <div class="space-y-2.5">${rightPanel}</div>
        </div>
      </div>
    `);
  } catch { /* silently fail */ }
}

async function convertAnnotations() {
  const selected = convVideos.filter(v => v.selected).map(v => v.name);
  if (selected.length === 0) return showToast('No videos selected', 'warning');

  const btn = document.getElementById('train-convert');
  const status = document.getElementById('train-convert-status');
  btn.disabled = true;
  try {
    const res = await api('/train/convert-annotations', {
      method: 'POST',
      body: {
        train_ratio: parseFloat(document.getElementById('train-ratio').value),
        videos: selected,
        model: selectedModel,
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
  const selected = videos.filter(v => v.selected).map(v => v.name);
  if (selected.length === 0) return showToast('No videos selected', 'warning');

  let stopVllm = false;
  const vllmStatus = await api('/system/vllm/status').catch(() => null);
  if (vllmStatus?.status === 'running') {
    const ok = await showConfirm({
      title: 'Stop vLLM for feature extraction?',
      body:
        'vLLM is running and holds most of the GPU VRAM.\n' +
        'Feature extraction needs the GPU and would OOM otherwise.\n\n' +
        'vLLM will be automatically restarted once extraction finishes.',
      confirmText: 'Stop & Extract',
      cancelText: 'Cancel',
      variant: 'warning',
    });
    if (!ok) return;
    stopVllm = true;
  }

  const btn = document.getElementById('train-extract');
  btn.disabled = true;
  try {
    const res = await api('/train/extract-features', {
      method: 'POST',
      body: {
        videos: selected,
        batch_size: parseInt(document.getElementById('train-feat-batch').value),
        model: selectedModel,
        stop_vllm: stopVllm,
      },
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
          loadVideos();
        }
      },
      onError: () => { btn.disabled = false; },
    }).start();
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

function showTrainingUI(data) {
  document.getElementById('train-progress').classList.remove('hidden');
  document.getElementById('train-bar').innerHTML = createProgressBar(data.progress, 'accent');
  document.getElementById('train-msg').textContent = data.message || '';
  const logsEl = document.getElementById('train-logs');
  if (data.logs?.length > 0) {
    logsEl.classList.remove('hidden');
    logsEl.innerHTML = data.logs.map(l => `<div>${l.replace(/</g, '&lt;')}</div>`).join('');
    logsEl.scrollTop = logsEl.scrollHeight;
  }
}

function subscribeTrainingJob(jobId, btn) {
  sseClient?.stop();
  sseClient = new SSEClient(`/api/jobs/${jobId}/events`, {
    onMessage: (data) => {
      showTrainingUI(data);
      // Refresh performance chart on validation results
      if (data.message?.includes('mAP')) {
        loadPerformance();
      }
      if (data.status === 'completed' || data.status === 'failed') {
        sseClient?.stop();
        btn.disabled = false;
        showToast(data.status === 'completed' ? 'Training complete!' : `Failed: ${data.error}`, data.status === 'completed' ? 'success' : 'error');
        loadStatus();
        loadPerformance();
      }
    },
    onError: () => { btn.disabled = false; },
  }).start();
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
        model: selectedModel,
      },
    });
    showTrainingUI(res);
    subscribeTrainingJob(res.id, btn);
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
    btn.disabled = false;
  }
}
