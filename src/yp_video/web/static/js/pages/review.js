/**
 * Review page — Review TAD predictions and save corrected annotations.
 *
 * Editor mechanics live in components/annotation_editor.js. This file owns
 * the page chrome: the multi-axis filter dropdown (split / quality / kind),
 * the source::name picker, and the per-row confidence pill that's specific
 * to TAD prediction output.
 */
import {
  api, API, pageHeader, btnSmall, showToast, selectCls, kbdHint,
} from '../shared.js';
import { AnnotationEditor, ANNOTATION_KEYS } from '../components/annotation_editor.js';

const editor = new AnnotationEditor({
  prefix: 'rev',
  saveEndpoint: API.review.annotations,
  videoStreamPath: vp => `/api/review/video/${encodeURIComponent(vp)}`,
  // TAD predictions carry a confidence score; render it as a colored pill.
  rowExtras: rowExtras,
  previewBackoff: 5,
  // Review's source dropdown switches files frequently; releasing the buffer
  // on page leave avoids holding stale streams open.
  unloadOnDeactivate: true,
});

let results = [];

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-5">
      ${pageHeader('Review', 'Review TAD predictions and correct annotations', `
        <select id="rev-filter" class="${selectCls}" title="Filter by split / quality / kind">
          <option value="all">All files</option>
          <option value="val">Validation only</option>
          <option value="train">Training only</option>
          <option value="predict-only">Predict only (no annotation)</option>
          <option value="failing">Failing (mAP &lt; 30%)</option>
          <option value="val-failing">Val + failing</option>
          <option value="broadcast">Broadcast only</option>
          <option value="sideline">Sideline only</option>
        </select>
        <select id="rev-results" class="${selectCls} max-w-[14rem] truncate">
          <option value="">Select result file...</option>
        </select>
        ${btnSmall('Load', 'id="rev-load"', 'primary')}
        ${btnSmall('Download', 'id="rev-download"')}
      `)}

      ${editor.bodyHTML()}

      ${kbdHint(ANNOTATION_KEYS)}
    </div>`;

  editor.bindEvents();
  document.getElementById('rev-load').addEventListener('click', loadFile);
  document.getElementById('rev-download').addEventListener('click', () => editor.openDownloadModal());
  document.getElementById('rev-filter').addEventListener('change', renderResultsDropdown);

  loadResults();
  editor.activate();
}

export const activate = () => editor.activate();
export const deactivate = () => editor.deactivate();

// ── Page-specific helpers ──────────────────────────────────────────────

function rowExtras(a) {
  if (a.score == null) return '';
  const color = a.score > 0.7 ? 'text-emerald-400' : a.score > 0.4 ? 'text-amber-400' : 'text-text-muted';
  const bg = a.score > 0.7 ? 'bg-emerald-500/10 ring-emerald-500/20'
           : a.score > 0.4 ? 'bg-amber-500/10 ring-amber-500/20'
           : 'bg-white/5 ring-white/10';
  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-heading font-medium tabular-nums ring-1 ${color} ${bg}">${(a.score * 100).toFixed(0)}%</span>`;
}

async function loadResults() {
  try {
    results = await api(API.review.results);
    renderResultsDropdown();
  } catch (e) {
    showToast(`Failed to load results: ${e.message}`, 'error');
  }
}

function renderResultsDropdown() {
  const sel = document.getElementById('rev-results');
  const filter = document.getElementById('rev-filter')?.value || 'all';
  while (sel.options.length > 1) sel.remove(1);

  const annotatedNames = new Set(
    results.filter(r => r.source === 'annotation').map(r => r.name),
  );

  // Single-axis dropdown that combines split / quality / kind. Each filter
  // value corresponds to one accept predicate.
  const passes = (r) => {
    const isVal = r.subset === 'validation';
    const isTrain = r.subset === 'training';
    const isFailing = typeof r.map === 'number' && r.map < 0.3;
    const isPredictOnly = r.source === 'tad-prediction' && !annotatedNames.has(r.name);
    switch (filter) {
      case 'val': return isVal;
      case 'train': return isTrain;
      case 'predict-only': return isPredictOnly;
      case 'failing': return isFailing;
      case 'val-failing': return isVal && isFailing;
      case 'broadcast': return r.kind === 'broadcast';
      case 'sideline': return r.kind === 'sideline';
      default: return true;
    }
  };

  let kept = 0;
  results.forEach(r => {
    if (!passes(r)) return;
    const opt = document.createElement('option');
    opt.value = `${r.source}::${r.name}`;
    const srcTag = r.source === 'annotation' ? '✅' : '🤖';
    const valTag = r.subset === 'validation' ? ' [VAL]' : '';
    const kindTag = r.kind === 'sideline' ? ' [SIDE]' : '';
    // mAP describes the model's prediction vs ground truth — only meaningful
    // on the prediction entry. Annotation entries are the GT itself.
    const mapTag = (r.source === 'tad-prediction' && typeof r.map === 'number')
      ? ` (mAP=${(r.map * 100).toFixed(0)}%)` : '';
    opt.textContent = `${srcTag}${valTag}${kindTag} ${r.name}${mapTag}`;
    sel.appendChild(opt);
    kept++;
  });

  sel.options[0].textContent = kept ? `Select result file... (${kept})` : 'No matches';
}

async function loadFile() {
  const raw = document.getElementById('rev-results').value;
  if (!raw) return;
  const sep = raw.indexOf('::');
  const source = sep >= 0 ? raw.slice(0, sep) : '';
  const name = sep >= 0 ? raw.slice(sep + 2) : raw;
  try {
    const data = await api(API.review.result(name, source ? { source } : {}));
    editor.loadFromData(data);
    showToast(`Loaded ${data.results?.length ?? 0} annotations`, 'success');
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}

