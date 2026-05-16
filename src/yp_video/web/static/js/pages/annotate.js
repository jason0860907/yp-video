/**
 * Annotate page — Rally annotation with timeline visualization.
 *
 * Most of the editor (video player + timeline + annotation list + keyboard
 * shortcuts) lives in components/annotation_editor.js. This file owns the
 * page-level chrome: the kind/source filter dropdowns, per-source progress
 * chips, and choosing which result file to load into the editor.
 */
import {
  api, API, pageHeader, btnSmall, showToast, selectCls, kbdHint,
} from '../shared.js';
import { AnnotationEditor, ANNOTATION_KEYS } from '../components/annotation_editor.js';

const editor = new AnnotationEditor({
  prefix: 'ann',
  saveEndpoint: API.annotate.annotations,
  videoStreamPath: vp => `/api/annotate/video/${encodeURIComponent(vp)}`,
  previewBackoff: 3,
});

let results = [];

export function render(container) {
  container.innerHTML = `
    <div class="max-w-screen-2xl mx-auto space-y-5">
      ${pageHeader('Annotate', 'Review and edit rally annotations', `
        <select id="ann-kind" class="${selectCls}" title="Filter by cut kind">
          <option value="all">All kinds</option>
          <option value="broadcast">Broadcast only</option>
          <option value="sideline">Sideline only</option>
        </select>
        <select id="ann-results" class="${selectCls} max-w-[14rem] truncate">
          <option value="">Select result file...</option>
        </select>
        ${btnSmall('Load', 'id="ann-load"', 'primary')}
        ${btnSmall('Download', 'id="ann-download"')}
      `)}

      ${editor.bodyHTML()}

      ${kbdHint(ANNOTATION_KEYS)}
    </div>`;

  editor.bindEvents();
  document.getElementById('ann-load').addEventListener('click', loadFile);
  document.getElementById('ann-download').addEventListener('click', () => editor.openDownloadModal());
  document.getElementById('ann-kind').addEventListener('change', renderResultsDropdown);

  loadResults();
  editor.activate();
}

export const activate = () => editor.activate();
export const deactivate = () => editor.deactivate();

// ── Page-specific data loading ─────────────────────────────────────────

async function loadResults() {
  try {
    results = await api(API.annotate.results);
    renderResultsDropdown();
  } catch (e) {
    showToast(`Failed to load results: ${e.message}`, 'error');
  }
}

function renderResultsDropdown() {
  const sel = document.getElementById('ann-results');
  const kindFilter = document.getElementById('ann-kind')?.value || 'all';
  // Preserve current selection if it still passes the filter; otherwise clear.
  const prev = sel.value;
  while (sel.options.length > 1) sel.remove(1);
  let prevStillVisible = false;
  let kept = 0;
  for (const r of results) {
    if (kindFilter !== 'all' && r.kind !== kindFilter) continue;
    const opt = document.createElement('option');
    opt.value = r.name;
    const tag = r.source.includes('annotation') ? '✅' : '⚡';
    const kindTag = r.kind === 'sideline' ? ' [SIDE]' : '';
    opt.textContent = `${tag}${kindTag} ${r.name}`;
    sel.appendChild(opt);
    if (r.name === prev) prevStillVisible = true;
    kept++;
  }
  sel.value = prevStillVisible ? prev : '';
  sel.options[0].textContent = kept ? `Select result file... (${kept})` : 'No matches';
}

async function loadFile() {
  const name = document.getElementById('ann-results').value;
  if (!name) return;
  try {
    const data = await api(API.annotate.result(name));
    editor.loadFromData(data);
    showToast(`Loaded ${data.results?.length ?? 0} annotations`, 'success');
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}
