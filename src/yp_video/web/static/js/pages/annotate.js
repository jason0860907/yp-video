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
        <select id="ann-results" class="${selectCls}">
          <option value="">Select result file...</option>
        </select>
        ${btnSmall('Load', 'id="ann-load"', 'primary')}
      `)}

      <div id="ann-stats" class="hidden flex flex-wrap items-center gap-2 px-1"></div>

      ${editor.bodyHTML()}

      ${kbdHint(ANNOTATION_KEYS)}
    </div>`;

  editor.bindEvents();
  document.getElementById('ann-load').addEventListener('click', loadFile);
  document.getElementById('ann-kind').addEventListener('change', renderResultsDropdown);

  loadResults();
  loadStats();
  editor.activate();
}

export const activate = () => editor.activate();
export const deactivate = () => editor.deactivate();

// ── Page-specific data loading ─────────────────────────────────────────

const SOURCE_CHIP_CLS = {
  vnl: 'bg-sky-500/10 text-sky-300 border-sky-500/20',
  u19: 'bg-purple-500/10 text-purple-300 border-purple-500/20',
  cev_u22: 'bg-pink-500/10 text-pink-300 border-pink-500/20',
  svl_japan: 'bg-rose-500/10 text-rose-300 border-rose-500/20',
  enterprise: 'bg-amber-500/10 text-amber-300 border-amber-500/20',
  tpvl: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20',
  other: 'bg-white/[0.06] text-text-muted border-border',
};

async function loadStats() {
  const el = document.getElementById('ann-stats');
  if (!el) return;
  try {
    const data = await api(API.annotate.stats);
    if (!data.by_source?.length) return;
    const chips = data.by_source.map(s => {
      const cls = SOURCE_CHIP_CLS[s.source] || SOURCE_CHIP_CLS.other;
      const done = s.cuts > 0 && s.annotated >= s.cuts;
      const mark = done ? '<span class="opacity-70">✓</span>' : '';
      return `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-medium ${cls}" title="${s.annotated} annotated / ${s.cuts} cuts">
        ${s.source}
        <span class="font-heading tabular-nums opacity-80">${s.annotated}/${s.cuts}</span>
        ${mark}
      </span>`;
    }).join('');
    const total = `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-border bg-surface-50/30 text-[11px] text-text-secondary ml-auto" title="annotated sets / total cuts">
      Total <span class="font-heading text-text-primary tabular-nums">${data.total_annotated}/${data.total_cuts}</span>
    </span>`;
    el.innerHTML = chips + total;
    el.classList.remove('hidden');
  } catch {
    // Stats are non-critical; stay silent on failure.
  }
}

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
