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
        ${btnSmall('✅ Mark Complete', 'id="ann-publish"', 'success')}
      `)}

      <div id="ann-publish-result"
           class="hidden rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 space-y-2">
        <div class="text-sm font-medium">
          ✅ Pushed to the app — paste this URL into VolleyIQ → Settings → Library manifest URL
        </div>
        <div class="flex gap-2 items-center">
          <input id="ann-manifest-url" readonly class="${selectCls} flex-1 text-xs" />
          ${btnSmall('Copy', 'id="ann-copy-url"', 'primary')}
        </div>
      </div>

      ${editor.bodyHTML()}

      ${kbdHint(ANNOTATION_KEYS)}
    </div>`;

  editor.bindEvents();
  document.getElementById('ann-load').addEventListener('click', loadFile);
  document.getElementById('ann-download').addEventListener('click', () => editor.openDownloadModal());
  document.getElementById('ann-publish').addEventListener('click', publishToApp);
  document.getElementById('ann-copy-url').addEventListener('click', copyManifestURL);
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
    document.getElementById('ann-publish-result').classList.add('hidden');
    showToast(`Loaded ${data.results?.length ?? 0} annotations`, 'success');
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}

// ── Mark complete → push to the iOS app ────────────────────────────────

async function publishToApp() {
  const video = editor.state.videoName;
  if (!video) return showToast('Load a result file first', 'warning');

  const btn = document.getElementById('ann-publish');
  btn.disabled = true;
  try {
    // Save the current annotations first so the export reads fresh data —
    // the export reads the on-disk rally-annotations file by basename.
    await api(API.annotate.annotations, {
      method: 'POST',
      body: {
        video,
        duration: editor.state.duration,
        annotations: editor.state.annotations,
      },
    });
    const res = await api(API.annotate.publish, { method: 'POST', body: { video } });
    document.getElementById('ann-manifest-url').value = res.manifest_url;
    document.getElementById('ann-publish-result').classList.remove('hidden');
    showToast(`Pushed ${res.rally_count} rallies to the app`, 'success');
  } catch (e) {
    showToast(`Publish failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

async function copyManifestURL() {
  const url = document.getElementById('ann-manifest-url').value;
  if (!url) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(url);
    } else {
      // navigator.clipboard needs a secure context (HTTPS / localhost);
      // fall back to a hidden textarea for plain-HTTP dashboard access.
      const ta = document.createElement('textarea');
      ta.value = url;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      if (!ok) throw new Error('execCommand copy failed');
    }
    showToast('Manifest URL copied', 'success');
  } catch (e) {
    showToast(`Copy failed: ${e.message}`, 'error');
  }
}
