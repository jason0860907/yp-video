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
  // Saving also pushes the match to the app — see pushToApp().
  onSaved: pushToApp,
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

      <!-- App-export footer. Saving pushes the match to the app; this row
           keeps the resulting import URL handy without being in the way. -->
      <div id="ann-app-export" class="hidden text-xs opacity-70 border-t border-white/10 pt-3">
        <div>📲 App import URL — paste into VolleyIQ → Settings → Library manifest URL</div>
        <div class="flex gap-2 items-center mt-1.5">
          <input id="ann-manifest-url" readonly class="${selectCls} flex-1 text-xs" />
          ${btnSmall('Copy', 'id="ann-copy-url"')}
        </div>
      </div>
    </div>`;

  editor.bindEvents();
  document.getElementById('ann-load').addEventListener('click', loadFile);
  document.getElementById('ann-download').addEventListener('click', () => editor.openDownloadModal());
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
    // Hide the previous match's import URL until this one is saved/pushed.
    document.getElementById('ann-app-export').classList.add('hidden');
    showToast(`Loaded ${data.results?.length ?? 0} annotations`, 'success');
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}

// ── Push to the app (runs automatically after every Save) ──────────────

/**
 * Post-save hook: upload the just-saved match to the app's R2 library.
 * The first push for a match also uploads its (large) cut video, so this
 * can be slow once per match; later pushes only re-send the small
 * manifest. Always notifies the user — success or failure — and never
 * throws, so a push failure can't masquerade as a failed Save.
 */
async function pushToApp(state) {
  if (!state.videoName) return;
  showToast('Pushing this match to the app…', 'info');
  try {
    const res = await api(API.annotate.publish, {
      method: 'POST',
      body: { video: state.videoName },
    });
    document.getElementById('ann-manifest-url').value = res.manifest_url;
    document.getElementById('ann-app-export').classList.remove('hidden');
    showToast(
      res.video_uploaded
        ? `Pushed to the app — video + ${res.rally_count} rallies uploaded`
        : `Pushed to the app — ${res.rally_count} rallies updated (video unchanged)`,
      'success',
    );
  } catch (e) {
    showToast(`Saved locally, but push to app failed: ${e.message}`, 'error');
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
