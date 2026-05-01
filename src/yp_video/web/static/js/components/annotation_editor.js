/**
 * Shared rally-annotation editor used by Annotate + Review pages.
 *
 * Handles the bottom 90% they had in common: <video>, timeline canvas,
 * annotation list with inline edit + delete + click-to-seek, mark-start /
 * mark-end mechanics, keyboard shortcuts, playhead-relative row highlight,
 * and save. Both pages now configure this with a small options bag and
 * just supply their own page header (filter dropdown, stats chips, etc).
 *
 * Event wiring uses delegation on the list container so the per-row
 * handlers don't have to be re-bound on every render.
 */
import {
  api, formatTime, parseTime, card, sectionTitle,
  btnSmall, showToast, showConfirm, emptyState,
} from '../shared.js';


export class AnnotationEditor {
  /**
   * @param {object} opts
   * @param {string} opts.prefix             DOM id prefix ('ann' / 'rev'); must
   *                                         be unique per page so two editors
   *                                         can coexist.
   * @param {string} opts.saveEndpoint       API path passed to api() on save.
   * @param {(videoPath: string) => string} opts.videoStreamPath
   *                                         How to turn a stored video path
   *                                         into a streamable URL.
   * @param {(a: object) => string} [opts.rowExtras]
   *                                         Extra HTML appended into each row
   *                                         (e.g. a confidence pill on Review).
   * @param {number} [opts.previewBackoff=3] Seconds before `end` that the
   *                                         "jump to end" button starts at.
   * @param {boolean} [opts.unloadOnDeactivate=false]
   *                                         Whether to fully detach the video
   *                                         src on page leave. Annotate keeps
   *                                         it (revisit shows last load);
   *                                         Review unloads (sources change).
   */
  constructor(opts) {
    this.prefix = opts.prefix;
    this.saveEndpoint = opts.saveEndpoint;
    this.videoStreamPath = opts.videoStreamPath;
    this.rowExtras = opts.rowExtras || (() => '');
    this.previewBackoff = opts.previewBackoff ?? 3;
    this.unloadOnDeactivate = opts.unloadOnDeactivate ?? false;

    this.state = { annotations: [], videoName: '', duration: 0, dirty: false };
    this.videoEl = null;
    this.timelineCanvas = null;
    this._animFrame = null;
    this._markStart = null;
    this._selectedIdx = -1;
    this._highlight = { inside: -1, prev: -1, next: -1 };
    this._handleKeydown = this._handleKeydown.bind(this);
    this._handleResize = this._resizeTimeline.bind(this);
  }

  // ── Public lifecycle ────────────────────────────────────────────────

  /** HTML for the editor body. The page concatenates this between its
   *  own header and footer. */
  bodyHTML() {
    const p = this.prefix;
    return `
      <div class="flex flex-col lg:flex-row gap-5">
        <div class="flex-1 min-w-0 space-y-4">
          ${card(`
            <div class="space-y-4">
              <div class="relative bg-black rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-lg shadow-black/30">
                <video id="${p}-player" class="w-full max-h-[45vh]" controls></video>
              </div>

              <div class="space-y-2">
                <div class="relative rounded-2xl overflow-hidden ring-1 ring-white/[0.06] shadow-inner" style="background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)">
                  <canvas id="${p}-timeline" class="w-full h-12 cursor-pointer" title="Click to seek"></canvas>
                </div>
                <div class="flex items-center justify-between px-0.5">
                  <span id="${p}-time" class="text-sm font-heading text-text-primary tabular-nums bg-surface-200/50 px-2.5 py-1 rounded-lg border border-border">00:00</span>
                  <div class="flex items-center gap-2">
                    ${btnSmall('Start [', `id="${p}-mark-start"`, 'primary')}
                    ${btnSmall('End ]', `id="${p}-mark-end"`, 'primary')}
                    ${btnSmall('Rally ↵', `id="${p}-add-rally"`, 'success')}
                    ${btnSmall('Non-Rally', `id="${p}-add-nonrally"`)}
                  </div>
                </div>
              </div>

              <div id="${p}-mark-info" class="hidden p-3 rounded-xl bg-primary/10 border border-primary/20 flex items-center gap-2.5">
                <span class="w-2 h-2 rounded-full bg-primary-light animate-pulse"></span>
                <span class="text-xs text-primary-light">Start marked at <strong id="${p}-mark-time" class="font-heading">0:00</strong> &mdash; press <kbd class="px-1.5 py-0.5 rounded bg-surface-200 border border-border text-[10px] font-heading text-text-secondary ml-0.5 mr-0.5">]</kbd> to set end</span>
              </div>
            </div>
          `)}
        </div>

        <div class="lg:w-[420px] lg:flex-shrink-0">
          ${card(`
            <div class="space-y-4">
              ${sectionTitle(
                `Annotations <span id="${p}-count" class="text-text-muted font-normal">(0)</span><span id="${p}-total-duration" class="text-text-muted/70 font-normal text-[11px] ml-1.5"></span>`,
                '',
                `${btnSmall('Save', `id="${p}-save"`, 'success')}
                 ${btnSmall('Clear', `id="${p}-clear"`, 'danger')}`,
              )}
              <div class="h-px bg-border"></div>
              <div id="${p}-list" class="space-y-1.5 max-h-[55vh] overflow-y-auto pr-1 scrollbar-thin"></div>
            </div>
          `)}
        </div>
      </div>`;
  }

  /** Wire event listeners — call once after the body HTML is in the DOM. */
  bindEvents() {
    const p = this.prefix;
    this.videoEl = document.getElementById(`${p}-player`);
    this.timelineCanvas = document.getElementById(`${p}-timeline`);

    document.getElementById(`${p}-mark-start`).addEventListener('click', () => this.markStart());
    document.getElementById(`${p}-mark-end`).addEventListener('click', () => this.markEnd());
    document.getElementById(`${p}-add-rally`).addEventListener('click', () => this.addAnnotation('rally'));
    document.getElementById(`${p}-add-nonrally`).addEventListener('click', () => this.addAnnotation('non-rally'));
    document.getElementById(`${p}-save`).addEventListener('click', () => this.save());
    document.getElementById(`${p}-clear`).addEventListener('click', () => this.clearAll());

    this.videoEl.addEventListener('loadedmetadata', () => {
      this.state.duration = this.videoEl.duration;
      this._resizeTimeline();
    });
    this.videoEl.addEventListener('timeupdate', () => {
      const tEl = document.getElementById(`${p}-time`);
      if (tEl) tEl.textContent = formatTime(this.videoEl.currentTime);
      this._refreshPlayingHighlight();
      // Auto-pause at end of selected segment
      if (this._selectedIdx >= 0 && this._selectedIdx < this.state.annotations.length) {
        const a = this.state.annotations[this._selectedIdx];
        if (!this.videoEl.paused && this.videoEl.currentTime >= a.end) {
          this.videoEl.pause();
          this.videoEl.currentTime = a.end;
          this._selectedIdx = -1;
        }
      }
    });

    this.timelineCanvas.addEventListener('click', (e) => {
      if (!this.state.duration) return;
      const rect = this.timelineCanvas.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      this.videoEl.currentTime = ratio * this.state.duration;
    });

    // Single delegated listener for all row interactions instead of
    // re-binding 4× per render.
    const listEl = document.getElementById(`${p}-list`);
    listEl.addEventListener('click', (e) => this._handleListClick(e));
    listEl.addEventListener('change', (e) => this._handleListChange(e));
  }

  activate() {
    document.addEventListener('keydown', this._handleKeydown);
    window.addEventListener('resize', this._handleResize);
    this._startTimelineLoop();
    this._resizeTimeline();
  }

  deactivate() {
    document.removeEventListener('keydown', this._handleKeydown);
    window.removeEventListener('resize', this._handleResize);
    if (this._animFrame) { cancelAnimationFrame(this._animFrame); this._animFrame = null; }
    if (this.videoEl) {
      if (!this.videoEl.paused) this.videoEl.pause();
      if (this.unloadOnDeactivate) {
        this.videoEl.removeAttribute('src');
        this.videoEl.load();
      }
    }
  }

  // ── Public mutators ────────────────────────────────────────────────

  /** Populate the editor from an API response. Sorts and re-renders. */
  loadFromData(data) {
    const videoPath = data.video || data.source_video || data.metadata?.video || '';
    this.state.videoName = videoPath;
    if (videoPath && this.videoEl) {
      // Pause + reset before swapping src so Chrome doesn't keep streaming
      // the previous file in the background.
      this.videoEl.pause();
      this.videoEl.removeAttribute('src');
      this.videoEl.load();
      this.videoEl.src = this.videoStreamPath(videoPath);
      this.videoEl.load();
    }

    this.state.annotations = (data.results || []).map(r => ({
      start: r.start ?? r.start_time ?? r.segment?.[0] ?? 0,
      end: r.end ?? r.end_time ?? r.segment?.[1] ?? 0,
      label: r.label || (r.in_rally ? 'rally' : 'non-rally'),
      score: r.confidence ?? r.score ?? null,
    }));
    this.state.annotations.sort((a, b) => a.start - b.start);
    this.state.dirty = false;
    this._renderAnnotations();
  }

  // ── Mark / add / delete ─────────────────────────────────────────────

  markStart() {
    if (!this.videoEl?.src) return;
    this._markStart = this.videoEl.currentTime;
    document.getElementById(`${this.prefix}-mark-info`).classList.remove('hidden');
    document.getElementById(`${this.prefix}-mark-time`).textContent = formatTime(this._markStart);
  }

  markEnd() {
    if (this._markStart == null) return;
    this.addAnnotation('rally');
  }

  addAnnotation(label) {
    if (this._markStart == null) return showToast('Mark start first with [', 'warning');
    const end = this.videoEl.currentTime;
    if (end <= this._markStart) return showToast('End must be after start', 'warning');
    this.state.annotations.push({ start: this._markStart, end, label });
    this.state.annotations.sort((a, b) => a.start - b.start);
    this._markStart = null;
    document.getElementById(`${this.prefix}-mark-info`).classList.add('hidden');
    this.state.dirty = true;
    this._renderAnnotations();
  }

  async clearAll() {
    if (this.state.annotations.length === 0) return;
    const ok = await showConfirm({
      title: 'Clear all annotations?',
      body: `This will remove ${this.state.annotations.length} annotation(s) from this view. The file on disk is not touched until you press Save.`,
      confirmText: 'Clear',
      variant: 'danger',
    });
    if (!ok) return;
    this.state.annotations = [];
    this._selectedIdx = -1;
    this.state.dirty = true;
    this._renderAnnotations();
  }

  async save() {
    if (!this.state.videoName) return showToast('No video loaded', 'warning');
    const btn = document.getElementById(`${this.prefix}-save`);
    btn.disabled = true;
    try {
      await api(this.saveEndpoint, {
        method: 'POST',
        body: {
          video: this.state.videoName,
          duration: this.state.duration,
          annotations: this.state.annotations,
        },
      });
      this.state.dirty = false;
      this._refreshDirtyIndicator();
      showToast('Annotations saved!', 'success');
    } catch (e) {
      showToast(`Save failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  // ── Keyboard ────────────────────────────────────────────────────────

  _handleKeydown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    switch (e.key) {
      case ' ': e.preventDefault(); this.videoEl.paused ? this.videoEl.play() : this.videoEl.pause(); break;
      case '[': e.preventDefault(); this.markStart(); break;
      case ']': e.preventDefault(); this.markEnd(); break;
      case 'Enter': e.preventDefault(); this.addAnnotation('rally'); break;
      case 'ArrowLeft': e.preventDefault(); this.videoEl.currentTime = Math.max(0, this.videoEl.currentTime - 5); break;
      case 'ArrowRight': e.preventDefault(); this.videoEl.currentTime += 5; break;
      case 't': case 'T':
        if (this._selectedIdx >= 0 && this._selectedIdx < this.state.annotations.length) {
          const a = this.state.annotations[this._selectedIdx];
          a.label = a.label === 'rally' ? 'non-rally' : 'rally';
          this.state.dirty = true;
          this._renderAnnotations();
        }
        break;
      case 'Delete': case 'Backspace':
        if (this._selectedIdx >= 0 && this._selectedIdx < this.state.annotations.length) {
          this.state.annotations.splice(this._selectedIdx, 1);
          this._selectedIdx = -1;
          this.state.dirty = true;
          this._renderAnnotations();
        }
        break;
    }
  }

  // ── Delegated list handlers ─────────────────────────────────────────

  _handleListClick(e) {
    const row = e.target.closest('.ae-item');
    if (!row) return;
    const idx = parseInt(row.dataset.idx);

    if (e.target.closest('[data-action="delete"]')) {
      this.state.annotations.splice(idx, 1);
      if (this._selectedIdx === idx) this._selectedIdx = -1;
      else if (this._selectedIdx > idx) this._selectedIdx--;
      this.state.dirty = true;
      this._renderAnnotations();
      return;
    }
    if (e.target.closest('[data-action="preview"]')) {
      this._selectedIdx = idx;
      const a = this.state.annotations[idx];
      this.videoEl.currentTime = Math.max(a.start, a.end - this.previewBackoff);
      this.videoEl.play();
      this._renderAnnotations();
      return;
    }
    // Inputs handle their own change events; ignore clicks inside them.
    if (e.target.tagName === 'INPUT') return;

    // Default: click anywhere else on the row → seek + play
    this._selectedIdx = idx;
    this.videoEl.currentTime = this.state.annotations[idx].start;
    this.videoEl.play();
    this._renderAnnotations();
  }

  _handleListChange(e) {
    if (e.target.tagName !== 'INPUT') return;
    const idx = parseInt(e.target.dataset.idx);
    const field = e.target.dataset.field;
    if (idx < 0 || isNaN(idx) || !field) return;
    this.state.annotations[idx][field] = parseTime(e.target.value);
    this.state.dirty = true;
    this._renderAnnotations();
  }

  // ── Render ──────────────────────────────────────────────────────────

  _refreshDirtyIndicator() {
    const btn = document.getElementById(`${this.prefix}-save`);
    if (!btn) return;
    btn.textContent = this.state.dirty ? 'Save •' : 'Save';
  }

  _renderAnnotations() {
    const p = this.prefix;
    const el = document.getElementById(`${p}-list`);
    // Count + total played time focus on rallies (the primary class). The
    // count is stored in `(N rally)` form so a quick scan tells the user
    // how many rallies were marked, and the trailing "MM:SS played" sums
    // their durations as a sanity check (e.g. 30-min broadcast → ~10 min
    // of actual play).
    const rallies = this.state.annotations.filter(a => a.label === 'rally');
    document.getElementById(`${p}-count`).textContent = `(${rallies.length} rally)`;

    const totalEl = document.getElementById(`${p}-total-duration`);
    if (totalEl) {
      const total = rallies.reduce((s, a) => s + (a.end - a.start), 0);
      totalEl.textContent = total > 0 ? `${formatTime(total)} played` : '';
    }

    this._refreshDirtyIndicator();
    // DOM was replaced — force highlight to re-apply on the new nodes.
    this._highlight = { inside: -1, prev: -1, next: -1 };

    if (this.state.annotations.length === 0) {
      el.innerHTML = emptyState(
        '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>',
        'No annotations',
        'Use [ ] to mark segments',
      );
      return;
    }

    el.innerHTML = this.state.annotations.map((a, i) => this._renderRow(a, i)).join('');
    this._refreshPlayingHighlight({ scroll: false });
  }

  _renderRow(a, i) {
    const isRally = a.label === 'rally';
    const selected = this._selectedIdx === i;
    const rowCls = isRally
      ? (selected ? 'border-emerald-500/40 bg-emerald-500/[0.08]' : 'border-emerald-500/15 bg-emerald-500/[0.04] hover:bg-emerald-500/[0.08]')
      : (selected ? 'border-primary/40 bg-primary/[0.08]' : 'border-border bg-surface-50/30 hover:bg-white/[0.04]');
    const labelCls = isRally
      ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/25'
      : 'bg-white/[0.06] text-text-muted ring-1 ring-white/10';
    const durationSec = (a.end - a.start).toFixed(1);

    return `
      <div class="ae-item flex items-center gap-2.5 px-3 py-2.5 rounded-xl border ${rowCls} cursor-pointer transition-all duration-200 group" data-idx="${i}">
        <span class="text-[10px] font-heading text-text-muted/60 w-4 text-right select-none">${i + 1}</span>
        <span class="${labelCls} px-2.5 py-0.5 rounded-full text-[11px] font-medium select-none">${a.label}</span>
        <div class="flex items-center gap-1.5 ml-auto">
          <input type="text" value="${formatTime(a.start)}" class="bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading focus:border-primary-light outline-none transition-colors duration-200 tabular-nums" data-idx="${i}" data-field="start">
          <span class="text-text-muted/40 text-[10px]">→</span>
          <input type="text" value="${formatTime(a.end)}" class="bg-transparent border-b border-white/10 text-text-primary text-[11px] w-14 text-center font-heading focus:border-primary-light outline-none transition-colors duration-200 tabular-nums" data-idx="${i}" data-field="end">
        </div>
        <span class="text-[10px] text-text-muted font-heading tabular-nums bg-surface-200/40 px-1.5 py-0.5 rounded">${durationSec}s</span>
        ${this.rowExtras(a)}
        <button class="text-primary-light hover:text-white cursor-pointer transition-colors duration-200" data-action="preview" title="Jump to end">
          <svg class="w-4 h-4 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
        </button>
        <button class="text-red-400/60 hover:text-red-400 cursor-pointer opacity-0 group-hover:opacity-100 transition-all duration-200" data-action="delete" title="Delete">
          <svg class="w-3.5 h-3.5 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>`;
  }

  // ── Playhead-relative row highlight ─────────────────────────────────

  _refreshPlayingHighlight({ scroll = true } = {}) {
    if (!this.videoEl) return;
    const t = this.videoEl.currentTime;

    let inside = -1, prev = -1, next = -1;
    for (let i = 0; i < this.state.annotations.length; i++) {
      const a = this.state.annotations[i];
      if (t >= a.start && t < a.end) { inside = i; prev = -1; next = -1; break; }
      if (a.end <= t) prev = i;
      if (a.start > t && next === -1) next = i;
    }

    if (inside === this._highlight.inside && prev === this._highlight.prev && next === this._highlight.next) return;

    const listEl = document.getElementById(`${this.prefix}-list`);
    if (!listEl) { this._highlight = { inside, prev, next }; return; }

    const STYLE = {
      inside: 'inset 3px 0 0 #F97316',
      prev:   'inset 3px 0 0 rgba(249, 115, 22, 0.4)',
      next:   'inset 3px 0 0 rgba(249, 115, 22, 0.4)',
    };
    const applyStyle = (idx, kind) => {
      if (idx < 0) return;
      const row = listEl.querySelector(`.ae-item[data-idx="${idx}"]`);
      if (!row) return;
      if (kind) {
        row.style.boxShadow = STYLE[kind];
        row.dataset.playing = '1';
      } else {
        row.style.boxShadow = '';
        delete row.dataset.playing;
      }
    };

    applyStyle(this._highlight.inside, null);
    applyStyle(this._highlight.prev, null);
    applyStyle(this._highlight.next, null);
    applyStyle(inside, 'inside');
    applyStyle(prev, 'prev');
    applyStyle(next, 'next');

    if (scroll) {
      const target = inside >= 0 ? inside : (next >= 0 ? next : prev);
      if (target >= 0) {
        const row = listEl.querySelector(`.ae-item[data-idx="${target}"]`);
        row?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }

    this._highlight = { inside, prev, next };
  }

  // ── Timeline canvas ─────────────────────────────────────────────────

  _resizeTimeline() {
    if (!this.timelineCanvas) return;
    const rect = this.timelineCanvas.getBoundingClientRect();
    if (rect.width === 0) return;
    this.timelineCanvas.width = rect.width * devicePixelRatio;
    this.timelineCanvas.height = rect.height * devicePixelRatio;
  }

  _startTimelineLoop() {
    const draw = () => {
      this._animFrame = requestAnimationFrame(draw);
      if (!this.timelineCanvas || !this.state.duration) return;

      const ctx = this.timelineCanvas.getContext('2d');
      const w = this.timelineCanvas.width;
      const h = this.timelineCanvas.height;
      const dpr = devicePixelRatio;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      ctx.fillRect(0, 0, w, h);

      for (const a of this.state.annotations) {
        const x1 = (a.start / this.state.duration) * w;
        const x2 = (a.end / this.state.duration) * w;
        if (a.label === 'rally') {
          const grad = ctx.createLinearGradient(x1, 0, x1, h);
          grad.addColorStop(0, 'rgba(34, 197, 94, 0.5)');
          grad.addColorStop(1, 'rgba(34, 197, 94, 0.25)');
          ctx.fillStyle = grad;
        } else {
          ctx.fillStyle = 'rgba(100, 116, 139, 0.2)';
        }
        ctx.beginPath();
        ctx.roundRect(x1, 2 * dpr, x2 - x1, h - 4 * dpr, 3 * dpr);
        ctx.fill();
      }

      if (this.videoEl && !isNaN(this.videoEl.currentTime)) {
        const px = (this.videoEl.currentTime / this.state.duration) * w;
        ctx.fillStyle = '#F97316';
        ctx.shadowColor = 'rgba(249,115,22,0.6)';
        ctx.shadowBlur = 6 * dpr;
        ctx.fillRect(px - 1 * dpr, 0, 2 * dpr, h);
        ctx.shadowBlur = 0;
      }

      if (this._markStart != null) {
        const mx = (this._markStart / this.state.duration) * w;
        ctx.fillStyle = '#6366F1';
        ctx.shadowColor = 'rgba(99,102,241,0.6)';
        ctx.shadowBlur = 6 * dpr;
        ctx.fillRect(mx - 1 * dpr, 0, 2 * dpr, h);
        ctx.shadowBlur = 0;
      }
    };
    draw();
  }
}


// Default keyboard hint shared by both pages.
export const ANNOTATION_KEYS = [
  ['Space', 'play/pause'],
  ['[', 'mark start'],
  [']', 'mark end'],
  ['Enter', 'rally'],
  ['← →', '±5s'],
  ['T', 'rally/non-rally'],
  ['Del', 'remove'],
];
