/**
 * Shared utilities & design system for YP Video Pipeline.
 * Design: Indigo/Orange accents, OLED dark, glass cards.
 */

// ── Design tokens ──
export const COLORS = {
  primary: '#6366F1',
  primaryLight: '#818CF8',
  accent: '#F97316',
  accentLight: '#FB923C',
  success: '#22C55E',
  error: '#EF4444',
  warning: '#FBBF24',
};

// ── Format helpers ──
export function formatTime(seconds) {
  if (seconds == null || isNaN(seconds)) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function parseTime(str) {
  if (str == null) return 0;
  const s = String(str).trim();
  if (s.includes(':')) {
    const parts = s.split(':');
    return (parseInt(parts[0]) || 0) * 60 + (parseFloat(parts[1]) || 0);
  }
  return parseFloat(s) || 0;
}

export function formatTimePrecise(seconds) {
  if (seconds == null || isNaN(seconds)) return '00:00.000';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
}

export function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

export function formatSpeed(bytesPerSec) {
  if (!bytesPerSec) return '0 B/s';
  return `${formatBytes(bytesPerSec)}/s`;
}

export function formatDuration(seconds) {
  if (!seconds) return '\u2014';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

// ── HTML escaping ──
// Use when interpolating untrusted strings (user filenames, server messages,
// log lines, error text) into innerHTML/template strings. Trusted UI labels
// don't need it. Returns '' for null/undefined so callers can chain freely.
const _ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => _ESCAPE_MAP[c]);
}

// ── API helper ──
export async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// ── API endpoint map ──
// Single source of truth so backend route renames don't require grepping.
// Strings are paths relative to /api (matching api()); leaves are either
// literal strings or functions that accept identifiers and return a path.
// SSE URLs include /api because new SSEClient() takes the absolute path.
const _q = (params) => {
  const e = Object.entries(params).filter(([, v]) => v != null && v !== '');
  return e.length ? '?' + e.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&') : '';
};
export const API = {
  jobs: {
    list: '/jobs',
    activeCount: '/jobs/active-count',
    get: id => `/jobs/${id}`,
    cancel: id => `/jobs/${id}/cancel`,
    eventsSSE: id => `/api/jobs/${id}/events`,
  },
  system: {
    stats: '/system/stats',
    videos: (params = {}) => `/system/videos${_q(params)}`,
    vllmStart: '/system/vllm/start',
    vllmStop: '/system/vllm/stop',
    vllmStatus: '/system/vllm/status',
  },
  upload: {
    start: '/upload/start',
    status: '/upload/status',
    download: '/upload/download',
    deleteLocal: '/upload/delete-local',
    deleteR2: '/upload/delete-r2',
    files: category => `/upload/files?category=${encodeURIComponent(category)}`,
    r2Files: category => `/upload/r2-files?category=${encodeURIComponent(category)}`,
  },
  download: {
    start: '/download/start',
    playlist: url => `/download/playlist?url=${encodeURIComponent(url)}`,
    cancel: sessionId => `/download/${sessionId}/cancel`,
    progressSSE: sessionId => `/api/download/${sessionId}/progress`,
  },
  cut: {
    videos: '/cut/videos',
    export: '/cut/export',
    video: name => `/cut/video/${encodeURIComponent(name)}`,
  },
  detect: {
    start: '/detect/start',
    convert: '/detect/convert',
  },
  annotate: {
    stats: '/annotate/stats',
    results: '/annotate/results',
    annotations: '/annotate/annotations',
    result: name => `/annotate/results/${encodeURIComponent(name)}`,
  },
  review: {
    results: '/review/results',
    annotations: '/review/annotations',
    result: (name, params = {}) => `/review/results/${encodeURIComponent(name)}${_q(params)}`,
  },
  predict: {
    videos: '/predict/videos',
    start: '/predict/start',
    results: '/predict/results',
    result: name => `/predict/results/${encodeURIComponent(name)}`,
  },
  train: {
    configDefaults: '/train/config-defaults',
    convertAnnotations: '/train/convert-annotations',
    extractFeatures: '/train/extract-features',
    start: '/train/start',
    status: (params = {}) => `/train/status${_q(params)}`,
    performance: (params = {}) => `/train/performance${_q(params)}`,
    checkpoints: (params = {}) => `/train/checkpoints${_q(params)}`,
  },
  vlm: {
    status: '/vlm/status',
    buildManifest: '/vlm/build-manifest',
    start: '/vlm/start',
    performance: '/vlm/performance',
  },
};

// ── SSE Client ──
// Survives transient disconnects: when the browser suspends EventSource (e.g.
// on mobile when the tab/app is backgrounded), the client retries with
// exponential backoff (1s → 2s → 5s → 10s, capped at 30s) and reconnects
// immediately when the page becomes visible again. Callers must call
// `stop()` exactly when they no longer want updates (terminal job state,
// page deactivate, etc.) — that is treated as a permanent close.
const _aliveClients = new Set();
const _BACKOFF_STEPS_MS = [1000, 2000, 5000, 10000, 30000];

export class SSEClient {
  constructor(url, handlers = {}) {
    this.url = url;
    this.handlers = handlers;
    this.source = null;
    this._alive = false;
    this._retry = 0;
    this._retryTimer = null;
  }

  start() {
    this._alive = true;
    _aliveClients.add(this);
    this._open();
    return this;
  }

  _open() {
    if (!this._alive) return;
    if (this.source) { this.source.close(); this.source = null; }
    this.source = new EventSource(this.url);
    this.source.onmessage = (e) => {
      this._retry = 0;  // reset backoff on any successful frame
      try {
        const data = JSON.parse(e.data);
        if (this.handlers.onMessage) this.handlers.onMessage(data);
      } catch { /* ignore parse errors */ }
    };
    this.source.onerror = () => {
      if (!this._alive) return;
      // Don't notify caller on transient errors — only when we actually give
      // up. Schedule a reconnect; the visibilitychange listener may also
      // trigger one sooner.
      this._scheduleReconnect();
    };
  }

  _scheduleReconnect() {
    if (!this._alive || this._retryTimer) return;
    if (this.source) { this.source.close(); this.source = null; }
    const delay = _BACKOFF_STEPS_MS[Math.min(this._retry, _BACKOFF_STEPS_MS.length - 1)];
    this._retry += 1;
    this._retryTimer = setTimeout(() => {
      this._retryTimer = null;
      this._open();
    }, delay);
  }

  // Force an immediate reconnect attempt (e.g. when the page returns to the
  // foreground after a long background period).
  _kick() {
    if (!this._alive) return;
    if (this._retryTimer) { clearTimeout(this._retryTimer); this._retryTimer = null; }
    this._retry = 0;
    if (!this.source || this.source.readyState === 2 /* CLOSED */) {
      this._open();
    }
  }

  stop() {
    this._alive = false;
    _aliveClients.delete(this);
    if (this._retryTimer) { clearTimeout(this._retryTimer); this._retryTimer = null; }
    if (this.source) { this.source.close(); this.source = null; }
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      _aliveClients.forEach(c => c._kick());
    }
  });
}

// ── Page Header ──
export function pageHeader(title, subtitle, actions = '') {
  return `
    <div class="flex items-start justify-between gap-4 mb-6">
      <div>
        <h2 class="text-xl font-heading font-bold text-text-primary tracking-tight">${title}</h2>
        <p class="text-text-muted text-sm mt-1">${subtitle}</p>
      </div>
      ${actions ? `<div class="flex items-center gap-2 flex-shrink-0">${actions}</div>` : ''}
    </div>`;
}

// ── Status Badge ──
export function createStatusBadge(status) {
  const map = {
    running:   'bg-indigo-500/15 text-indigo-400 ring-1 ring-indigo-500/20',
    completed: 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20',
    failed:    'bg-red-500/15 text-red-400 ring-1 ring-red-500/20',
    cancelled: 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20',
    pending:   'bg-white/5 text-text-muted ring-1 ring-white/10',
    stopped:   'bg-white/5 text-text-muted ring-1 ring-white/10',
  };
  const dotMap = {
    running: 'animate-pulse-dot',
    completed: '',
    failed: '',
  };
  const cls = map[status] || map.pending;
  const dotAnim = dotMap[status] || '';
  return `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium ${cls}">
    <span class="w-1.5 h-1.5 rounded-full bg-current ${dotAnim}"></span>${status}
  </span>`;
}

// ── Progress Bar ──
export function createProgressBar(progress, variant = 'primary') {
  const pct = Math.round((progress || 0) * 100);
  const gradients = {
    primary: 'background: linear-gradient(90deg, #4F46E5, #818CF8)',
    accent:  'background: linear-gradient(90deg, #EA580C, #FB923C)',
    success: 'background: linear-gradient(90deg, #16A34A, #22C55E)',
  };
  const glows = {
    primary: 'box-shadow: 0 0 10px rgba(99,102,241,0.35)',
    accent:  'box-shadow: 0 0 10px rgba(249,115,22,0.35)',
    success: 'box-shadow: 0 0 10px rgba(34,197,94,0.35)',
  };
  return `
    <div class="w-full bg-white/[0.06] rounded-full h-1.5 overflow-hidden">
      <div class="h-full rounded-full transition-all duration-500 ease-out relative" style="width: ${pct}%; ${gradients[variant] || gradients.primary}; ${pct > 3 ? (glows[variant] || glows.primary) : ''}">
        ${pct > 5 ? '<div class="absolute inset-0 shimmer-bg rounded-full"></div>' : ''}
      </div>
    </div>`;
}

// ── Toast Notifications ──
let _toastCount = 0;
export function showToast(message, type = 'info') {
  const icons = {
    info:    '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    success: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    error:   '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    warning: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>',
  };
  const colors = {
    info:    'border-indigo-500/25 text-indigo-300',
    success: 'border-emerald-500/25 text-emerald-300',
    error:   'border-red-500/25 text-red-300',
    warning: 'border-amber-500/25 text-amber-300',
  };
  const bgs = {
    info:    'rgba(99,102,241,0.06)',
    success: 'rgba(34,197,94,0.06)',
    error:   'rgba(239,68,68,0.06)',
    warning: 'rgba(251,191,36,0.06)',
  };
  const n = _toastCount++;
  const toast = document.createElement('div');
  toast.className = `fixed right-4 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl border ${colors[type] || colors.info} backdrop-blur-xl shadow-2xl text-sm font-medium`;
  toast.style.cssText = `bottom: ${16 + (n % 4) * 56}px; transform: translateX(120%); opacity: 0; transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1); background: ${bgs[type] || bgs.info}; backdrop-filter: blur(16px);`;
  toast.innerHTML = `${icons[type] || icons.info}<span>${message}</span>
    <button class="ml-2 opacity-50 hover:opacity-100 cursor-pointer transition-opacity" aria-label="Dismiss">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
    </button>`;
  document.body.appendChild(toast);
  const dismiss = () => {
    toast.style.transform = 'translateX(120%)'; toast.style.opacity = '0';
    setTimeout(() => { toast.remove(); _toastCount = Math.max(0, _toastCount - 1); }, 300);
  };
  toast.querySelector('button').addEventListener('click', dismiss);
  requestAnimationFrame(() => { toast.style.transform = 'translateX(0)'; toast.style.opacity = '1'; });
  setTimeout(dismiss, 3500);
}

// ── Modal dialog ──
// showConfirm({ title, body, confirmText, cancelText, variant }) → Promise<boolean>
// variant: 'warning' (amber) | 'danger' (red) | 'info' (indigo)
export function showConfirm({
  title = 'Confirm',
  body = '',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'warning',
} = {}) {
  const variants = {
    info:    { ring: 'ring-indigo-500/25', glow: 'rgba(99,102,241,0.12)', iconBg: 'bg-indigo-500/15 text-indigo-300 ring-indigo-500/25',
               svg: '<path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>' },
    warning: { ring: 'ring-amber-500/25', glow: 'rgba(251,191,36,0.10)', iconBg: 'bg-amber-500/15 text-amber-300 ring-amber-500/25',
               svg: '<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>' },
    danger:  { ring: 'ring-red-500/25', glow: 'rgba(239,68,68,0.12)', iconBg: 'bg-red-500/15 text-red-300 ring-red-500/25',
               svg: '<path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>' },
  };
  const v = variants[variant] || variants.warning;

  return new Promise(resolve => {
    const overlay = document.createElement('div');
    overlay.className = 'fixed inset-0 z-[60] flex items-center justify-center p-4';
    overlay.style.cssText = 'background: rgba(0,0,0,0.55); backdrop-filter: blur(8px); opacity: 0; transition: opacity 0.2s ease;';

    overlay.innerHTML = `
      <div class="relative w-full max-w-md rounded-2xl border border-white/10 ring-1 ${v.ring} p-6 shadow-2xl"
           style="background: linear-gradient(180deg, rgba(20,20,26,0.98), rgba(12,12,16,0.98)); backdrop-filter: blur(20px); box-shadow: 0 24px 64px -12px ${v.glow}, 0 0 0 1px rgba(255,255,255,0.03); opacity: 0; transform: translateY(8px) scale(0.98); transition: all 0.22s cubic-bezier(0.34, 1.56, 0.64, 1);">
        <div class="flex items-start gap-4">
          <div class="w-10 h-10 rounded-xl flex items-center justify-center ring-1 flex-shrink-0 ${v.iconBg}">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">${v.svg}</svg>
          </div>
          <div class="flex-1 min-w-0">
            <h3 class="text-base font-heading font-semibold text-text-primary">${title}</h3>
            <div class="mt-2 text-sm text-text-secondary leading-relaxed whitespace-pre-line">${body}</div>
          </div>
        </div>
        <div class="mt-6 flex items-center justify-end gap-2.5">
          <button data-act="cancel" class="inline-flex items-center justify-center gap-2 bg-white/[0.04] hover:bg-white/[0.08] text-text-secondary hover:text-text-primary border border-white/10 hover:border-white/20 px-4 py-2 rounded-xl font-medium text-sm cursor-pointer transition-all duration-200">${cancelText}</button>
          <button data-act="confirm" class="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm text-white cursor-pointer transition-all duration-200 hover:-translate-y-px active:translate-y-0 hover:shadow-lg"
                  style="${variant === 'danger'
                    ? 'background: linear-gradient(135deg, #EF4444, #DC2626); box-shadow: 0 2px 12px rgba(239,68,68,0.25)'
                    : 'background: linear-gradient(135deg, #F97316, #EA580C); box-shadow: 0 2px 12px rgba(249,115,22,0.25)'}">${confirmText}</button>
        </div>
      </div>`;

    document.body.appendChild(overlay);
    const dialogEl = overlay.firstElementChild;

    const close = (result) => {
      dialogEl.style.opacity = '0';
      dialogEl.style.transform = 'translateY(8px) scale(0.98)';
      overlay.style.opacity = '0';
      document.removeEventListener('keydown', onKey);
      setTimeout(() => { overlay.remove(); resolve(result); }, 200);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); close(false); }
      else if (e.key === 'Enter') { e.preventDefault(); close(true); }
    };

    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(false); });
    overlay.querySelector('[data-act="cancel"]').addEventListener('click', () => close(false));
    overlay.querySelector('[data-act="confirm"]').addEventListener('click', () => close(true));
    document.addEventListener('keydown', onKey);

    requestAnimationFrame(() => {
      overlay.style.opacity = '1';
      dialogEl.style.opacity = '1';
      dialogEl.style.transform = 'translateY(0) scale(1)';
      overlay.querySelector('[data-act="confirm"]').focus();
    });
  });
}

// ── Empty State ──
export function emptyState(icon, title, subtitle = '') {
  return `
    <div class="flex flex-col items-center justify-center py-12 text-center">
      <div class="w-12 h-12 rounded-2xl bg-surface-200 border border-border flex items-center justify-center text-text-muted mb-4">${icon}</div>
      <p class="text-text-secondary text-sm font-medium">${title}</p>
      ${subtitle ? `<p class="text-text-muted text-xs mt-1.5 max-w-xs">${subtitle}</p>` : ''}
    </div>`;
}

// ── Skeleton Loading ──
export function skeleton(lines = 3, type = 'text') {
  if (type === 'card') {
    return `<div class="space-y-4 animate-pulse">
      ${Array(lines).fill(0).map(() => `
        <div class="rounded-2xl bg-surface-100 border border-border p-5 space-y-3">
          <div class="h-4 bg-surface-200 rounded w-1/3"></div>
          <div class="h-3 bg-surface-200 rounded w-2/3"></div>
          <div class="h-3 bg-surface-200 rounded w-1/2"></div>
        </div>`).join('')}
    </div>`;
  }
  return `<div class="space-y-3 animate-pulse">
    ${Array(lines).fill(0).map((_, i) => `<div class="h-3 bg-surface-200 rounded" style="width: ${85 - i * 15}%"></div>`).join('')}
  </div>`;
}

// ── Card wrapper ──
export function card(content, extraClass = '') {
  return `<div class="glass-card rounded-2xl p-5 transition-colors duration-200 ${extraClass}">${content}</div>`;
}

// ── Stat Card ──
export function statCard(label, value, ok = true) {
  return `
    <div class="rounded-xl p-3 text-center border transition-colors duration-200 ${ok
      ? 'border-emerald-500/15 bg-emerald-500/[0.04]'
      : 'border-border bg-surface-50/50'}">
      <div class="text-base font-heading font-bold ${ok ? 'text-emerald-400' : 'text-text-muted'}">${value}</div>
      <div class="text-[11px] text-text-muted mt-0.5">${label}</div>
    </div>`;
}

// ── Step Badge ──
export function stepBadge(number, variant = 'primary') {
  const styles = {
    primary: 'background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(99,102,241,0.05)); color: #818CF8; border: 1px solid rgba(99,102,241,0.15)',
    accent: 'background: linear-gradient(135deg, rgba(249,115,22,0.20), rgba(249,115,22,0.05)); color: #FB923C; border: 1px solid rgba(249,115,22,0.18)',
  };
  return `<span class="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold font-heading flex-shrink-0" style="${styles[variant] || styles.primary}">${number}</span>`;
}

// ── Button builders ──
export function btnPrimary(text, attrs = '') {
  return `<button class="relative inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm text-white cursor-pointer transition-all duration-200 hover:-translate-y-px active:translate-y-0 hover:shadow-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none" style="background: linear-gradient(135deg, #F97316, #EA580C); box-shadow: 0 2px 12px rgba(249,115,22,0.25)" ${attrs}>${text}</button>`;
}

export function btnSecondary(text, attrs = '') {
  return `<button class="inline-flex items-center justify-center gap-2 bg-white/[0.06] hover:bg-white/[0.10] text-text-primary border border-border-light hover:border-border-bright px-5 py-2.5 rounded-xl font-semibold text-sm cursor-pointer transition-all duration-200 hover:-translate-y-px active:translate-y-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary disabled:opacity-40 disabled:cursor-not-allowed" ${attrs}>${text}</button>`;
}

export function btnDanger(text, attrs = '') {
  return `<button class="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm text-white cursor-pointer transition-all duration-200 hover:-translate-y-px active:translate-y-0 hover:shadow-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-red-500 disabled:opacity-40 disabled:cursor-not-allowed" style="background: linear-gradient(135deg, #EF4444, #DC2626); box-shadow: 0 2px 12px rgba(239,68,68,0.25)" ${attrs}>${text}</button>`;
}

export function btnSmall(text, attrs = '', variant = 'default') {
  const variants = {
    default: 'bg-white/[0.06] text-text-secondary hover:text-text-primary hover:bg-white/[0.10] border border-border hover:border-border-light',
    primary: 'bg-primary/10 text-primary-light hover:bg-primary/20 border border-primary/15 hover:border-primary/25',
    danger:  'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/15 hover:border-red-500/25',
    success: 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/15 hover:border-emerald-500/25',
  };
  return `<button class="${variants[variant] || variants.default} px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary disabled:opacity-40 disabled:cursor-not-allowed" ${attrs}>${text}</button>`;
}

// ── Input classes ──
export const inputCls = 'bg-surface-100 text-text-primary border border-border-light rounded-xl px-3.5 py-2.5 text-sm transition-all duration-200 focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15 placeholder:text-text-muted';
export const selectCls = inputCls + ' cursor-pointer appearance-none';

// ── Select builder ──
export function selectInput(id, options, attrs = '') {
  const opts = options.map(o => {
    const val = typeof o === 'string' ? o : o.value;
    const label = typeof o === 'string' ? o : o.label;
    return `<option value="${val}">${label}</option>`;
  }).join('');
  return `<select id="${id}" class="${selectCls}" ${attrs}>${opts}</select>`;
}

// ── Keyboard Shortcut Hint ──
export function kbdHint(shortcuts) {
  return `<div class="flex items-center gap-4 text-[11px] text-text-muted px-1">
    <span class="font-heading text-text-secondary">Keys:</span>
    ${shortcuts.map(s => `<span class="inline-flex items-center gap-1"><kbd class="px-1.5 py-0.5 rounded bg-surface-200 border border-border text-[10px] font-heading text-text-secondary">${s[0]}</kbd> ${s[1]}</span>`).join('')}
  </div>`;
}

// ── Section Divider ──
export function sectionTitle(title, subtitle = '', actions = '') {
  return `<div class="flex items-center justify-between">
    <div>
      <h3 class="text-sm font-heading font-semibold text-text-primary">${title}</h3>
      ${subtitle ? `<p class="text-[11px] text-text-muted mt-0.5">${subtitle}</p>` : ''}
    </div>
    ${actions ? `<div class="flex items-center gap-2">${actions}</div>` : ''}
  </div>`;
}

// ── Cut-kind tabs ──
// All / Broadcast / Sideline filter, used on Predict + Train (and any future
// page that lists cuts). The `prefix` namespaces buttons so multiple sets of
// tabs can coexist on the same page if needed.
export function kindTabs(prefix) {
  const tab = (k, label) => `<button type="button" data-prefix="${prefix}" data-kind="${k}"
      class="kind-tab px-3 py-1 text-xs font-heading rounded-md transition-colors duration-150"
      aria-pressed="${k === 'all' ? 'true' : 'false'}">${label} <span class="opacity-60 ml-1" data-prefix="${prefix}" data-count="${k}">0</span></button>`;
  return `<div class="inline-flex rounded-lg border border-border bg-surface-100 p-0.5" role="tablist" aria-label="Cut kind">
      ${tab('all', 'All')}${tab('broadcast', 'Broadcast')}${tab('sideline', 'Sideline')}
    </div>`;
}

export function updateKindTabs(prefix, kindFilter, list) {
  const counts = { all: list.length, broadcast: 0, sideline: 0 };
  for (const v of list) {
    if (v.kind === 'broadcast') counts.broadcast++;
    else if (v.kind === 'sideline') counts.sideline++;
  }
  document.querySelectorAll(`.kind-tab[data-prefix="${prefix}"]`).forEach(btn => {
    const k = btn.dataset.kind;
    const active = k === kindFilter;
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    btn.classList.toggle('bg-primary', active);
    btn.classList.toggle('text-white', active);
    btn.classList.toggle('text-text-secondary', !active);
    btn.classList.toggle('hover:bg-white/[0.04]', !active);
    const cnt = btn.querySelector(`[data-count="${k}"]`);
    if (cnt) cnt.textContent = counts[k];
  });
}

// ── Video status badges ──
// Small composable helpers used by Predict / Train / Review video lists. Pages
// compose only the badges they care about (e.g. Predict shows the missing-
// features pill explicitly while Train hides it).
export const badges = {
  annotated: () => '<span title="Annotated">✅</span>',
  preAnnotated: () => '<span title="Pre-annotation">⚡</span>',
  predicted: () => '<span title="Predicted">🤖</span>',
  hasFeatures: () =>
    '<span title="Features extracted" class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>features</span>',
  noFeatures: () =>
    '<span title="No features for selected model" class="inline-flex items-center gap-1.5 text-[11px] text-amber-400/80 bg-amber-500/10 ring-1 ring-amber-500/20 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>no features</span>',
  predictedPill: () =>
    '<span class="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-500/10 ring-1 ring-emerald-500/20 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>predicted</span>',
  pendingPill: () =>
    '<span class="inline-flex items-center gap-1.5 text-[11px] text-text-muted bg-white/5 ring-1 ring-white/10 px-2.5 py-0.5 rounded-full font-medium"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>pending</span>',
};

// ── Job progress rendering ──
// Single source of truth for "background job → progress row" UI used on
// Predict / Detect / Upload. Train uses a bespoke layout and stays separate.
export function getJobStatusColor(status) {
  if (status === 'running') return 'text-primary-light';
  if (status === 'completed') return 'text-emerald-400';
  if (status === 'failed') return 'text-red-400';
  if (status === 'cancelled') return 'text-amber-400';
  return 'text-text-muted';
}

export function getJobStatusLabel(job) {
  const pct = Math.round((job.progress || 0) * 100);
  if (job.status === 'failed') return 'failed';
  if (job.status === 'cancelled') return 'cancelled';
  if (job.status === 'completed') return job.message?.includes('failed') ? 'partial' : 'done';
  return pct + '%';
}

// Render a single job's progress row.
//   detail: optional HTML appended below the bar (e.g. bytes/ETA)
//   showLogs: collapse a `<details>` with job.logs when status is failed/partial
//   truncateMsg: add `truncate` class to message/error (default true)
export function renderJobProgress(job, { detail = '', showLogs = false, truncateMsg = true } = {}) {
  const color = getJobStatusColor(job.status);
  const label = getJobStatusLabel(job);
  const isRunning = job.status === 'running';
  const isDone = job.status === 'completed';
  const isFailed = job.status === 'failed';
  const showMessage = job.message && (isRunning || isDone || isFailed);
  const trunc = truncateMsg ? ' truncate' : '';
  const hasLogs = Array.isArray(job.logs) && job.logs.length > 0;
  const showLogsBlock = showLogs && hasLogs && (isFailed || (isDone && job.message?.includes('failed')));
  const logsHtml = showLogsBlock
    ? `<details class="mt-1">
         <summary class="text-[10px] text-text-muted cursor-pointer hover:text-text-primary">Show logs (${job.logs.length} lines)</summary>
         <pre class="mt-1 max-h-64 overflow-y-auto rounded-lg bg-black/40 border border-white/5 p-2 font-mono text-[10px] text-red-300/80 whitespace-pre-wrap break-words">${job.logs.map(escapeHtml).join('\n')}</pre>
       </details>`
    : '';
  return `
      <div class="space-y-1.5">
        <div class="flex items-center justify-between">
          <span class="text-xs text-text-primary font-medium truncate">${escapeHtml(job.name)}</span>
          <span class="text-[11px] ${color} tabular-nums font-medium">${label}</span>
        </div>
        ${createProgressBar(job.progress)}
        ${detail ? `<div class="text-[11px] text-text-muted tabular-nums">${detail}</div>` : ''}
        ${showMessage ? `<p class="text-[10px] text-text-muted${trunc}">${escapeHtml(job.message)}</p>` : ''}
        ${job.error ? `<p class="text-[10px] text-red-400/80${trunc}">${escapeHtml(job.error)}</p>` : ''}
        ${logsHtml}
      </div>`;
}

// ── Sidebar state ──
let _pollInterval = null;

export function startSidebarPolling() {
  if (_pollInterval) return;
  pollSidebar();
  _pollInterval = setInterval(pollSidebar, 30000);
}

async function pollSidebar() {
  try {
    const [vllm, stats, jobs] = await Promise.all([
      api(API.system.vllmStatus),
      api(API.system.stats),
      api(API.jobs.activeCount),
    ]);

    // vLLM indicator
    const dot = document.getElementById('vllm-dot');
    const label = document.getElementById('vllm-label');
    if (dot && label) {
      const statusMap = {
        running:  ['bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]', 'vLLM: running'],
        starting: ['bg-amber-400 animate-pulse', 'vLLM: starting...'],
        stopped:  ['bg-text-muted', 'vLLM: stopped'],
        error:    ['bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.5)]', 'vLLM: error'],
      };
      const [cls, txt] = statusMap[vllm.status] || statusMap.stopped;
      dot.className = `w-2 h-2 rounded-full flex-shrink-0 ring-2 ring-surface-100 ${cls}`;
      label.textContent = txt;
    }

    // Stats
    const statsEl = document.getElementById('sidebar-stats');
    if (statsEl) {
      const items = [
        ['Videos', stats.videos],
        ['Cuts', stats.cuts],
        ['VLM-Pred', stats.pre_annotations],
        ['Annotations', stats.annotations],
        ['VJEPA-B', stats.vjepa_b],
        ['VJEPA-L', stats.vjepa_l],
        ['TAD-Pred', stats.predictions],
      ];
      statsEl.innerHTML = items.map(([k, v]) => `
        <div class="flex justify-between items-center">
          <span>${k}</span>
          <span class="font-heading text-text-secondary">${v || 0}</span>
        </div>
      `).join('');
    }

    // Job count badge
    const badge = document.getElementById('sidebar-job-count');
    if (badge) {
      const count = jobs.count || 0;
      badge.textContent = count;
      badge.classList.toggle('hidden', count === 0);
    }
  } catch { /* silently fail */ }
}
