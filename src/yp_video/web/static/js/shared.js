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

// ── SSE Client ──
export class SSEClient {
  constructor(url, handlers = {}) {
    this.url = url;
    this.handlers = handlers;
    this.source = null;
  }

  start() {
    this.source = new EventSource(this.url);
    this.source.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (this.handlers.onMessage) this.handlers.onMessage(data);
      } catch { /* ignore parse errors */ }
    };
    this.source.onerror = () => {
      if (this.handlers.onError) this.handlers.onError();
      this.stop();
    };
    return this;
  }

  stop() {
    if (this.source) {
      this.source.close();
      this.source = null;
    }
  }
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

// ── Sidebar state ──
let _pollInterval = null;

export function startSidebarPolling() {
  if (_pollInterval) return;
  pollSidebar();
  _pollInterval = setInterval(pollSidebar, 10000);
}

async function pollSidebar() {
  try {
    const [vllm, stats, jobs] = await Promise.all([
      api('/system/vllm/status'),
      api('/system/stats'),
      api('/jobs/active-count'),
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
        ['Annotations', stats.annotations],
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
