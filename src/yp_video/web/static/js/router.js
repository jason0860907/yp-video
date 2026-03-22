/**
 * Hash-based SPA router for YP Video Pipeline.
 * Supports page transitions and skeleton loading.
 */
import { startSidebarPolling, skeleton } from './shared.js';

const pages = {};
let currentDestroy = null;

async function loadPage(name) {
  if (!pages[name]) {
    try {
      const mod = await import(`./pages/${name}.js`);
      pages[name] = mod;
    } catch (e) {
      console.error(`Failed to load page: ${name}`, e);
      document.getElementById('app').innerHTML = `
        <div class="flex items-center justify-center h-full">
          <div class="text-center space-y-3">
            <div class="w-12 h-12 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400 mx-auto">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            </div>
            <p class="text-red-400 text-sm font-medium">Failed to load page: ${name}</p>
            <p class="text-text-muted text-xs">${e.message}</p>
          </div>
        </div>`;
      return;
    }
  }
  return pages[name];
}

async function navigate() {
  const hash = window.location.hash || '#/download';
  const pageName = hash.replace('#/', '') || 'download';

  // Destroy previous page
  if (currentDestroy) {
    currentDestroy();
    currentDestroy = null;
  }

  // Update sidebar active state
  document.querySelectorAll('.sidebar-link').forEach(link => {
    const isActive = link.dataset.page === pageName;
    link.classList.toggle('active', isActive);
    link.classList.toggle('text-text-primary', isActive);
    link.classList.toggle('text-text-secondary', !isActive);
  });

  // Show skeleton loading
  const container = document.getElementById('app');
  container.innerHTML = `<div class="max-w-4xl mx-auto space-y-6 pt-2">${skeleton(3, 'card')}</div>`;

  // Load and render page with transition
  const mod = await loadPage(pageName);
  if (mod) {
    container.innerHTML = '';
    container.classList.add('page-enter');
    mod.render(container);
    if (mod.destroy) currentDestroy = mod.destroy;

    // Remove transition class after animation
    const onEnd = () => {
      container.classList.remove('page-enter');
      container.removeEventListener('animationend', onEnd);
    };
    container.addEventListener('animationend', onEnd);
  }
}

// Init
window.addEventListener('hashchange', navigate);
window.addEventListener('DOMContentLoaded', () => {
  navigate();
  startSidebarPolling();
});

// If already loaded (module scripts are deferred)
if (document.readyState !== 'loading') {
  navigate();
  startSidebarPolling();
}
