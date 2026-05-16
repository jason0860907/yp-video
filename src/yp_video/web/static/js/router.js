/**
 * Hash-based SPA router with keep-alive pages.
 *
 * Pages are rendered once and kept in the DOM. Navigation toggles
 * visibility and calls activate/deactivate lifecycle hooks.
 */
import { startSidebarPolling, initSidebarToggle, skeleton } from './shared.js';

const pages = {};       // module cache: { name: mod }
const containers = {};  // DOM containers: { name: HTMLDivElement }
let activePage = null;
let _navigating = false;

async function loadPage(name) {
  if (!pages[name]) {
    try {
      const mod = await import(`./pages/${name}.js`);
      pages[name] = mod;
    } catch (e) {
      console.error(`Failed to load page: ${name}`, e);
      return null;
    }
  }
  return pages[name];
}

async function navigate() {
  const hash = window.location.hash || '#/download';
  const pageName = hash.replace('#/', '') || 'download';

  if (activePage === pageName || _navigating) return;
  _navigating = true;

  // Deactivate + hide current page
  if (activePage) {
    pages[activePage]?.deactivate?.();
    if (containers[activePage]) containers[activePage].style.display = 'none';
  }

  // Update sidebar active state
  document.querySelectorAll('.sidebar-link').forEach(link => {
    const isActive = link.dataset.page === pageName;
    link.classList.toggle('active', isActive);
    link.classList.toggle('text-text-primary', isActive);
    link.classList.toggle('text-text-secondary', !isActive);
  });

  const app = document.getElementById('app');

  if (!containers[pageName]) {
    // First visit: show skeleton, load module, render
    const skeletonEl = document.createElement('div');
    skeletonEl.innerHTML = `<div class="max-w-4xl mx-auto space-y-6 pt-2">${skeleton(3, 'card')}</div>`;
    app.appendChild(skeletonEl);

    const mod = await loadPage(pageName);
    skeletonEl.remove();

    if (!mod) {
      app.innerHTML = `
        <div class="flex items-center justify-center h-full">
          <div class="text-center space-y-3">
            <div class="w-12 h-12 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400 mx-auto">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            </div>
            <p class="text-red-400 text-sm font-medium">Failed to load page: ${pageName}</p>
          </div>
        </div>`;
      activePage = null;
      _navigating = false;
      return;
    }

    const container = document.createElement('div');
    container.dataset.page = pageName;
    container.classList.add('page-enter');
    app.appendChild(container);

    const onEnd = () => {
      container.classList.remove('page-enter');
      container.removeEventListener('animationend', onEnd);
    };
    container.addEventListener('animationend', onEnd);

    // render() builds DOM + calls activate() internally
    mod.render(container);
    containers[pageName] = container;
  } else {
    // Subsequent visit: show + re-activate
    containers[pageName].style.display = '';
    pages[pageName]?.activate?.();
  }

  activePage = pageName;
  _navigating = false;
}

// Init
window.addEventListener('hashchange', navigate);
window.addEventListener('DOMContentLoaded', () => {
  navigate();
  startSidebarPolling();
  initSidebarToggle();
});

if (document.readyState !== 'loading') {
  navigate();
  startSidebarPolling();
  initSidebarToggle();
}
