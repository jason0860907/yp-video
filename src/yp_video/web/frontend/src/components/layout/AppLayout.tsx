import { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

const STORAGE_KEY = 'sidebarCollapsed';

/** App shell: collapsible sidebar + scrollable main outlet. */
export function AppLayout() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(STORAGE_KEY) === '1');

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
  }, [collapsed]);

  return (
    <div className="flex h-screen overflow-hidden">
      {collapsed && (
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          title="Show sidebar"
          aria-label="Show sidebar"
          className="fixed left-3 top-3 z-50 flex h-9 w-9 items-center justify-center rounded-lg border border-border-light bg-surface-100/90 text-text-secondary shadow-lg backdrop-blur transition-colors hover:bg-surface-200 hover:text-text-primary"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      )}

      <Sidebar collapsed={collapsed} onCollapse={() => setCollapsed(true)} />

      <main id="app-main" className="flex-1 overflow-y-auto px-6 py-6 lg:px-8 lg:py-7">
        <Outlet />
      </main>
    </div>
  );
}
