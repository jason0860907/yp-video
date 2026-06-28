import { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';

const STORAGE_KEY = 'sidebarCollapsed';

/** App shell: top bar + collapsible sidebar + scrollable main outlet. */
export function AppLayout() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(STORAGE_KEY) === '1');

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
    // Drives the "give the video more room when the sidebar is hidden" rule.
    document.body.classList.toggle('sidebar-collapsed', collapsed);
    return () => document.body.classList.remove('sidebar-collapsed');
  }, [collapsed]);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Topbar onToggleSidebar={() => setCollapsed((c) => !c)} />
      <div className="flex min-h-0 flex-1">
        {!collapsed && <Sidebar />}
        <main id="app-main" className="flex-1 overflow-y-auto px-6 py-6 lg:px-8 lg:py-7">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
