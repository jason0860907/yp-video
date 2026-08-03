import { useQuery } from '@tanstack/react-query';
import { NavLink, useLocation } from 'react-router-dom';
import { API, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { usePresence } from '@/lib/usePresence';
import { LabelProgress } from '@/components/labeling/LabelProgress';
import { Icon } from '@/components/ui/Icon';
import type { ActiveCount } from '@/types/api';
import { NAV, PATH_SECTION, type NavItem, type NavSection } from './nav';
import { useNavSections } from './useNavSections';

const HEADING =
  'flex w-full items-center gap-2 px-3 pb-1.5 pt-3 text-[10px] font-semibold uppercase tracking-widest text-text-muted';

function SectionHeading({
  section,
  expanded,
  onToggle,
}: {
  section: NavSection;
  expanded: boolean;
  onToggle: () => void;
}) {
  if (!section.collapsible) return <p className={HEADING}>{section.title}</p>;
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={expanded}
      className={cn(HEADING, 'rounded-lg transition-colors hover:text-text-secondary')}
    >
      <span>{section.title}</span>
      <svg
        className={cn('ml-auto h-3 w-3 transition-transform duration-150', expanded && 'rotate-90')}
        fill="none"
        stroke="currentColor"
        strokeWidth={2.2}
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
      </svg>
    </button>
  );
}

function NavRow({ item, badge }: { item: NavItem; badge?: number }) {
  return (
    <NavLink
      to={item.path}
      className={({ isActive }) =>
        cn(
          'flex items-center gap-3 rounded-lg border-l-2 px-3 py-2 text-sm transition-colors duration-150',
          isActive
            ? 'border-primary bg-gradient-to-r from-primary/20 to-transparent text-text-primary'
            : 'border-transparent text-text-secondary hover:bg-ink/5 hover:text-text-primary',
        )
      }
    >
      {({ isActive }) => (
        <>
          <Icon paths={item.icon} className={isActive ? 'text-primary-light' : undefined} />
          {item.label}
          {badge !== undefined && badge > 0 && (
            <span className="ml-auto flex h-5 w-5 items-center justify-center rounded-full bg-primary text-[10px] font-semibold text-on-primary">
              {badge}
            </span>
          )}
        </>
      )}
    </NavLink>
  );
}

export function Sidebar() {
  const jobs = useQuery({
    queryKey: ['jobs-active-count'],
    queryFn: () => apiFetch<ActiveCount>(API.jobs.activeCount),
    refetchInterval: 30_000,
  });
  const presence = usePresence();
  const { pathname } = useLocation();
  const { isOpen, toggle } = useNavSections(PATH_SECTION[pathname]);

  const jobCount = jobs.data?.count ?? 0;

  return (
    <aside className="flex h-full w-[212px] flex-shrink-0 flex-col border-r border-border bg-sidebar">
      {/* nav */}
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 pt-3">
        {NAV.map((section) => {
          const expanded = !section.collapsible || isOpen(section.title);
          return (
            <div key={section.title}>
              <SectionHeading
                section={section}
                expanded={expanded}
                onToggle={() => toggle(section.title)}
              />
              {expanded &&
                section.items.map((item) => (
                  <NavRow
                    key={item.path}
                    item={item}
                    badge={item.path === '/jobs' ? jobCount : undefined}
                  />
                ))}
            </div>
          );
        })}
      </nav>

      {/* footer: who is here + stats. vLLM lives on the Jobs page, where it
          can also be started — a read-only dot here was one more poll for a
          number nobody could act on. */}
      <div className="mx-2 mb-3 space-y-2.5 rounded-xl border border-border bg-surface-100 p-3">
        {presence && (
          <div
            className="flex items-center gap-2 text-xs text-text-muted"
            title="Browsers with this page in the foreground; active = input within the last 5 min"
          >
            <span
              className={cn(
                'h-2 w-2 flex-shrink-0 rounded-full ring-2 ring-surface-100',
                presence.active > 0 ? 'bg-sky-400 shadow-[0_0_6px_rgba(56,189,248,0.5)]' : 'bg-text-muted',
              )}
            />
            <span className="font-mono text-[11px]">
              Online: {presence.online} ({presence.active} active)
            </span>
          </div>
        )}
        <div className="h-px w-full bg-border" />
        <LabelProgress />
      </div>
    </aside>
  );
}
