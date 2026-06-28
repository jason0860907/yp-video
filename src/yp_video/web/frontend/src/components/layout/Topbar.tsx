import { cn } from '@/lib/cn';
import { PALETTES, setPalette, toggleTheme, useTheme } from '@/lib/theme';

interface TopbarProps {
  onToggleSidebar: () => void;
}

/** Slim app bar — sidebar toggle, brand mark, workspace label, and the
 *  theme + palette controls. (No workspace switcher / teams: yp-video is the
 *  Pipeline workspace only.) */
export function Topbar({ onToggleSidebar }: TopbarProps) {
  const { theme, palette } = useTheme();

  return (
    <header className="z-30 flex h-[52px] flex-shrink-0 items-center gap-3.5 border-b border-border bg-surface-100 px-3.5">
      <button
        type="button"
        onClick={onToggleSidebar}
        title="Toggle sidebar"
        aria-label="Toggle sidebar"
        className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted transition-colors hover:bg-white/[0.06] hover:text-text-primary"
      >
        <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      <div className="flex items-center gap-2.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent">
          <svg className="h-4 w-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.4}>
            <circle cx="12" cy="12" r="9" />
            <path d="M3 12c3.5 1.5 5 4.5 5 9M21 12c-3.5 1.5-5 4.5-5 9M12 3c-2 3-2 6 0 9 2 3 2 6 0 9" />
          </svg>
        </div>
        <span className="font-mono text-[13.5px] font-bold tracking-tight text-text-primary">VolleyIQ</span>
        <span className="rounded-md border border-border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
          Pipeline
        </span>
      </div>

      <div className="flex-1" />

      {/* theme toggle */}
      <button
        type="button"
        onClick={toggleTheme}
        title="Toggle light / dark"
        aria-label="Toggle light / dark"
        className="flex h-[30px] w-[30px] items-center justify-center rounded-lg border border-border bg-surface-50 text-text-secondary transition-colors hover:text-text-primary"
      >
        {theme === 'dark' ? (
          <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
          </svg>
        ) : (
          <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="4" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32l1.41-1.41" />
          </svg>
        )}
      </button>

      {/* palette chips */}
      <div className="flex items-center gap-1.5 px-0.5">
        {PALETTES.map((p) => (
          <button
            key={p.key}
            type="button"
            onClick={() => setPalette(p.key)}
            title={p.label}
            aria-label={p.label}
            aria-pressed={palette === p.key}
            className={cn(
              'h-[15px] w-[15px] rounded-full transition-transform hover:scale-110',
              palette === p.key ? 'ring-2 ring-text-primary ring-offset-2 ring-offset-surface-100' : 'ring-1 ring-border',
            )}
            style={{ background: p.swatch }}
          />
        ))}
      </div>
    </header>
  );
}
