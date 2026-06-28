import { useEffect, useSyncExternalStore } from 'react';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { getConfirm, settleConfirm, subscribeConfirm, type ConfirmVariant } from './confirm';

const ICON: Record<ConfirmVariant, { ring: string; icon: string; path: string }> = {
  info: { ring: 'ring-primary/30', icon: 'text-primary-light', path: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
  warning: {
    ring: 'ring-amber-500/25',
    icon: 'text-amber-300',
    path: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  },
  danger: { ring: 'ring-red-500/25', icon: 'text-red-300', path: 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
};

/** Mounted once at the app root. Renders the active confirm() request. */
export function ConfirmDialog() {
  const active = useSyncExternalStore(subscribeConfirm, getConfirm);

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') settleConfirm(false);
      else if (e.key === 'Enter') settleConfirm(true);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [active]);

  if (!active) return null;
  const variant = active.variant ?? 'warning';
  const v = ICON[variant];

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)' }}
      onClick={(e) => e.target === e.currentTarget && settleConfirm(false)}
    >
      <div className={cn('w-full max-w-md rounded-2xl border border-border bg-surface-100 p-6 shadow-2xl ring-1', v.ring)}>
        <div className="flex items-start gap-4">
          <div className={cn('flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl ring-1', v.ring, v.icon)}>
            <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d={v.path} />
            </svg>
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="font-heading text-base font-semibold text-text-primary">{active.title}</h3>
            {active.body && <div className="mt-2 whitespace-pre-line text-sm leading-relaxed text-text-secondary">{active.body}</div>}
          </div>
        </div>
        <div className="mt-6 flex items-center justify-end gap-2.5">
          <Button onClick={() => settleConfirm(false)}>{active.cancelText ?? 'Cancel'}</Button>
          <Button intent={variant === 'danger' ? 'danger' : 'primary'} onClick={() => settleConfirm(true)}>
            {active.confirmText ?? 'Confirm'}
          </Button>
        </div>
      </div>
    </div>
  );
}
