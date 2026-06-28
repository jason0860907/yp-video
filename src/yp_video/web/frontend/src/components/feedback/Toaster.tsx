import { useSyncExternalStore } from 'react';
import { cn } from '@/lib/cn';
import { dismissToast, getToasts, subscribeToasts, type ToastType } from './toast';

const TONE: Record<ToastType, { cls: string; bg: string; icon: string }> = {
  info: {
    cls: 'border-primary/30 text-primary-light',
    bg: 'rgba(45,95,63,0.12)',
    icon: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  },
  success: {
    cls: 'border-emerald-500/25 text-emerald-300',
    bg: 'rgba(52,199,89,0.10)',
    icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  },
  error: {
    cls: 'border-red-500/25 text-red-300',
    bg: 'rgba(255,69,58,0.10)',
    icon: 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  },
  warning: {
    cls: 'border-amber-500/25 text-amber-300',
    bg: 'rgba(232,178,58,0.10)',
    icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  },
};

/** Mounted once at the app root. Renders the live toast stack bottom-right. */
export function Toaster() {
  const toasts = useSyncExternalStore(subscribeToasts, getToasts);

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[70] flex flex-col gap-2.5">
      {toasts.map((t) => {
        const tone = TONE[t.type];
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => dismissToast(t.id)}
            style={{ background: tone.bg, backdropFilter: 'blur(16px)' }}
            className={cn(
              'pointer-events-auto flex items-center gap-2.5 rounded-xl border px-4 py-3 text-left',
              'text-sm font-medium shadow-2xl animate-fade-in',
              tone.cls,
            )}
          >
            <svg className="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d={tone.icon} />
            </svg>
            <span>{t.message}</span>
          </button>
        );
      })}
    </div>
  );
}
