/** Global toast store. `toast.success(...)` etc. enqueue a transient message;
 *  <Toaster> renders the live list via useSyncExternalStore. Auto-dismiss
 *  after 3.5s, matching the legacy behaviour. */

export type ToastType = 'info' | 'success' | 'error' | 'warning';

export interface ToastItem {
  id: number;
  message: string;
  type: ToastType;
}

const DISMISS_MS = 3500;

let items: ToastItem[] = [];
let seq = 0;
const listeners = new Set<() => void>();

const emit = () => listeners.forEach((l) => l());

export function subscribeToasts(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function getToasts(): ToastItem[] {
  return items;
}

export function dismissToast(id: number): void {
  items = items.filter((t) => t.id !== id);
  emit();
}

function push(message: string, type: ToastType): void {
  const id = ++seq;
  items = [...items, { id, message, type }];
  emit();
  setTimeout(() => dismissToast(id), DISMISS_MS);
}

export const toast = {
  info: (message: string) => push(message, 'info'),
  success: (message: string) => push(message, 'success'),
  error: (message: string) => push(message, 'error'),
  warning: (message: string) => push(message, 'warning'),
};
