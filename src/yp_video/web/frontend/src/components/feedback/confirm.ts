/** Promise-based confirm dialog. `await confirm({...})` resolves true/false;
 *  <ConfirmDialog> renders the active request. One at a time. */

export type ConfirmVariant = 'info' | 'warning' | 'danger';

export interface ConfirmOptions {
  title: string;
  body?: string;
  confirmText?: string;
  cancelText?: string;
  variant?: ConfirmVariant;
}

interface ActiveConfirm extends ConfirmOptions {
  resolve: (ok: boolean) => void;
}

let current: ActiveConfirm | null = null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export function subscribeConfirm(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
export function getConfirm(): ActiveConfirm | null {
  return current;
}

export function confirm(options: ConfirmOptions): Promise<boolean> {
  // If one is already open, reject it as cancelled before replacing.
  current?.resolve(false);
  return new Promise<boolean>((resolve) => {
    current = { ...options, resolve };
    emit();
  });
}

export function settleConfirm(ok: boolean): void {
  current?.resolve(ok);
  current = null;
  emit();
}
