import type { PointerEvent as ReactPointerEvent } from 'react';

/** Follow one pointer drag from its pointerdown: capture the pointer on the
 *  element it started on, report every move, and clean up on release. */
export function trackPointer(
  e: ReactPointerEvent,
  onMove: (ev: PointerEvent) => void,
  onUp: () => void = () => {},
) {
  const target = e.currentTarget as HTMLElement;
  target.setPointerCapture(e.pointerId);
  const up = () => {
    target.removeEventListener('pointermove', onMove);
    target.removeEventListener('pointerup', up);
    onUp();
  };
  target.addEventListener('pointermove', onMove);
  target.addEventListener('pointerup', up);
}
