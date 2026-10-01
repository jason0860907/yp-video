import { useEffect, useRef } from 'react';

/** The Label panels' shared player keys: Space = play/pause, ←/→ = one frame
 *  (Shift: ten). Each panel supplies what those mean for its player — e.g.
 *  replay a rally from its start, or step through its own frame clock.
 *
 *  Space is play/pause everywhere but in a text field: it is pre-empted on a
 *  focused <video> (no double toggle) and <select> (no native menu), and a
 *  focused button or box is blurred so Space does not also press it. The
 *  arrows leave every form control its own meaning (a slider's nudge, a
 *  select's options). */
export function useVideoKeys(togglePlay: () => void, step: (frames: number) => void) {
  // The latest callbacks, without re-binding the listener every render.
  const actions = useRef({ togglePlay, step });
  useEffect(() => {
    actions.current = { togglePlay, step };
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === 'TEXTAREA') return;
      const inputType = tag === 'INPUT' ? (target as HTMLInputElement).type : null;
      if (e.key === ' ') {
        if (inputType && inputType !== 'range' && inputType !== 'checkbox') return;
        e.preventDefault();
        if (tag === 'BUTTON' || inputType === 'checkbox') target?.blur();
        actions.current.togglePlay();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (inputType || tag === 'SELECT') return;
        e.preventDefault();
        actions.current.step((e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1));
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);
}
