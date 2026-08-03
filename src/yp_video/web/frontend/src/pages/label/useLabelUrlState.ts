/** ?video=<cut filename>&mode=<rally|action|association|reid> — the Label
 *  page's shareable, refresh-safe state.
 *
 *  Writes use `replace: true` for both keys: the browser Back button cannot
 *  be dirty-guarded, so history stays shallow rather than offering a way
 *  around the guard. An unknown video is kept as-is — a name that is not
 *  listed yet still loads once the lists arrive.
 */

import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { LabelMode } from '@/lib/labelStatus';

const MODES: readonly string[] = ['rally', 'action', 'association', 'reid'];
const parseMode = (raw: string | null): LabelMode => (raw && MODES.includes(raw) ? (raw as LabelMode) : 'rally');

export function useLabelUrlState() {
  const [params, setParams] = useSearchParams();
  const video = params.get('video') ?? '';
  const mode = parseMode(params.get('mode'));

  const set = useCallback(
    (next: { video?: string; mode?: LabelMode }) => {
      setParams(
        (prev) => {
          const out = new URLSearchParams(prev);
          if (next.video !== undefined) {
            if (next.video) out.set('video', next.video);
            else out.delete('video');
          }
          if (next.mode !== undefined) out.set('mode', next.mode);
          return out;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  return { video, mode, set };
}
