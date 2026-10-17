/** ?user=&match=&tab=&scope=&clip=&fix= — the App Review page's shareable,
 *  refresh-safe state, the way useLabelUrlState keeps Label's. Writes
 *  replace history; an empty value drops its key. */

import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { Scope, Tab } from './types';

const TABS: readonly string[] = ['clips', 'review'];
const SCOPES: readonly string[] = ['rallies', 'actions', 'scores'];

export function useAppReviewUrlState() {
  const [params, setParams] = useSearchParams();
  const tab = params.get('tab');
  const scope = params.get('scope');
  const state = {
    user: params.get('user') ?? '',
    match: params.get('match') ?? '',
    tab: (tab && TABS.includes(tab) ? tab : 'clips') as Tab,
    scope: (scope && SCOPES.includes(scope) ? scope : 'rallies') as Scope,
    clip: params.get('clip') ?? '',
    /** The correction open in Review. */
    fix: params.get('fix') ?? '',
  };

  const set = useCallback(
    (next: Partial<typeof state>) =>
      setParams(
        (prev) => {
          const out = new URLSearchParams(prev);
          for (const [k, v] of Object.entries(next)) {
            if (v) out.set(k, v);
            else out.delete(k);
          }
          return out;
        },
        { replace: true },
      ),
    [setParams],
  );

  return { ...state, set };
}
