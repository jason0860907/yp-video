import type { CutKind } from '@/types/api';

/** Every cut kind in display order — the one list each kind picker, filter
 *  and count reads (config.CUT_KINDS on the server). */
export const CUT_KIND_OPTIONS: { kind: CutKind; label: string; badge: string }[] = [
  { kind: 'broadcast', label: 'Broadcast', badge: 'cast' },
  { kind: 'sideline', label: 'Sideline', badge: 'side' },
  { kind: 'user-baseline', label: 'User · Baseline', badge: 'user·底線' },
  { kind: 'user-corner', label: 'User · Corner', badge: 'user·角落' },
  { kind: 'user-high', label: 'User · High', badge: 'user·高處' },
  { kind: 'user-center', label: 'User · Center', badge: 'user·中線' },
];
