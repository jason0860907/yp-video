import { twMerge } from 'tailwind-merge';

/** className joiner with Tailwind conflict resolution — the last conflicting
 *  utility wins, so callers can override component defaults (e.g. `mb-0`
 *  beating a built-in `mb-2.5`) without `!important`. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return twMerge(parts.filter(Boolean).join(' '));
}
