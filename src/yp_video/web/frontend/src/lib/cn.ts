/** Minimal className joiner — truthy strings only, no dedupe needed here. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ');
}
