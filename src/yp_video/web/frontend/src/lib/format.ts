/** Display formatters for the data role (times, sizes, durations). Ported
 *  verbatim from the legacy shared helpers. */

const pad = (n: number, len: number) => String(n).padStart(len, '0');

/** `mm:ss` */
export function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || isNaN(seconds)) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${pad(m, 2)}:${pad(s, 2)}`;
}

/** Parse `mm:ss` or a bare seconds string back to seconds. */
export function parseTime(str: string | number | null | undefined): number {
  if (str == null) return 0;
  const s = String(str).trim();
  if (s.includes(':')) {
    const [mm, ss] = s.split(':');
    return (parseInt(mm ?? '0', 10) || 0) * 60 + (parseFloat(ss ?? '0') || 0);
  }
  return parseFloat(s) || 0;
}

/** `mm:ss.mmm` */
export function formatTimePrecise(seconds: number | null | undefined): string {
  if (seconds == null || isNaN(seconds)) return '00:00.000';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${pad(m, 2)}:${s.toFixed(3).padStart(6, '0')}`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

export function formatSpeed(bytesPerSec: number | null | undefined): string {
  if (!bytesPerSec) return '0 B/s';
  return `${formatBytes(bytesPerSec)}/s`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
