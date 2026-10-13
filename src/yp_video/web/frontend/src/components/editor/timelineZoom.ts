/** The window lengths every Label timeline offers, and the one it opens at.
 *  spv = seconds per view; 0 means "All" (the whole video). */
export const ZOOMS: ReadonlyArray<{ label: string; spv: number }> = [
  { label: 'All', spv: 0 },
  { label: '10m', spv: 600 },
  { label: '5m', spv: 300 },
  { label: '3m', spv: 180 },
  { label: '30s', spv: 30 },
  { label: '10s', spv: 10 },
];

/** One rally fills the view: wide enough to see it whole, close enough to
 *  place a boundary or an action without zooming first. */
export const DEFAULT_SPV = 30;
