// Six fixed action hues (kept identical to the legacy editor). Shared by the
// Action Label editor and the ReID Label overlay/action lists.
export const ACTION_COLORS: Record<string, string> = {
  serve: '#38BDF8',
  receive: '#22C55E',
  set: '#A78BFA',
  spike: '#F97316',
  block: '#EF4444',
  score: '#FBBF24',
};

export const actionColor = (label?: string | null) => ACTION_COLORS[label ?? ''] ?? '#8E8E93';
