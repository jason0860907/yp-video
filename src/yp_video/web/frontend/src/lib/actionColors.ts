// Fixed action hues. Shared by the Action Label editor and the ReID Label
// overlay/action lists.
export const ACTION_COLORS: Record<string, string> = {
  serve: '#38BDF8',
  receive: '#22C55E',
  set: '#A78BFA',
  spike: '#F97316',
  block: '#EF4444',
  score: '#FBBF24',
  net: '#F472B6',
  ground: '#FBBF24',
  antenna: '#FBBF24',
};

export const actionColor = (label?: string | null) => ACTION_COLORS[label ?? ''] ?? '#8E8E93';

// The court drawings' floor: neutral slate darker than every action hue — the
// six span every hue, so a coloured floor would swallow one of them. The ball
// is white for the same reason (yellow was the score's colour).
export const COURT = {
  surface: '#4A5361',
  freeZone: '#2E343E',
  ball: '#FFFFFF',
};
