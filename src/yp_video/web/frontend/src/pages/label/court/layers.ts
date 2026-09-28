/** The overlay layers the user can switch off, in toggle order. */
export const LAYERS = [
  ['court', 'Court lines'],
  ['net', 'Net'],
  ['path', 'Ball path'],
  ['action', 'Action point'],
  ['marks', 'Marks'],
] as const;
export type Layer = (typeof LAYERS)[number][0];
export type Layers = Record<Layer, boolean>;
