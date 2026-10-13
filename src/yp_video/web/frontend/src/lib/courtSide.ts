import type { CourtSide } from '@/types/api';

export const COURT_SIDES: CourtSide[] = ['left', 'right', 'near', 'far'];
/** One-letter chips for tight rows; the full side name goes in the title. */
export const SIDE_DISPLAY: Record<CourtSide, string> = { left: 'L', right: 'R', near: 'N', far: 'F' };
