import type { CourtSide } from '@/types/api';

export const COURT_SIDES: CourtSide[] = ['left', 'right', 'near', 'far'];
/** One-character chips for tight rows; the full side name goes in the title. */
export const SIDE_DISPLAY: Record<CourtSide, string> = { left: '左', right: '右', near: '近', far: '遠' };
