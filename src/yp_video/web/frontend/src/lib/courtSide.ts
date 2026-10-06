import type { CourtSide } from '@/types/api';

export const COURT_SIDES: CourtSide[] = ['left', 'right', 'near', 'far'];
export const SIDE_DISPLAY: Record<CourtSide, string> = { left: '左', right: '右', near: '近', far: '遠' };
