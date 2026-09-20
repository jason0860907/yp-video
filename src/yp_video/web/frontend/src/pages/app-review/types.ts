export type WindowMode = 'full_play' | 'to_next' | 'whole_rally';
export interface Player {
  number: number;
  name: string;
  player_id?: string;
}
export interface Touch {
  kind: string;
  time: number;
  event_id: string;
  player: Player | null;
}
export interface Clip {
  key: string;
  kind: string;
  start: number;
  end: number;
  rally_index: number | null;
  event_id?: string;
  player?: Player | null;
  touches: Touch[];
  result?: string | null;
  loss_reason?: string | null;
  winner?: string | null;
  court_score?: Record<string, number>;
  default_window?: [number, number];
}
export interface Preview {
  rallies: Clip[];
  actions: Clip[];
  scores: Clip[];
  roster: Player[];
  court_totals: Record<string, Record<string, number>>;
  warnings: string[];
}
export interface Bundle {
  result: { match_id: string; job_id: string; user_id: string; video_r2_key: string };
  corrections?: unknown;
  identification?: unknown;
  library_rallies?: unknown;
}
export interface Candidate {
  id: string;
  scope: string;
  value: unknown;
  clip: { start: number; end: number } | null;
  can_remove_event?: boolean;
  rally?: unknown;
  reason?: string | null;
}
export interface Review {
  id: string;
  bundle: Bundle;
  preview: Preview;
  candidates: Candidate[];
  decisions: Record<string, { decision: string; note: string; actor: string; at: string }>;
  application: unknown | null;
}
export interface ReviewSummary {
  id: string;
  match_id: string;
  job_id: string;
  reviewed: number;
  total: number;
}
export const KIND_LABEL: Record<string, string> = {
  rally: '回合',
  serve: '發球',
  receive: '接球',
  set: '舉球',
  spike: '扣球',
  block: '攔網',
  score: '得分',
  other: '其他',
};
