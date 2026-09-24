export type WindowMode = 'full_play' | 'to_next' | 'whole_rally';
export type Scope = 'rallies' | 'actions' | 'scores';

export interface AppUser {
  id: string;
  provider: string;
  email: string | null;
  created_at: number;
  match_count: number;
  updated_at: number;
}

/** A match row as the App's /sync serves it, plus the R2 keys behind its URLs. */
export interface LibraryMatch {
  id: string;
  owner_id: string;
  title: string;
  angle: string;
  status: 'processing' | 'ready' | 'failed' | 'deleted';
  total_duration_seconds: number;
  recorded_at: number | null;
  folder_id: string | null;
  updated_at: number;
  deleted_at: number | null;
  source_video: { id: string; r2_key: string | null; public_url: string | null } | null;
  analysis: { id: string; status: string | null; step: string | null } | null;
  r2_keys: {
    source: string | null;
    result: string | null;
    corrections: string | null;
    reid: string | null;
  };
}

export interface Tag {
  id: string;
  name: string;
  color_hue: number;
  deleted_at: number | null;
}

export interface Library {
  matches: LibraryMatch[];
  tags: Tag[];
  rally_tags: { rally_id: string; tag_id: string; match_id: string; deleted_at: number | null }[];
  folders: { id: string; name: string; deleted_at: number | null }[];
}

export interface Player {
  number: number;
  name: string;
  player_id?: string | null;
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
  index?: number;
  /** Library rally UUID — rallies only. */
  id?: string;
  start: number;
  end: number;
  rally_index: number | null;
  player?: Player | null;
  touches: Touch[];
  loss_reason?: string | null;
  loss_reason_inferred?: boolean;
  winner?: string | null;
  court_score?: { set: number; points: Record<string, number>; unknown: number } | null;
  tag_ids?: string[];
}
export interface CourtTotal {
  set: number;
  points: Record<string, number>;
  unknown: number;
}
export interface Preview {
  rallies: Clip[];
  actions: Clip[];
  scores: Clip[];
  roster: Player[];
  court_totals: CourtTotal[];
  warnings: string[];
  loss_reasons: Record<string, number>;
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
export interface Decision {
  decision: string;
  note: string;
  actor: string;
  at: string;
}
export interface ReviewSummary {
  id: string;
  created_at: string;
  reviewed: number;
  total: number;
}
export interface MatchDetail {
  match: LibraryMatch;
  preview: Preview;
  candidates: Candidate[];
  review_id: string | null;
  reviews: ReviewSummary[];
}
export interface Review {
  id: string;
  created_at: string;
  candidates: Candidate[];
  decisions: Record<string, Decision>;
  application: unknown | null;
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

export const SIDE_LABEL: Record<string, string> = {
  left: '左側',
  right: '右側',
  near: '近側',
  far: '遠側',
};

const LOSS_REASON: Record<string, string> = {
  spike_error: '扣球失誤',
  blocked: '被攔網',
  set_error: '舉球失誤',
  receive_error: '接發失誤',
  chance_ball_error: '嗆司接噴',
  defense_error: '防守失誤',
  serve_error: '發球失誤',
  other: '其他',
};

/** The App's LossReason.displayName: built-ins by code, custom ones carry
 *  the user's own text after the prefix. */
export const lossReasonLabel = (code: string) =>
  LOSS_REASON[code] ?? (code.startsWith('custom:') ? code.slice(7) : code);

export const DECISION_LABEL: Record<string, string> = {
  accepted_feedback: '已審核保留',
  rejected: '不採用',
  remove_event: '已移除人工標註事件',
  remove_rally: '已移除人工標註回合',
  update_rally: '已更新回合界線',
};

export const playerLabel = (p: Player) => `#${p.number} ${p.name}`;

export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
};

export const fmtDate = (epoch: number) =>
  new Date(epoch * 1000).toLocaleString('zh-TW', { dateStyle: 'short', timeStyle: 'short' });
