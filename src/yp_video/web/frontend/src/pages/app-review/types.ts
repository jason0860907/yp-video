import { SIDE_DISPLAY } from '@/lib/courtSide';
import type { CourtSide } from '@/types/api';
import { formatActionTime } from '@/lib/actionEditorModel';

export type WindowMode = 'full_play' | 'to_next' | 'whole_rally';
export type Scope = 'rallies' | 'actions' | 'scores';
export type Tab = 'clips' | 'review';

export interface AppUser {
  id: string;
  provider: string;
  email: string | null;
  display_name: string | null;
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
  /** The full original clip for actions / scores, just the span otherwise. */
  clip: (Partial<Clip> & { start: number; end: number; anchor_time?: number }) | null;
  /** The original analysis' rally it belongs to; null for match-wide ones. */
  rally_index: number | null;
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
/** GET …/video/meta — the source's probe; the frame clock needs its rate. */
export interface VideoMeta {
  fps: number;
  num_frames: number;
  duration: number;
}
export interface Review {
  id: string;
  created_at: string;
  candidates: Candidate[];
  decisions: Record<string, Decision>;
  application: unknown | null;
}

const LOSS_REASON: Record<string, string> = {
  spike_error: 'Spike error',
  blocked: 'Blocked',
  set_error: 'Set error',
  receive_error: 'Receive error',
  chance_ball_error: 'Free-ball error',
  defense_error: 'Defense error',
  serve_error: 'Serve error',
  other: 'Other',
};

/** The App's LossReason.displayName: built-ins by code, custom ones carry
 *  the user's own text after the prefix. */
export const lossReasonLabel = (code: string) =>
  LOSS_REASON[code] ?? (code.startsWith('custom:') ? code.slice(7) : code);

export const DECISION_LABEL: Record<string, string> = {
  accepted_feedback: 'Adopted as feedback',
  rejected: 'Not adopted',
  remove_event: 'Label event removed',
  remove_rally: 'Label rally removed',
  update_rally: 'Rally bounds updated',
};

const RALLY_LEVEL = new Set(['rally', 'rally_bounds', 'winner']);
const at = (c: Candidate) => c.clip?.anchor_time ?? c.clip?.start ?? Infinity;
/** Review order: what happened to a rally before what happened inside it,
 *  then time. */
export const byReviewOrder = (a: Candidate, b: Candidate) =>
  Number(!RALLY_LEVEL.has(a.scope)) - Number(!RALLY_LEVEL.has(b.scope)) ||
  at(a) - at(b);

/** mm:ss.t — two touches in the same second stay apart. */
export const preciseTime = (t: number) => `${formatActionTime(t)}.${Math.floor((t % 1) * 10)}`;

export const playerLabel = (p: Player) => `#${p.number} ${p.name}`;

/** A court side as Label shows it (遠/近/左/右); anything else verbatim. */
export const sideLabel = (side: string) => SIDE_DISPLAY[side as CourtSide] ?? side;
