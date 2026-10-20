import { SIDE_DISPLAY } from '@/lib/courtSide';
import type { CourtSide } from '@/types/api';
import { formatActionTime } from '@/lib/actionEditorModel';

export type WindowMode = 'full_play' | 'to_next' | 'whole_rally';
export type Scope = 'rallies' | 'actions' | 'scores';
export type Tab = 'clips' | 'analysis' | 'review';

export interface AppUser {
  id: string;
  provider: string;
  email: string | null;
  display_name: string | null;
  created_at: number;
  match_count: number;
  updated_at: number;
}

export type JobKind = 'analysis' | 'identify' | 'court';

/** The newest job of one kind for a match (admin library). */
export interface MatchJob {
  kind: JobKind;
  mode: 'standard' | 'advanced' | null;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'superseded';
  step: string | null;
  progress: number;
  error_code: string | null;
  error: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

/** GET …/overview, per analysed match: the user's corrections by kind. */
export type MatchEdits =
  | { edits: Record<string, number>; edited_at: string | null; warnings: string[] }
  | { error: string };

/** A match row as the App's /sync serves it, plus the R2 keys behind its URLs
 *  and its newest job of each kind. */
export interface LibraryMatch {
  id: string;
  owner_id: string;
  title: string;
  angle: string;
  status: 'processing' | 'ready' | 'failed' | 'deleted';
  total_duration_seconds: number;
  recorded_at: number | null;
  folder_id: string | null;
  created_at: number;
  updated_at: number;
  deleted_at: number | null;
  source_video: { id: string; r2_key: string | null; public_url: string | null } | null;
  analysis: { id: string; status: string | null; step: string | null } | null;
  jobs: Partial<Record<JobKind, MatchJob>>;
  r2_keys: {
    source: string | null;
    result: string | null;
    corrections: string | null;
    reid: string | null;
  };
}

/** The library's live matches, newest first — the picker's and Overview's order. */
export const liveMatches = (library: Library | undefined) =>
  (library?.matches ?? [])
    .filter((m) => m.deleted_at === null)
    .sort((a, b) => (b.recorded_at ?? b.updated_at) - (a.recorded_at ?? a.updated_at));

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
  hue: number;
  player_id?: string | null;
}
export interface Touch {
  kind: string;
  time: number;
  event_id: string;
  player: Player | null;
  /** The ball was seen at contact. */
  visible: boolean;
}
export interface CourtScore {
  set: number;
  points: Record<string, number>;
  unknown: number;
}
/** One 組織進攻 of a rally: its own touches' span and the spiker. */
export interface Attack {
  key: string;
  start: number;
  end: number;
  player: Player | null;
}
export interface Clip {
  key: string;
  kind: string;
  index?: number;
  /** Actions: the number among clips of the same kind (扣球 #3). */
  kind_index?: number;
  /** Rallies: the model's highlight score (★). */
  score?: number;
  /** Library rally UUID — rallies only. */
  id?: string;
  start: number;
  end: number;
  set?: number;
  rally_index: number | null;
  player?: Player | null;
  touches: Touch[];
  loss_reason?: string | null;
  loss_reason_inferred?: boolean;
  winner?: string | null;
  court_score?: CourtScore | null;
  tag_ids?: string[];
  /** Rallies: its attacks, in time order. */
  attacks?: Attack[];
}
export type CourtTotal = CourtScore;

/** The App's 分析 → 摘要 numbers (app_review/stats.py). */
export interface Stats {
  rally_count: number;
  in_rally_seconds: number;
  team_sides: { start: string; switches: number[] } | null;
  team_score: { us: number; them: number } | null;
  has_teams: boolean;
  findings: { kind: 'top_loss_reason'; team: 'us' | null; reason: string; count: number; total: number }[];
  progression: { rally: number; side: string; points: number }[];
  players: { player: Player; touches: Record<string, number>; deciding: number }[];
  losses: { side: string; total: number; reasons: { reason: string; count: number; percent: number }[] }[];
}
export interface Preview {
  rallies: Clip[];
  actions: Clip[];
  scores: Clip[];
  roster: Player[];
  court_totals: CourtTotal[];
  warnings: string[];
  stats: Stats;
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

/** A court side as Label shows it (遠/近/左/右); anything else verbatim. */
export const sideLabel = (side: string) => SIDE_DISPLAY[side as CourtSide] ?? side;
