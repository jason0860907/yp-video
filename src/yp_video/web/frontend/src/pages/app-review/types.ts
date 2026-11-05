import { SIDE_DISPLAY } from '@/lib/courtSide';
import type { CourtSide } from '@/types/api';
import { formatActionTime } from '@/lib/actionEditorModel';
import type { components } from '@/types/openapi';

export type WindowMode = 'full_play' | 'to_next' | 'whole_rally';
export type Scope = 'rallies' | 'actions' | 'scores';
export type Tab = 'clips' | 'analysis' | 'review';

export type AppUser = components['schemas']['AppUser'];

export type JobKind = 'analysis' | 'identify' | 'court';

/** The newest job of one kind for a match (admin library). */
export interface MatchJob {
  kind: JobKind;
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
export type MatchEdits = components['schemas']['MatchEditCounts'] | components['schemas']['MatchReadError'];

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

export type Player = components['schemas']['Player'];
export type Touch = components['schemas']['Touch'];
export type CourtScore = components['schemas']['CourtScore'];
/** One 組織進攻 of a rally: its own touches' span and the spiker. */
export type Attack = components['schemas']['Attack'];
export type RallyClip = components['schemas']['RallyClip'];
/** An action or score clip. */
export type ActionClip = components['schemas']['Clip'];
/** Any clip the feed lists; `kind === 'rally'` tells them apart. */
export type Clip = RallyClip | ActionClip;
/** What the player plays: a feed clip, or a correction's span played
 *  outside the feed (client-side, `kind` 'other' when it has no clip). */
export interface PlayableClip {
  key: string;
  kind: Clip['kind'] | 'other';
  rally_index: number | null;
  start: number;
  end: number;
  touches: Touch[];
  attacks?: Attack[];
}
export type CourtTotal = CourtScore;

/** The App's 分析 → 摘要 numbers (app_review/stats.py). */
export type Stats = components['schemas']['Stats'];
export type Preview = components['schemas']['Preview'];
export type Candidate = components['schemas']['Candidate'];
export type Decision = components['schemas']['Decision'];
export type ReviewSummary = components['schemas']['ReviewSummary'];
/** `match` passes through the Worker's admin API, whose contract lives in
 *  volleyiq-backend — open on this backend. */
export type MatchDetail = Omit<components['schemas']['MatchDetail'], 'match'> & { match: LibraryMatch };
/** GET …/video/meta — the source's probe; the frame clock needs its rate. */
export type VideoMeta = components['schemas']['VideoMeta'];
export type Review = components['schemas']['Review'];

export const DECISION_LABEL: Record<string, string> = {
  accepted_feedback: 'Adopted as feedback',
  rejected: 'Not adopted',
  remove_event: 'Label event removed',
  remove_rally: 'Label rally removed',
  update_rally: 'Rally bounds updated',
};

const RALLY_LEVEL = new Set(['rally', 'rally_bounds', 'winner']);
/** A candidate's full original clip — actions and scores carry one; other
 *  scopes only a span. */
export const candidateClip = (c: Candidate) => (c.clip && 'kind' in c.clip ? c.clip : null);
const at = (c: Candidate) => candidateClip(c)?.anchor_time ?? c.clip?.start ?? Infinity;
/** Review order: what happened to a rally before what happened inside it,
 *  then time. */
export const byReviewOrder = (a: Candidate, b: Candidate) =>
  Number(!RALLY_LEVEL.has(a.scope)) - Number(!RALLY_LEVEL.has(b.scope)) ||
  at(a) - at(b);

/** mm:ss.t — two touches in the same second stay apart. */
export const preciseTime = (t: number) => `${formatActionTime(t)}.${Math.floor((t % 1) * 10)}`;

/** A court side as Label shows it (遠/近/左/右); anything else verbatim. */
export const sideLabel = (side: string) => SIDE_DISPLAY[side as CourtSide] ?? side;
