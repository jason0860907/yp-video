/** The App's feed rows in its iPad list mode (no thumbnails): RallyCard,
 *  ActionClipCard and the Score scope's card, with the small pieces they
 *  share — ScoreBadge, CourtScoreView, the tinted tag and the player chip.
 *  `extra` is Review's own chip (Edited), which the App does not have. */

import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';
import {
  APP,
  kindOf,
  lossReasonLabel,
  OPPOSITE,
  playerSoftColor,
  rallyTime,
  SIDE,
  tagColor,
  WINNER,
} from './appTerms';
import type { Clip, CourtScore, Player, Tag } from './types';

/** ScoreBadge: ★ score, accent from 85, primary from 70. */
export function ScoreBadge({ score }: { score: number }) {
  const bg = score >= 85 ? APP.accent : score >= 70 ? APP.primary : undefined;
  return (
    <span
      className={cn('inline-flex shrink-0 items-center gap-0.5 rounded-full px-[7px] py-px text-[11px] font-bold tabular-nums', !bg && 'bg-ink/10 text-text-primary')}
      style={bg ? { background: bg, color: '#000' } : undefined}
    >
      ★ {score}
    </span>
  );
}

/** CourtScoreView: 左 3 : 右 2, the side that just scored in primary. */
export function CourtScoreText({ score, scorer }: { score: CourtScore; scorer?: string | null }) {
  const sides = Object.keys(score.points);
  if (!sides.length) return null;
  return (
    <span className="whitespace-nowrap text-xs font-semibold tabular-nums text-text-primary">
      {sides.map((side, i) => (
        <span key={side}>
          {i > 0 && ' : '}
          <span style={side === scorer ? { color: APP.primary } : undefined}>
            {SIDE[side]} {score.points[side]}
          </span>
        </span>
      ))}
    </span>
  );
}

/** ActionClipCard.tag: a tinted label. */
export function TintTag({ children, color }: { children: ReactNode; color: string }) {
  return (
    <span className="whitespace-nowrap rounded-md px-[7px] py-0.5 text-[10px] font-bold" style={{ color, background: `${color}1A` }}>
      {children}
    </span>
  );
}

export function PlayerChip({ player }: { player: Player }) {
  return (
    <span
      className="truncate rounded-md px-2 py-0.5 text-[11px] font-semibold text-black"
      style={{ background: playerSoftColor(player) }}
    >
      #{player.number} {player.name}
    </span>
  );
}

function Row({
  selected,
  emphasis,
  onClick,
  children,
}: {
  selected: boolean;
  emphasis?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'relative w-full overflow-hidden rounded-xl border p-2 text-left transition-colors',
        selected ? 'border-primary/70 bg-primary/10' : 'border-border bg-surface-100/60 hover:bg-surface-100',
      )}
    >
      {emphasis && <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: APP.danger }} />}
      <div className="space-y-1">{children}</div>
    </button>
  );
}

const span = (c: { start: number; end: number }) => `${rallyTime(c.start)}–${rallyTime(c.end)}`;

/** RallyCard: Rally #i, its span, the running score and ★; then 組織進攻 N,
 *  the loss reason (accent while guessed, primary once marked) and up to two
 *  favourites. */
export function RallyCard({
  rally,
  tags,
  selected,
  onClick,
  extra,
}: {
  rally: Clip;
  tags: Tag[];
  selected: boolean;
  onClick: () => void;
  extra?: ReactNode;
}) {
  const attacks = rally.attacks?.length ?? 0;
  const tint = rally.loss_reason_inferred ? APP.accent : APP.primary;
  return (
    <Row selected={selected} onClick={onClick}>
      <div className="flex items-baseline gap-1.5">
        <span className="shrink-0 text-[13px] font-semibold text-text-primary">Rally #{rally.rally_index}</span>
        <span className="truncate font-mono text-[10px] text-text-secondary">{span(rally)}</span>
        <span className="flex-1" />
        {rally.court_score && <CourtScoreText score={rally.court_score} scorer={rally.winner} />}
        {rally.score != null && <ScoreBadge score={rally.score} />}
      </div>
      {(attacks > 0 || rally.loss_reason || tags.length > 0 || extra) && (
        <div className="flex flex-wrap items-center gap-1.5">
          {attacks > 0 && <span className="text-[10px] font-bold text-text-secondary">組織進攻 {attacks}</span>}
          {rally.loss_reason && (
            <span className="rounded-full border px-[5px] text-[10px] font-bold" style={{ color: tint, borderColor: `${tint}99` }}>
              {lossReasonLabel(rally.loss_reason)}
            </span>
          )}
          {tags.slice(0, 2).map((t) => (
            <span key={t.id} className="rounded-full px-[5px] text-[10px] font-bold text-white" style={{ background: tagColor(t.color_hue) }}>
              {t.name}
            </span>
          ))}
          {tags.length > 2 && <span className="text-[10px] font-bold text-text-secondary">+{tags.length - 2}</span>}
          {extra}
        </div>
      )}
    </Row>
  );
}

/** ActionClipCard in the Action scope: 扣球 #3 and the player; then where
 *  the touch is — its rally and the clip's span. */
export function ActionCard({
  clip,
  number,
  selected,
  onClick,
  extra,
}: {
  clip: Clip;
  /** Per kind under a kind filter, across all kinds otherwise. */
  number: number;
  selected: boolean;
  onClick: () => void;
  extra?: ReactNode;
}) {
  return (
    <Row selected={selected} onClick={onClick}>
      <div className="flex items-baseline gap-1.5">
        <span className="shrink-0 text-[13px] font-semibold text-text-primary">
          {kindOf(clip.kind).long} #{number}
        </span>
        <span className="flex-1" />
        {clip.player && <PlayerChip player={clip.player} />}
      </div>
      <div className="flex items-center gap-1.5 font-mono text-[10px] text-text-secondary">
        {clip.rally_index != null && <span>Rally #{clip.rally_index}</span>}
        <span>{span(clip)}</span>
        <span className="flex-1" />
        {extra}
      </div>
    </Row>
  );
}

/** ActionClipCard in the Score scope: the rally, span and running score; the
 *  winner; who played the deciding ball and why the other side lost. */
export function ScoreCard({
  clip,
  selected,
  onClick,
  extra,
}: {
  clip: Clip;
  selected: boolean;
  onClick: () => void;
  extra?: ReactNode;
}) {
  const losing = clip.winner ? OPPOSITE[clip.winner] : null;
  const last = [...clip.touches].reverse().find((t) => t.kind !== 'score');
  const who = clip.player ? `#${clip.player.number} ${clip.player.name}` : '未配對';
  return (
    <Row selected={selected} emphasis={!!losing && !!clip.loss_reason} onClick={onClick}>
      <div className="flex items-baseline gap-1.5">
        <span className="shrink-0 text-[13px] font-semibold text-text-primary">
          {clip.rally_index != null ? 'Rally' : 'Score'} #{clip.index}
        </span>
        <span className="truncate font-mono text-[10px] text-text-secondary">{span(clip)}</span>
        <span className="flex-1" />
        {clip.court_score && Object.keys(clip.court_score.points).length > 0 && (
          <CourtScoreText score={clip.court_score} />
        )}
      </div>
      <div className="flex items-center gap-1.5">
        {extra}
        <span className="flex-1" />
        <TintTag color={clip.winner ? APP.primary : 'var(--text-secondary)'}>
          {clip.winner ? WINNER[clip.winner] : '勝方待確認'}
        </TintTag>
      </div>
      {(last || losing) && (
        <div className="flex items-center gap-1.5">
          {last && (
            <span className="truncate text-[10px] font-semibold text-text-secondary">
              決定分：{who} · {kindOf(last.kind).long}
            </span>
          )}
          <span className="flex-1" />
          {losing && (
            <TintTag color={clip.loss_reason ? APP.danger : 'var(--text-secondary)'}>
              {SIDE[losing]}失分 · {clip.loss_reason ? lossReasonLabel(clip.loss_reason) : '未標記'}
            </TintTag>
          )}
        </div>
      )}
    </Row>
  );
}

