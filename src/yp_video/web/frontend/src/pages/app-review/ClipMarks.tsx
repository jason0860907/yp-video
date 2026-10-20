/** What the App's detail shows under the scrubber, read-only: the clip's
 *  meta line and the marks the user can change there — winner, loss reason,
 *  player, favourites (RallyDetailView / ActionDetailView / ScoreDetailView
 *  in their wide layout). A mark still at the App's guess reads as a guess. */

import { CourtScoreText, PlayerChip, ScoreBadge, TintTag } from './AppRows';
import { APP, FRAMING, kindOf, lossReasonLabel, rallyTime, tagColor, WINNER } from './appTerms';
import type { Clip, Scope, Tag, WindowMode } from './types';

const span = (c: Clip) => `${rallyTime(c.start)}–${rallyTime(c.end)}`;

export function ClipMarks({ clip, scope, mode, number, tags }: { clip: Clip; scope: Scope; mode: WindowMode; number: number; tags: Tag[] }) {
  return (
    <div className="mt-3 space-y-2 border-t border-border pt-3 text-xs">
      {scope === 'rallies' ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold text-text-primary">Rally #{clip.rally_index}</span>
          {clip.score != null && <ScoreBadge score={clip.score} />}
          <span className="font-mono text-text-secondary">{span(clip)}</span>
          {clip.court_score && <CourtScoreText score={clip.court_score} scorer={clip.winner} />}
          <span className="flex-1" />
          <Winner side={clip.winner} />
        </div>
      ) : scope === 'actions' ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold text-text-primary">
            {kindOf(clip.kind).long} #{number}
          </span>
          <span className="font-mono text-text-secondary">
            {clip.rally_index != null && `Rally #${clip.rally_index} · `}
            {span(clip)}
          </span>
          <span className="flex-1" />
          <TintTag color="var(--text-secondary)">{FRAMING[mode]}</TintTag>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold" style={{ color: clip.winner ? APP.primary : 'var(--text-secondary)' }}>
            {clip.winner ? WINNER[clip.winner] : '勝方待確認'}
          </span>
          <span className="font-mono text-text-secondary">
            {span(clip)}
            {clip.rally_index != null && ` · 來自 Rally #${clip.rally_index}`}
          </span>
          {clip.court_score && <CourtScoreText score={clip.court_score} scorer={clip.winner} />}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {scope === 'actions' ? (
          <Mark label="球員">{clip.player ? <PlayerChip player={clip.player} /> : <span className="text-text-muted">未指定</span>}</Mark>
        ) : (
          <>
            <Mark label="失分原因">
              {clip.loss_reason ? (
                clip.loss_reason_inferred ? (
                  <span className="rounded-full border px-2 py-0.5 font-bold" style={{ color: APP.accent, borderColor: `${APP.accent}99` }}>
                    自動判定：{lossReasonLabel(clip.loss_reason)}
                  </span>
                ) : (
                  <span className="rounded-full px-2 py-0.5 font-bold text-white" style={{ background: APP.danger }}>
                    {lossReasonLabel(clip.loss_reason)}
                  </span>
                )
              ) : (
                <span className="text-text-muted">未標記</span>
              )}
            </Mark>
            {scope === 'scores' && (
              <Mark label="決定分">
                {clip.player ? <PlayerChip player={clip.player} /> : <span className="text-text-muted">未配對</span>}
              </Mark>
            )}
          </>
        )}
        {tags.length > 0 && (
          <Mark label="分類">
            {tags.map((t) => (
              <span key={t.id} className="rounded-full px-2 py-0.5 font-bold text-white" style={{ background: tagColor(t.color_hue) }}>
                {t.name}
              </span>
            ))}
          </Mark>
        )}
      </div>
    </div>
  );
}

function Winner({ side }: { side?: string | null }) {
  return <TintTag color={side ? APP.primary : 'var(--text-secondary)'}>{side ? WINNER[side] : '勝方待確認'}</TintTag>;
}

function Mark({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-text-muted">{label}</span>
      {children}
    </span>
  );
}
