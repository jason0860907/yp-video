import { useState } from 'react';
import { cn } from '@/lib/cn';
import { ACTOR_KIND_LABELS, METRIC_LABELS, TASK_LABELS, TASK_ORDER } from '@/components/train/metricLabels';
import type {
  ActorBreakdown,
  LocationBreakdown,
  SpottingBreakdown,
  TaskMetricSnapshot,
  TaskMetrics,
  WinnerBreakdown,
} from '@/types/api';

/** One epoch's task metrics, as the panel wants them. */
export interface TaskSnapshot {
  epoch: number;
  tasks: TaskMetrics;
}

const pct = (v: number | null | undefined) =>
  typeof v === 'number' && Number.isFinite(v) ? (v * 100).toFixed(1) : '—';

// One dense-table idiom for every task, so the five heads read as one panel.
const BOX = 'min-w-0 rounded-lg border border-border bg-surface-100 px-3 py-2.5';
const CAPTION = 'text-[9px] uppercase tracking-wider text-text-muted';
const TABLE = 'mt-1 font-mono text-[10px] tabular-nums';
const HEAD = 'py-0.5 text-left font-normal';
const NUM = 'py-0.5 pl-5 text-right';
const NUM_HEAD = cn(NUM, 'font-normal');

const hasBreakdown = (tasks: TaskMetrics) =>
  Object.values(tasks).some((task) => task.validation.breakdown != null);

/** Per-task validation detail for one epoch: per-class / per-tolerance /
 *  per-video tables for the spotting heads, the confusion matrix for winner,
 *  per-kind accuracy for actor. Shows the best checkpoint's epoch by default
 *  and lets the reader flip to the latest one. */
export function TaskBreakdownPanel({ best, latest }: { best?: TaskSnapshot; latest?: TaskSnapshot }) {
  const [which, setWhich] = useState<'best' | 'latest'>('best');
  const options = [
    best ? { key: 'best' as const, label: 'Best', snapshot: best } : null,
    latest && latest.epoch !== best?.epoch ? { key: 'latest' as const, label: 'Latest', snapshot: latest } : null,
  ].filter((o): o is NonNullable<typeof o> => o != null && hasBreakdown(o.snapshot.tasks));
  if (!options.length) return null;
  const active = options.find((o) => o.key === which) ?? options[0]!;
  const names = TASK_ORDER.filter((task) => active.snapshot.tasks[task]?.validation.breakdown != null);

  return (
    <div className="mt-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold text-text-primary">Validation breakdown · ep{active.snapshot.epoch}</span>
        {options.length > 1 && (
          <div className="flex gap-1.5">
            {options.map((o) => (
              <button
                key={o.key}
                type="button"
                onClick={() => setWhich(o.key)}
                className={cn(
                  'rounded-lg border px-2.5 py-1 font-mono text-[11px] font-medium transition-colors',
                  o.key === active.key
                    ? 'border-primary/25 bg-primary/15 text-primary-light'
                    : 'border-border bg-surface-50 text-text-secondary hover:text-text-primary',
                )}
              >
                {o.label} · ep{o.snapshot.epoch}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="mt-2 space-y-3">
        {names.map((task) => (
          <TaskSection key={task} task={task} snapshot={active.snapshot.tasks[task]!} />
        ))}
      </div>
    </div>
  );
}

function TaskSection({ task, snapshot }: { task: string; snapshot: TaskMetricSnapshot }) {
  const validation = snapshot.validation;
  const primary = snapshot.primary_metric;
  const n = primary === 'player_top1' ? validation.counts.player_events : validation.counts.events;
  const noun = task === 'rally' ? 'rallies' : 'events';
  const breakdown = validation.breakdown!;
  return (
    <div>
      <div className="flex flex-wrap items-baseline gap-x-2 text-[11px]">
        <span className="font-semibold text-text-primary">{TASK_LABELS[task] ?? task}</span>
        <span className="text-text-muted">{METRIC_LABELS[primary] ?? primary}</span>
        <span className="font-mono tabular-nums text-text-primary">{pct(validation.metrics[primary])}%</span>
        {typeof n === 'number' && (
          <span className="text-text-muted">
            · {n.toLocaleString()} {noun}
          </span>
        )}
      </div>
      <div className="mt-1.5 grid grid-cols-1 items-start gap-3 xl:grid-cols-[auto_minmax(0,1fr)]">
        {task === 'winner' ? (
          <WinnerView bd={breakdown as WinnerBreakdown} majority={validation.metrics.majority_baseline} />
        ) : task === 'actor' ? (
          <ActorView bd={breakdown as ActorBreakdown} />
        ) : task === 'location' ? (
          <LocationView bd={breakdown as LocationBreakdown} />
        ) : (
          <SpottingView bd={breakdown as SpottingBreakdown} segment={task === 'rally'} noun={noun} />
        )}
      </div>
    </div>
  );
}

function PerVideoTable({
  rows,
  metric,
  noun,
}: {
  rows: Array<{ video: string; value: number; events: number }>;
  metric: string;
  noun: string;
}) {
  if (!rows.length) return null;
  return (
    <div className={BOX}>
      <div className={CAPTION}>By video</div>
      <table className={cn(TABLE, 'w-full')}>
        <thead>
          <tr className="text-text-muted">
            <th className={HEAD}>Video</th>
            <th className="w-12 py-0.5 text-right font-normal">{metric}</th>
            <th className="w-14 py-0.5 text-right font-normal">{noun}</th>
          </tr>
        </thead>
        <tbody>
          {[...rows]
            .sort((a, b) => b.value - a.value)
            .map((r) => (
              <tr key={r.video}>
                <td className="max-w-0 truncate py-0.5 pr-3 text-left text-text-secondary" title={r.video}>
                  {r.video}
                </td>
                <td className="py-0.5 text-right text-text-primary">{pct(r.value)}</td>
                <td className="py-0.5 text-right text-text-secondary">{r.events}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}

function SpottingView({ bd, segment, noun }: { bd: SpottingBreakdown; segment: boolean; noun: string }) {
  return (
    <>
      <div className={BOX}>
        <div className={CAPTION}>{segment ? 'By class (AP @ tIoU)' : 'By class (AP @ frame tolerance)'}</div>
        <table className={TABLE}>
          <thead>
            <tr className="text-text-muted">
              <th className={HEAD}>Class</th>
              {bd.tolerances.map((t) => (
                <th key={t} className={NUM_HEAD}>
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(bd.classes).map(([cls, aps]) => (
              <tr key={cls}>
                <td className="py-0.5 pr-2 text-left text-text-secondary">{cls}</td>
                {aps.map((v, i) => (
                  <td key={i} className={cn(NUM, 'text-text-secondary')}>
                    {pct(v)}
                  </td>
                ))}
              </tr>
            ))}
            <tr className="border-t border-border text-text-primary">
              <td className="py-0.5 pr-2 text-left">overall</td>
              {bd.overall.map((v, i) => (
                <td key={i} className={NUM}>
                  {pct(v)}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <PerVideoTable rows={bd.per_video.map((v) => ({ video: v.video, value: v.temporal, events: v.events }))} metric="mAP" noun={noun} />
    </>
  );
}

function LocationView({ bd }: { bd: LocationBreakdown }) {
  return (
    <>
      <div className={BOX}>
        <div className={CAPTION}>mAP @ pixel tolerance</div>
        <table className={TABLE}>
          <thead>
            <tr className="text-text-muted">
              {bd.pixel_tolerances.map((px) => (
                <th key={px} className={cn(NUM_HEAD, 'first:pl-0')}>
                  {px}px
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="text-text-primary">
              {bd.overall_by_px.map((v, i) => (
                <td key={i} className={cn(NUM, 'first:pl-0')}>
                  {pct(v)}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <PerVideoTable rows={bd.per_video.map((v) => ({ video: v.video, value: v.spatial, events: v.events }))} metric="mAP" noun="events" />
    </>
  );
}

function WinnerView({ bd, majority }: { bd: WinnerBreakdown; majority: number | null | undefined }) {
  // A camera view supervises only two of the four sides; drop sides that
  // never appear as truth or prediction so the matrix shows what was scored.
  const live = bd.classes
    .map((side, i) => ({ side, i }))
    .filter(({ i }) => bd.confusion[i]!.some((n) => n > 0) || bd.confusion.some((row) => row[i]! > 0));
  return (
    <div className={BOX}>
      <div className={CAPTION}>
        Confusion (rows truth · columns predicted)
        {typeof majority === 'number' ? ` · majority baseline ${pct(majority)}%` : ''}
      </div>
      <table className={TABLE}>
        <thead>
          <tr className="text-text-muted">
            <th className={HEAD}>truth \ pred</th>
            {live.map(({ side }) => (
              <th key={side} className={NUM_HEAD}>
                {side}
              </th>
            ))}
            <th className={cn(NUM_HEAD, 'border-l border-border')}>recall</th>
          </tr>
        </thead>
        <tbody>
          {live.map(({ side, i }) => (
            <tr key={side}>
              <td className="py-0.5 pr-2 text-left text-text-secondary">{side}</td>
              {live.map(({ i: j }) => (
                <td key={j} className={cn(NUM, i === j ? 'font-semibold text-text-primary' : 'text-text-secondary')}>
                  {bd.confusion[i]![j]!.toLocaleString()}
                </td>
              ))}
              <td className={cn(NUM, 'border-l border-border text-text-primary')}>{pct(bd.recall[side])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ActorView({ bd }: { bd: ActorBreakdown }) {
  return (
    <div className={BOX}>
      <div className={CAPTION}>By target kind</div>
      <table className={TABLE}>
        <thead>
          <tr className="text-text-muted">
            <th className={HEAD}>Kind</th>
            <th className={NUM_HEAD}>events</th>
            <th className={NUM_HEAD}>correct</th>
            <th className={NUM_HEAD}>rate</th>
          </tr>
        </thead>
        <tbody>
          {bd.kinds.map((row) => (
            <tr key={row.kind}>
              <td className="py-0.5 pr-2 text-left text-text-secondary">{ACTOR_KIND_LABELS[row.kind] ?? row.kind}</td>
              <td className={cn(NUM, 'text-text-secondary')}>{row.events.toLocaleString()}</td>
              <td className={cn(NUM, 'text-text-secondary')}>{row.correct.toLocaleString()}</td>
              <td className={cn(NUM, 'text-text-primary')}>{pct(row.rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
