/** ReID Train: measure the embedders on the labels we have, then export them
 *  as a CLIP-ReIdent dataset.
 *
 *  Model training is not here yet, and that ordering is the point — with two
 *  labeled videos there is nothing to train that would generalize, and no way
 *  to tell whether it helped. So this page answers "is fine-tuning worth it?"
 *  first: per-session mAP/Rank-1 for every registered embedder, plus the
 *  clustering threshold those same labeled pairs imply.
 */

import { useEffect, useMemo, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { Field, SelectArch, fieldCls } from '@/components/train/Field';
import { JobProgress } from '@/components/job/JobProgress';
import { toast } from '@/components/feedback/toast';
import { useSSE } from '@/lib/useSSE';
import { isTerminal } from '@/lib/job';
import type { Job, ReidGroupEval, ReidModelEval, ReidPerfData, ReidTrainStatus } from '@/types/api';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

interface ExportForm {
  name: string;
  split_mode: string;
  test_ratio: number;
  seed: number;
  link: boolean;
  masked: boolean;
  overwrite: boolean;
}

const BASE_FORM: ExportForm = {
  name: '',
  split_mode: 'auto',
  test_ratio: 0.25,
  seed: 42,
  link: true,
  masked: false,
  overwrite: false,
};

export function ReidTrainPage() {
  const [form, setForm] = useState<ExportForm>(BASE_FORM);
  const [job, setJob] = useState<Job | null>(null);
  const [openModel, setOpenModel] = useState<string | null>(null);
  const set = <K extends keyof ExportForm>(key: K, value: ExportForm[K]) => setForm((f) => ({ ...f, [key]: value }));

  const statusQuery = useQuery({
    queryKey: ['reid-train-status'],
    queryFn: () => apiFetch<ReidTrainStatus>(API.reidTrain.status),
    refetchInterval: job && !isTerminal(job.status) ? false : 20_000,
  });
  const status = statusQuery.data;

  const perfQuery = useQuery({
    queryKey: ['reid-train-performance'],
    queryFn: () => apiFetch<ReidPerfData>(API.reidTrain.performance()),
    enabled: Boolean(status?.totals.labeled_videos),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  // Adopt a job that was already running when the page mounted.
  useEffect(() => {
    if (status?.active_job && !job) setJob(status.active_job);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status?.active_job]);

  useSSE<Job>(job && !isTerminal(job.status) ? API.jobs.eventsSSE(job.id) : null, (data) => {
    setJob(data);
    if (isTerminal(data.status)) {
      if (data.status === 'completed') toast.success(data.message || 'Dataset built');
      else if (data.status === 'failed') toast.error(`Export failed: ${data.error ?? 'unknown error'}`);
      void statusQuery.refetch();
    }
  });

  const startExport = async () => {
    try {
      const started = await apiFetch<Job>(API.reidTrain.start, {
        method: 'POST',
        body: { ...form, name: form.name.trim() || null },
      });
      setJob(started);
    } catch (e) {
      toast.error(`Export failed to start: ${errMsg(e)}`);
    }
  };

  const models = perfQuery.data?.models ?? [];
  const isolated = (status?.sessions ?? []).filter((s) => s.is_isolated);
  const busy = Boolean(job && !isTerminal(job.status));

  return (
    <div className="mx-auto max-w-screen-2xl">
      <PageHeader
        subtitle={
          status
            ? `${status.totals.assigned_events} labeled events across ${status.totals.labeled_videos} video(s)`
            : undefined
        }
        actions={
          <>
            <Button
              size="sm"
              onClick={() => {
                if (!status?.totals.assigned_events) {
                  toast.warning('Nothing labeled yet — assign players on the ReID Label page first');
                  return;
                }
                window.location.href = apiUrl(
                  API.reidTrain.exportPlan({
                    split_mode: form.split_mode,
                    test_ratio: form.test_ratio,
                    seed: form.seed,
                    masked: form.masked,
                  }),
                );
              }}
            >
              Export plan JSONL
            </Button>
            <Button size="sm" intent="primary" disabled={busy || !status?.totals.assigned_events} onClick={() => void startExport()}>
              {busy ? 'Building…' : 'Build dataset'}
            </Button>
          </>
        }
      />

      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Labeled videos" value={status?.totals.labeled_videos ?? '—'} />
        <StatTile label="Assigned events" value={status?.totals.assigned_events ?? '—'} />
        <StatTile label="Identities" value={status?.totals.identities ?? '—'} />
        <StatTile
          label="Sessions"
          value={status?.totals.sessions ?? '—'}
          sub={isolated.length ? `${isolated.length} unlinked` : undefined}
          tintClass={isolated.length ? 'text-amber-400' : 'text-text-primary'}
        />
      </div>

      <div className="space-y-5">
        <Card>
          <SectionLabel>Recording sessions</SectionLabel>
          {!status?.sessions.length ? (
            <EmptyState
              icon={
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.5 20.25a8.25 8.25 0 0115 0" />
                </svg>
              }
              title="No labeled videos"
              subtitle="Assign players on the ReID Label page first"
            />
          ) : (
            <div className="space-y-2">
              {status.sessions.map((s) => (
                <div key={s.id} className="rounded-xl border border-border bg-surface-50 p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-md bg-primary/15 px-2 py-0.5 font-mono text-[11px] font-bold text-primary-light">{s.id}</span>
                    <span className="text-xs text-text-secondary">
                      {s.stems.length} video(s) · {s.players.length} players · {s.n_assigned} events
                    </span>
                    {s.is_isolated && <Badge tone="warning">unlinked</Badge>}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {s.stems.map((stem) => (
                      <span key={stem} className="rounded bg-surface-200/50 px-1.5 py-0.5 font-mono text-[10px] text-text-muted">
                        {stem}
                      </span>
                    ))}
                  </div>
                  <p className="mt-2 text-[11px] text-text-muted">
                    {Object.keys(s.shared).length > 0 ? (
                      <>merged by: <span className="text-text-secondary">{Object.keys(s.shared).join(', ')}</span></>
                    ) : (
                      'shares no player name with any other video — sessions are inferred from shared names, so relabel these consistently to merge them'
                    )}
                  </p>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card>
          <SectionLabel>Embedder comparison</SectionLabel>
          <p className="mb-3 text-[11px] text-text-muted">
            Leave-one-out within each session: every labeled crop queries every other. mAP rewards ranking a
            player's whole set well (what clustering needs); Rank-1 only asks whether the single closest crop is
            the same player.
          </p>
          {perfQuery.isPending ? (
            <div className="py-8 text-center text-xs text-text-muted">Scoring…</div>
          ) : perfQuery.isError ? (
            <p className="text-xs text-red-400">{errMsg(perfQuery.error)}</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[46rem] text-xs">
                <thead className="text-[10px] uppercase tracking-widest text-text-muted">
                  <tr>
                    {['model', 'mAP', 'Rank-1', 'Rank-5', 'ids', 'crops', 'coverage', 'AUC'].map((h, i) => (
                      <th key={h} className={cn('px-2 py-1.5', i ? 'text-right' : 'text-left')}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => <ModelRows key={m.model} m={m} open={openModel === m.model} onToggle={() => setOpenModel(openModel === m.model ? null : m.model)} />)}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card>
          <SectionLabel>Threshold calibration</SectionLabel>
          <p className="mb-3 text-[11px] text-text-muted">
            The cosine-distance cutoff these labels imply (Youden's J over every within-session pair), against
            what <span className="font-mono">EMBEDDER_THRESHOLDS</span> currently ships. Applying it is a source
            edit — this only suggests.
          </p>
          <div className="grid gap-2 lg:grid-cols-2">
            {models.filter((m) => m.threshold).map((m) => <ThresholdCard key={m.model} m={m} />)}
          </div>
        </Card>

        <Card>
          <SectionLabel>Dataset export</SectionLabel>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Field label="Split mode">
              <SelectArch value={form.split_mode} options={status?.split_modes ?? ['auto']} onChange={(v) => set('split_mode', v)} />
            </Field>
            <Field label="Test ratio">
              <input type="number" min={0.05} max={0.9} step={0.05} value={form.test_ratio} onChange={(e) => set('test_ratio', Number(e.target.value))} className={fieldCls} />
            </Field>
            <Field label="Seed">
              <input type="number" value={form.seed} onChange={(e) => set('seed', Number(e.target.value))} className={fieldCls} />
            </Field>
            <Field label="Name (optional)">
              <input value={form.name} onChange={(e) => set('name', e.target.value)} placeholder="reid_<mode>_<timestamp>" className={fieldCls} />
            </Field>
          </div>
          <div className="mt-3 flex flex-wrap gap-4">
            {([
              ['link', 'Symlink crops', 'Link into reid/crops instead of copying — the export is then not movable without --dereference'],
              ['masked', 'Masked crops', 'Link the background-suppressed crops the masked embedders saw'],
              ['overwrite', 'Overwrite', 'Replace an existing dataset of the same name'],
            ] as const).map(([key, label, title]) => (
              <label key={key} className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary" title={title}>
                <input type="checkbox" checked={form[key]} onChange={(e) => set(key, e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
                {label}
              </label>
            ))}
          </div>
          {status?.datasets.length ? (
            <div className="mt-4">
              <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-text-muted">Existing</p>
              <div className="space-y-1">
                {status.datasets.map((d) => (
                  <div key={d.name} className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-50 px-2.5 py-1.5 text-[11px]">
                    <span className="font-mono text-text-primary">{d.name}</span>
                    <span className="text-text-muted">
                      {d.counts.n_rows} rows · {d.counts.n_players} players · train {d.counts.n_train} / test {d.counts.n_test}
                    </span>
                    <span className="ml-auto font-mono text-[10px] text-text-muted">{String(d.config.split_mode ?? '')}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </Card>

        {job && (
          <Card>
            <SectionLabel>Export job</SectionLabel>
            <JobProgress job={job} showLogs />
          </Card>
        )}
      </div>
    </div>
  );
}

function ModelRows({ m, open, onToggle }: { m: ReidModelEval; open: boolean; onToggle: () => void }) {
  const w = m.crop_weighted;
  if (!w || !m.totals) {
    return (
      <tr className="border-t border-border">
        <td className="px-2 py-1.5 font-mono text-text-secondary">{m.model}</td>
        <td colSpan={7} className="px-2 py-1.5 text-right text-text-muted">
          not embedded for {m.skipped.length} session(s)
        </td>
      </tr>
    );
  }
  return (
    <>
      <tr className="cursor-pointer border-t border-border hover:bg-surface-200/30" onClick={onToggle}>
        <td className="px-2 py-1.5 font-mono text-text-primary">
          <span className={cn('mr-1 inline-block transition-transform', open && 'rotate-90')}>▸</span>
          {m.model}
        </td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-primary-light">{pct(w.m_ap)}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums">{pct(w.rank1)}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-secondary">{pct(w.rank5)}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{m.totals.n_ids}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{m.totals.n_crops}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{pct(m.totals.coverage)}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{m.threshold?.auc.toFixed(3)}</td>
      </tr>
      {open && m.groups.map((g) => <GroupRow key={g.group_id} g={g} />)}
    </>
  );
}

function GroupRow({ g }: { g: ReidGroupEval }) {
  return (
    <tr className="border-t border-border/50 bg-surface-100/40 text-[11px]">
      <td className="px-2 py-1 pl-7 text-text-muted">
        <span className="font-mono">{g.group_id}</span> {g.stems.join(', ')}
      </td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-secondary">{pct(g.scores.m_ap)}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-secondary">{pct(g.scores.rank1)}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted">{pct(g.scores.rank5)}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted">{g.n_ids}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted">{g.n_crops}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted" title={`${g.dropped_unembedded} assigned events had no embedding`}>
        {pct(g.coverage)}
      </td>
      <td className="px-2 py-1 text-right text-text-muted" title="Query = first video, gallery = the rest. Needs a session spanning more than one video.">
        {g.cross_video ? `x-video ${pct(g.cross_video.rank1)}` : '—'}
      </td>
    </tr>
  );
}

function ThresholdCard({ m }: { m: ReidModelEval }) {
  const t = m.threshold!;
  const cur = m.current_threshold;
  const changed = (k: keyof typeof cur) => t.slider[k] !== cur[k];
  const json = useMemo(() => `"${m.model}": ${JSON.stringify(t.slider)},`, [m.model, t.slider]);
  return (
    <div className="rounded-xl border border-border bg-surface-50 p-3">
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-xs text-text-primary">{m.model}</span>
        <span className="text-[10px] text-text-muted">
          ARI {t.ari.toFixed(3)} · {t.n_clusters} groups for {t.n_ids} players · AUC {t.auc.toFixed(3)}
        </span>
        <button
          type="button"
          onClick={() => {
            void navigator.clipboard?.writeText(json);
            toast.success('Copied');
          }}
          className="ml-auto rounded-md border border-border px-2 py-0.5 text-[10px] text-text-secondary transition-colors hover:bg-ink/[0.06]"
        >
          Copy
        </button>
      </div>
      <table className="w-full text-[11px]">
        <tbody>
          {(['min', 'max', 'default', 'step'] as const).map((k) => (
            <tr key={k}>
              <td className="py-0.5 text-text-muted">{k}</td>
              <td className="py-0.5 text-right font-mono tabular-nums text-text-muted">{cur[k]}</td>
              <td className="w-4 text-center text-text-muted">→</td>
              <td className={cn('py-0.5 text-right font-mono tabular-nums', changed(k) ? 'text-amber-400' : 'text-text-secondary')}>
                {t.slider[k]}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <AriCurve t={t} current={cur.default} />
      <p className="mt-1 text-[10px] text-text-muted">
        Over-splitting is the safe side: merging two groups of one player is a drag, a group holding
        two players has to be spotted first.
      </p>
    </div>
  );
}

/** Cluster quality (ARI) against the threshold, over the band that still
 *  clusters usefully. The peak is the suggestion; past it average-linkage
 *  chains and the whole board collapses into a few mixed groups. Inline SVG,
 *  same technique as components/train/TrainPerfCard.tsx. */
function AriCurve({ t, current }: { t: NonNullable<ReidModelEval['threshold']>; current: number }) {
  const pts = t.curve;
  if (pts.length < 2) return null;
  const W = 260;
  const H = 56;
  const x0 = pts[0]!.t;
  const x1 = pts[pts.length - 1]!.t;
  const span = x1 - x0 || 1;
  const X = (v: number) => ((v - x0) / span) * W;
  const Y = (v: number) => H - v * H;
  const line = pts.map((p) => `${X(p.t).toFixed(1)},${Y(p.ari).toFixed(1)}`).join(' ');
  const peak = pts.reduce((a, b) => (a.ari >= b.ari ? a : b));
  const inRange = current >= x0 && current <= x1;
  return (
    <div className="mt-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-14 w-full" preserveAspectRatio="none">
        <polyline points={line} fill="none" stroke="currentColor" strokeWidth={1.5} className="text-primary-light" vectorEffect="non-scaling-stroke" />
        <line x1={X(peak.t)} x2={X(peak.t)} y1={0} y2={H} stroke="currentColor" strokeWidth={1} className="text-amber-400" vectorEffect="non-scaling-stroke" />
        {inRange && (
          <line x1={X(current)} x2={X(current)} y1={0} y2={H} strokeDasharray="3 3" stroke="currentColor" strokeWidth={1} className="text-text-muted" vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      <div className="flex justify-between text-[9px] text-text-muted">
        <span>{x0}</span>
        <span className="text-amber-400">peak {t.suggested} · ARI {t.ari.toFixed(2)}</span>
        <span>{x1}</span>
      </div>
      <p className="text-[9px] text-text-muted">
        {inRange ? 'dashed = current default' : `current default ${current} sits outside the useful band`}
      </p>
    </div>
  );
}
