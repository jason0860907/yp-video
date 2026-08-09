/** ReID Train: measure the embedders on the labels we have, export a yp-reid
 *  dataset, and fine-tune the appearance model on it.
 *
 *  The ordering on the page mirrors the workflow: the embedder comparison is
 *  the baseline a fine-tune has to beat, the export turns labels into a
 *  Contract A dataset, and the fine-tune card spawns yp-reid training on one.
 *  Every new best rewrites its candidate package under reid/checkpoints/.
 *  Production stays on the official paper checkpoint unless a future
 *  explicit promotion workflow changes that policy.
 */

import { useMemo, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import reidExportSchema from '@contracts/reid_export_request.schema.json';
import reidTrainSchema from '@contracts/reid_train_request.schema.json';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { useSchemaForm } from '@/lib/schemaForm';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { fieldCls } from '@/components/train/Field';
import { SchemaForm } from '@/components/form/SchemaForm';
import {
  SchemaCheckboxField,
  SchemaNumberField,
  SchemaSearchSelectField,
  SchemaSelectField,
  SchemaTextField,
} from '@/components/form/SchemaFields';
import { FieldShell } from '@/components/form/FieldLabel';
import { JobProgress } from '@/components/job/JobProgress';
import { toast } from '@/components/feedback/toast';
import { useSingleJob } from '@/lib/useSingleJob';
import type { Job, ReidModelEval, ReidPerfData, ReidRun, ReidTrainStatus, ReidVideoEval } from '@/types/api';
import type { ReidExportRequest } from '@/types/contracts/reid_export_request.schema';
import type { ReidTrainRequest } from '@/types/contracts/reid_train_request.schema';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

type ExportForm = Required<ReidExportRequest>;
type TrainForm = Required<ReidTrainRequest>;

export function ReidTrainPage() {
  const exportForm = useSchemaForm<ExportForm>(reidExportSchema);
  // dataset: '' = newest export, resolved at submit.
  const trainForm = useSchemaForm<TrainForm>(reidTrainSchema, { dataset: '' });
  const [openModel, setOpenModel] = useState<string | null>(null);

  const statusQuery = useQuery({
    queryKey: ['reid-train-status'],
    queryFn: () => apiFetch<ReidTrainStatus>(API.reidTrain.status),
  });
  const status = statusQuery.data;

  const { job, setJob, running: busy, cancel: cancelJob } = useSingleJob({
    activeJob: status?.active_job,
    label: (settled) => (settled.type === 'reid_train' ? 'Training' : 'Export'),
  });

  const perfQuery = useQuery({
    queryKey: ['reid-train-performance'],
    queryFn: () => apiFetch<ReidPerfData>(API.reidTrain.performance()),
    enabled: Boolean(status?.totals.ready_videos),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  const startExport = async () => {
    try {
      const started = await apiFetch<Job>(API.reidTrain.export, {
        method: 'POST',
        body: exportForm.values,
      });
      setJob(started);
    } catch (e) {
      toast.error(`Export failed to start: ${errMsg(e)}`);
    }
  };

  // Download the export plan (writes nothing) so the split can be inspected
  // before committing to a dataset build.
  const downloadPlan = () => {
    if (!status?.totals.assigned_events) {
      toast.warning(
        status?.totals.pending_videos
          ? 'No finished videos — press Done on the ReID Label page to include them'
          : 'Nothing labeled yet — assign players on the ReID Label page first',
      );
      return;
    }
    const { split_mode, test_ratio, seed, masked } = exportForm.values;
    window.location.href = apiUrl(
      API.reidTrain.exportPlan({ split_mode, test_ratio, seed, masked }),
    );
  };

  const startTrain = async () => {
    const dataset = trainForm.values.dataset || status?.datasets[0]?.name;
    if (!dataset) return;
    try {
      const started = await apiFetch<Job>(API.reidTrain.train, {
        method: 'POST',
        body: { ...trainForm.values, dataset },
      });
      setJob(started);
      toast.success('ReID training started');
    } catch (e) {
      toast.error(`Training failed to start: ${errMsg(e)}`);
    }
  };

  const models = perfQuery.data?.models ?? [];

  return (
    <div className="mx-auto max-w-screen-2xl">
      <PageHeader
        subtitle={
          status
            ? `${status.totals.assigned_events} events across ${status.totals.ready_videos} finished video(s)`
            : undefined
        }
      />

      <div className="mb-5 space-y-3">
        {status?.totals.pending_videos ? (
          <div className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/[0.06] px-3 py-2 text-xs text-amber-300">
            <svg className="mt-0.5 h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0 3.75h.008M10.34 3.94l-8.4 14.55A1.5 1.5 0 003.24 21h17.52a1.5 1.5 0 001.3-2.51L13.66 3.94a1.5 1.5 0 00-2.6 0z" />
            </svg>
            <span>
              <span className="font-semibold">{status.totals.pending_videos}</span> labeled video(s) are not marked done —
              excluded from training &amp; scoring. Press <span className="font-semibold">Done</span> on the ReID Label
              page to include them.
            </span>
          </div>
        ) : null}
      </div>

      <div className="space-y-5">
        <Card>
          <SectionLabel>① Dataset export</SectionLabel>
          <p className="mb-3 text-[11px] text-text-muted">
            Pack the finished videos' labeled crops into a yp-reid dataset. Preview the split with
            <span className="font-medium"> Export plan JSONL</span> (writes nothing), then
            <span className="font-medium"> Build dataset</span> to commit it under reid/datasets/.
          </p>
          <SchemaForm form={exportForm}>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <SchemaSelectField name="split_mode" label="Split mode" />
              <SchemaNumberField name="test_ratio" label="Test ratio" step={0.05} />
              <SchemaNumberField name="seed" label="Seed" />
              <SchemaTextField name="name" label="Name (optional)" placeholder="reid_<mode>_<timestamp>" />
            </div>
            <div className="mt-3 flex flex-wrap gap-4">
              <SchemaCheckboxField name="masked" label="Masked crops" />
              <SchemaCheckboxField name="overwrite" label="Overwrite" />
            </div>
          </SchemaForm>
          <div className="mt-4 flex items-center gap-2">
            <Button className="flex-1" disabled={!status?.totals.assigned_events} onClick={downloadPlan}>
              Export plan JSONL
            </Button>
            <Button intent="primary" className="flex-1" disabled={busy || !status?.totals.assigned_events} onClick={() => void startExport()}>
              {busy && job?.type === 'reid_dataset_export' ? 'Building…' : 'Build dataset'}
            </Button>
          </div>
          {status?.datasets.length ? (
            <div className="mt-4">
              <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-text-muted">Existing datasets</p>
              <div className="space-y-1">
                {status.datasets.map((d) => (
                  <div key={d.name} className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-50 px-2.5 py-1.5 text-[11px]">
                    <span className="font-mono text-text-primary">{d.name}</span>
                    <span className="text-text-muted">
                      {d.counts.n_samples} crops · {d.counts.n_players} players · train {d.counts.n_train} / test {d.counts.n_test}
                    </span>
                    <span className="ml-auto font-mono text-[10px] text-text-muted">{String(d.config.split_mode ?? '')}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </Card>

        <Card>
          <SectionLabel>② Fine-tune</SectionLabel>
          <p className="mb-3 text-[11px] text-text-muted">
            Spawns yp-reid training on an exported dataset (GPU-locked job). Every new best rewrites the
            checkpoint package, so even a cancelled run leaves its best-so-far usable — the clip-reident
            embedder rebinds to whichever package leads the runs table below.
          </p>
          <SchemaForm form={trainForm}>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              {/* '' falls back to the newest export at submit — runtime data
                  the contract's plain string field can't express. */}
              <FieldShell label="Dataset" description={trainForm.fields.dataset?.description}>
                <select
                  value={trainForm.values.dataset || status?.datasets[0]?.name || ''}
                  onChange={(e) => trainForm.set('dataset', e.target.value)}
                  className={cn(fieldCls, 'cursor-pointer appearance-none')}
                >
                  {(status?.datasets ?? []).map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </FieldShell>
              <SchemaSearchSelectField
                name="init_checkpoint"
                label="Init checkpoint"
                options={(status?.runs ?? []).map((r) => ({ value: r.path, label: r.run_name }))}
                placeholder="fresh (OpenAI ViT-L/14)"
              />
              <SchemaTextField name="run_name" label="Run name (optional)" placeholder="reid_<timestamp>" />
              <SchemaNumberField name="epochs" label="Epochs" />
              <SchemaNumberField name="batch_size" label="Batch size (≤ identities)" />
              <SchemaNumberField name="lr" label="Learning rate" />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-4">
              <SchemaCheckboxField name="overwrite" label="Overwrite" />
              {!status?.reid_engine_available && (
                <span className="text-[11px] text-amber-400">
                  clip-reident engine not registered — needs the yp-reid venv and at least one checkpoint package
                </span>
              )}
            </div>
          </SchemaForm>
          <div className="mt-4 flex items-center gap-2">
            <Button intent="primary" className="flex-1" disabled={busy || !status?.datasets.length} onClick={() => void startTrain()}>
              {busy && job?.type === 'reid_train' ? 'Training…' : 'Start Training'}
            </Button>
            {busy && <Button onClick={() => void cancelJob()}>Cancel</Button>}
          </div>
        </Card>

        {status?.runs.length ? <RunsCard runs={status.runs} /> : null}

        {job && (
          <Card>
            <SectionLabel>{job.type === 'reid_train' ? 'Training job' : 'Export job'}</SectionLabel>
            <JobProgress job={job} showLogs />
            {job.type === 'reid_train' && <TrainEvalTiles job={job} />}
          </Card>
        )}

        <div className="pt-3">
          <h2 className="text-sm font-semibold text-text-secondary">Diagnostics</h2>
          <p className="mt-0.5 text-[11px] text-text-muted">
            Reference only — how well the current embedder separates your finished labels, and the sessions those
            labels form. Nothing here changes what trains.
          </p>
        </div>


        <Card>
          <SectionLabel>Recording sessions</SectionLabel>
          {!status?.sessions.length ? (
            <EmptyState
              icon={
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.5 20.25a8.25 8.25 0 0115 0" />
                </svg>
              }
              title="No finished videos"
              subtitle="Mark labeling done on the ReID Label page — only finished cuts train"
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
            Leave-one-out within each video — the labeling unit: every labeled crop queries every other crop of
            the same cut. mAP rewards ranking a player's whole set well (what clustering needs); Rank-1 only asks
            whether the single closest crop is the same player. Click a row for the per-video breakdown.
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
                    {['model', 'mAP', 'Rank-1', 'Rank-5', 'videos', 'crops', 'coverage', 'AUC'].map((h, i) => (
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

        <CrossVideoCard models={models} />

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
      </div>
    </div>
  );
}

/** Per-epoch eval snapshots the trainer streams (job.params.last_eval / .best).
 *  Not the SPOT TrainProgress shape — ReID eval is retrieval, not spotting. */
function TrainEvalTiles({ job }: { job: Job }) {
  const evalData = job.params?.last_eval as { epoch?: number; m_ap?: number; rank1?: number; rank5?: number } | undefined;
  const best = job.params?.best as { epoch?: number; value?: number } | undefined;
  if (!evalData && !best) return null;
  const tiles: Array<[string, string]> = [];
  if (evalData?.m_ap != null) tiles.push([`mAP (epoch ${evalData.epoch})`, pct(evalData.m_ap)]);
  if (evalData?.rank1 != null) tiles.push(['Rank-1', pct(evalData.rank1)]);
  if (evalData?.rank5 != null) tiles.push(['Rank-5', pct(evalData.rank5)]);
  // Without a test split the trainer tracks train loss instead of mAP.
  if (best?.value != null) tiles.push([`Best (epoch ${best.epoch})`, evalData ? pct(best.value) : best.value.toFixed(4)]);
  return (
    <div className="mt-3 grid grid-cols-2 gap-3 lg:grid-cols-4">
      {tiles.map(([label, value]) => <StatTile key={label} label={label} value={value} />)}
    </div>
  );
}

/** Stable official checkpoint followed by unpromoted training candidates. */
function RunsCard({ runs }: { runs: ReidRun[] }) {
  return (
    <Card>
      <SectionLabel>Checkpoint runs</SectionLabel>
      <p className="mb-3 text-[11px] text-text-muted">
        The official paper release remains active. Training outputs are candidates and never replace it
        automatically; select one explicitly on Predict to evaluate it, then recalibrate thresholds because
        distance scales move with the weights.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[40rem] text-xs">
          <thead className="text-[10px] uppercase tracking-widest text-text-muted">
            <tr>
              {['run', 'source', 'best', 'Rank-1', 'dim', 'created'].map((h, i) => (
                <th key={h} className={cn('px-2 py-1.5', i ? 'text-right' : 'text-left')}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.path} className="border-t border-border">
                <td className="px-2 py-1.5 font-mono text-text-primary" title={r.note ?? r.path}>
                  {r.run_name}
                  {r.active && <Badge tone="success" className="ml-2">active</Badge>}
                  {!r.active && <Badge className="ml-2">candidate</Badge>}
                </td>
                <td className="px-2 py-1.5 text-right text-text-muted">{r.source ?? '—'}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-primary-light">
                  {r.best_value == null
                    ? '—'
                    : `${r.best_metric} ${r.best_metric === 'train_loss' ? r.best_value.toFixed(4) : pct(r.best_value)}`}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-secondary">
                  {r.metrics?.rank1 != null ? pct(r.metrics.rank1) : '—'}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{r.embedding_dim ?? '—'}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{r.created_at ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
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
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{m.totals.n_videos}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{m.totals.n_crops}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{pct(m.totals.coverage)}</td>
        <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{m.threshold?.auc.toFixed(3)}</td>
      </tr>
      {open && m.videos.map((v) => <VideoRow key={v.stem} v={v} />)}
    </>
  );
}

function VideoRow({ v }: { v: ReidVideoEval }) {
  return (
    <tr className="border-t border-border/50 bg-surface-100/40 text-[11px]">
      <td className="truncate px-2 py-1 pl-7 font-mono text-text-muted" title={v.stem}>{v.stem}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-secondary">{pct(v.scores.m_ap)}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-secondary">{pct(v.scores.rank1)}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted">{pct(v.scores.rank5)}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted" title="players scored">{v.n_ids}</td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted">{v.n_crops}</td>
      <td
        className={cn('px-2 py-1 text-right font-mono tabular-nums', v.coverage < 0.95 ? 'text-amber-400' : 'text-text-muted')}
        title={`${v.dropped_unembedded} assigned events have no embedding (no-actor or miss), ${v.dropped_singletons} players had a single crop`}
      >
        {pct(v.coverage)}
      </td>
      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-muted">{v.threshold.suggested}</td>
    </tr>
  );
}

/** Cross-video: the same session's other recordings as the gallery. Separate
 *  from the main table because it answers a different question — not "can
 *  this cut be clustered" but "does an identity survive into the next one". */
function CrossVideoCard({ models }: { models: ReidModelEval[] }) {
  const rows = models.flatMap((m) => m.cross_video.map((c) => ({ model: m.model, c })));
  if (!rows.length) return null;
  const sample = rows[0]!.c;
  return (
    <Card>
      <SectionLabel>Cross-video</SectionLabel>
      <p className="mb-3 text-[11px] text-text-muted">
        Query is <span className="font-mono">{sample.query_stem}</span>, gallery the session's other
        recording(s). Only players named on both sides can be scored — {sample.n_skipped} of{' '}
        {sample.n_scored + sample.n_skipped} queries were skipped because that player simply was not on court
        for the other cut, which is normal and not a labeling gap.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[34rem] text-xs">
          <thead className="text-[10px] uppercase tracking-widest text-text-muted">
            <tr>
              {['model', 'mAP', 'Rank-1', 'Rank-5', 'shared ids', 'scored'].map((h, i) => (
                <th key={h} className={cn('px-2 py-1.5', i ? 'text-right' : 'text-left')}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...rows].sort((a, b) => b.c.scores.m_ap - a.c.scores.m_ap).map(({ model, c }) => (
              <tr key={`${model}-${c.session_id}`} className="border-t border-border">
                <td className="px-2 py-1.5 font-mono text-text-primary">{model}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-primary-light">{pct(c.scores.m_ap)}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums">{pct(c.scores.rank1)}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-secondary">{pct(c.scores.rank5)}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{c.n_ids_shared}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-muted">{c.n_scored}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
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
