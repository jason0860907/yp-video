import { useEffect, useRef, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { API, apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { useSingleJob } from '@/lib/useSingleJob';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { Field, SelectArch, fieldCls } from '@/components/train/Field';
import { TrainJobCard } from '@/components/train/TrainJobCard';
import { TrainPerfCard } from '@/components/train/TrainPerfCard';
import { toast } from '@/components/feedback/toast';
import type {
  ActionPerfData,
  FusionModelStatus,
  FusionRecipeId,
  Job,
} from '@/types/api';

interface TrainForm {
  run_name: string;
  resume_run: string;
  init_checkpoint: string;
  camera_view: 'all' | 'broadcast' | 'sideline';
  audio_backend: 'logmel' | 'none';
  feature_arch: string;
  temporal_arch: string;
  clip_len: number;
  sample_fps: number;
  batch_size: number;
  num_epochs: number;
  warm_up_epochs: number;
  learning_rate: number;
  num_workers: number;
  val_ratio: number;
  split_seed: number;
  gpu: number;
  stop_vllm: boolean;
}

const TRAIN_DEFAULTS: TrainForm = {
  run_name: '',
  resume_run: '',
  init_checkpoint: '',
  camera_view: 'all',
  audio_backend: 'logmel',
  feature_arch: 'rny008_gsm',
  temporal_arch: 'gru',
  clip_len: 64,
  sample_fps: 30,
  batch_size: 8,
  num_epochs: 50,
  warm_up_epochs: 3,
  learning_rate: 0.00003,
  num_workers: 4,
  val_ratio: 0.2,
  split_seed: 42,
  gpu: 0,
  stop_vllm: false,
};

const TRAIN_NUMBERS: Array<{
  key: keyof TrainForm;
  label: string;
  min: number;
  max?: number;
  step?: number;
}> = [
  { key: 'clip_len', label: 'Clip len', min: 8, max: 256 },
  { key: 'sample_fps', label: 'Sample fps', min: 0, max: 120 },
  { key: 'batch_size', label: 'Batch', min: 1, max: 64 },
  { key: 'num_epochs', label: 'Epochs', min: 1, max: 1000 },
  { key: 'warm_up_epochs', label: 'Warmup', min: 0, max: 100 },
  { key: 'learning_rate', label: 'Learning rate', min: 0, step: 0.00001 },
  { key: 'num_workers', label: 'Workers', min: 0, max: 32 },
  { key: 'gpu', label: 'GPU', min: 0, max: 7 },
];

const warningCls =
  'rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-[11px] leading-relaxed text-amber-400';

export function FusionTrainPage() {
  const [recipe, setRecipe] = useState<FusionRecipeId>('association_action');
  const [trainForm, setTrainForm] = useState<TrainForm>(TRAIN_DEFAULTS);
  const [perfRun, setPerfRun] = useState('');
  const [datasetScope, setDatasetScope] = useState<
    'joint_only' | 'partial_labels'
  >('joint_only');
  const [validationMode, setValidationMode] = useState<'manual' | 'ratio'>(
    'manual',
  );
  const [validationVideos, setValidationVideos] = useState<Set<string>>(
    new Set(),
  );

  const statusQuery = useQuery({
    queryKey: ['fusion-model-status'],
    queryFn: () => apiFetch<FusionModelStatus>(API.fusionModel.status),
  });
  const status = statusQuery.data;

  const {
    job: trainJob,
    setJob: setTrainJob,
    running: trainingRunning,
    cancel: cancelTrain,
  } = useSingleJob({
    activeJob: status?.active_job,
    label: 'Fusion training',
  });

  const performanceQuery = useQuery({
    queryKey: ['fusion-model-performance', perfRun],
    queryFn: () =>
      apiFetch<ActionPerfData>(
        perfRun
          ? `${API.fusionModel.performance}?run=${encodeURIComponent(perfRun)}`
          : API.fusionModel.performance,
      ),
    refetchInterval: trainingRunning ? 30_000 : false,
    placeholderData: keepPreviousData,
  });

  const chosenRecipe = status?.recipes.find((item) => item.id === recipe);
  const recipeBlocked = Boolean(chosenRecipe && !chosenRecipe.available);
  const annotations = status?.action_annotations;
  const isResuming = Boolean(trainForm.resume_run);
  const validationSeeded = useRef(false);
  const validationChoices = (annotations?.per_video ?? []).filter(
    (video) =>
      (trainForm.camera_view === 'all' ||
        video.view === trainForm.camera_view) &&
      (datasetScope === 'partial_labels' || video.has_association_label),
  );
  const eligibleVideos = validationChoices.length;
  const eligibleEvents = validationChoices.reduce(
    (total, video) => total + video.events,
    0,
  );
  const selectedValidationCount = validationChoices.filter((video) =>
    validationVideos.has(video.video),
  ).length;
  const selectedValidationEvents = validationChoices
    .filter((video) => validationVideos.has(video.video))
    .reduce((total, video) => total + video.events, 0);
  const canTrain =
    Boolean(status?.spot_available) &&
    !recipeBlocked &&
    !trainingRunning &&
    (isResuming ||
      validationMode === 'ratio' ||
      selectedValidationCount > 0);

  useEffect(() => {
    if (validationSeeded.current || !annotations?.per_video?.length) return;
    setValidationVideos(
      new Set(
        annotations.per_video
          .filter((video) => video.is_val)
          .map((video) => video.video),
      ),
    );
    validationSeeded.current = true;
  }, [annotations?.per_video]);

  const setTrain = <K extends keyof TrainForm>(key: K, value: TrainForm[K]) =>
    setTrainForm((form) => ({ ...form, [key]: value }));

  const startTrain = async () => {
    if (
      !isResuming &&
      validationMode === 'manual' &&
      selectedValidationCount === 0
    ) {
      toast.warning(
        'Select at least one validation video for this camera view',
      );
      return;
    }
    try {
      const job = await apiFetch<Job>(API.fusionModel.train, {
        method: 'POST',
        body: {
          recipe,
          ...trainForm,
          validation_mode: validationMode,
          dataset_scope: datasetScope,
          validation_videos:
            validationMode === 'manual'
              ? validationChoices
                  .filter((video) => validationVideos.has(video.video))
                  .map((video) => `${video.video}_actions.jsonl`)
              : [],
          run_name: trainForm.run_name.trim() || null,
          init_checkpoint: trainForm.init_checkpoint || null,
        },
      });
      setPerfRun('');
      setTrainJob(job);
      toast.success('Fusion training started');
    } catch (error) {
      toast.error(`Fusion training failed to start: ${errMsg(error)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <Button
            intent="primary"
            onClick={() => void startTrain()}
            disabled={!canTrain}
          >
            {trainingRunning ? 'Training…' : 'Train fusion checkpoint'}
          </Button>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile
          label="Eligible videos"
          value={eligibleVideos}
          tintClass="text-primary-light"
        />
        <StatTile
          label="Action events"
          value={eligibleEvents.toLocaleString()}
          tintClass="text-primary-light"
        />
        <StatTile
          label="Validation videos"
          value={
            isResuming
              ? 'frozen'
              : validationMode === 'manual'
                ? selectedValidationCount
                : `${Math.round(trainForm.val_ratio * 100)}%`
          }
          tintClass="text-primary-light"
        />
        <StatTile
          label="Status"
          value={
            trainingRunning
              ? 'training'
              : eligibleVideos
                ? 'ready'
                : 'not ready'
          }
          tintClass={
            trainingRunning || eligibleVideos
              ? 'text-primary-light'
              : 'text-amber-400'
          }
        />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <p className="mb-4 text-xs leading-relaxed text-text-muted">
            Trains action classification, contact location and actor selection
            in one SPOT checkpoint. Actor supervision comes from reviewed
            Association Label records. The default joint-only scope requires
            every included video to supervise both tasks; partial-label union
            is available as an explicit experiment.
          </p>
          <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
            <Field label="Recipe" className="col-span-3">
              <select
                value={recipe}
                onChange={(event) =>
                  setRecipe(event.target.value as FusionRecipeId)
                }
                className={cn(fieldCls, 'cursor-pointer appearance-none')}
              >
                {(status?.recipes ?? []).map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                    {item.available ? '' : ' — planned'}
                  </option>
                ))}
              </select>
              {recipeBlocked && chosenRecipe ? (
                <span className={cn(warningCls, 'mt-1 block')}>
                  {chosenRecipe.name} is planned but not trainable yet:{' '}
                  {chosenRecipe.blocked_on}
                </span>
              ) : null}
            </Field>
            <Field label="Run name" className="col-span-2">
              <input
                value={trainForm.run_name}
                onChange={(event) =>
                  setTrain('run_name', event.target.value)
                }
                placeholder="yp_fusion_association_action_..."
                disabled={isResuming}
                className={cn(fieldCls, isResuming && 'opacity-50')}
              />
            </Field>
            <Field label="Camera view">
              <select
                value={trainForm.camera_view}
                disabled={isResuming}
                onChange={(event) =>
                  setTrain(
                    'camera_view',
                    event.target.value as TrainForm['camera_view'],
                  )
                }
                className={cn(
                  fieldCls,
                  'cursor-pointer appearance-none',
                  isResuming && 'opacity-50',
                )}
              >
                <option value="all">All</option>
                <option value="broadcast">Broadcast</option>
                <option value="sideline">Sideline</option>
              </select>
            </Field>
            <Field label="Dataset scope" className="col-span-3">
              <select
                value={datasetScope}
                disabled={isResuming}
                onChange={(event) =>
                  setDatasetScope(
                    event.target.value as 'joint_only' | 'partial_labels',
                  )
                }
                className={cn(
                  fieldCls,
                  'cursor-pointer appearance-none',
                  isResuming && 'opacity-50',
                )}
              >
                <option value="joint_only">
                  Joint supervision only — Action ∩ Association (
                  {status?.supervision.joint_videos ?? 0} videos)
                </option>
                <option value="partial_labels">
                  Partial-label union — all Action videos (
                  {status?.supervision.action_videos ?? 0} videos)
                </option>
              </select>
              <span className="block text-[10px] leading-relaxed text-text-muted">
                {datasetScope === 'joint_only'
                  ? 'Every training video must produce actor targets; missing Association supervision fails the run.'
                  : `${status?.supervision.action_only_videos ?? 0} Action-only videos update the shared backbone and Action head, while their actor loss is masked.`}
              </span>
            </Field>
            <Field label="Init checkpoint" className="col-span-2">
              <select
                value={trainForm.init_checkpoint}
                disabled={isResuming}
                onChange={(event) =>
                  setTrain('init_checkpoint', event.target.value)
                }
                className={cn(
                  fieldCls,
                  'cursor-pointer appearance-none',
                  isResuming && 'opacity-50',
                )}
              >
                <option value="">— From scratch —</option>
                {(status?.init_checkpoints ?? []).map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Audio">
              <select
                value={trainForm.audio_backend}
                disabled={isResuming}
                onChange={(event) =>
                  setTrain(
                    'audio_backend',
                    event.target.value as TrainForm['audio_backend'],
                  )
                }
                className={cn(
                  fieldCls,
                  'cursor-pointer appearance-none',
                  isResuming && 'opacity-50',
                )}
              >
                <option value="logmel">Log-mel fusion</option>
                <option value="none">Visual only</option>
              </select>
            </Field>
            <Field label="Resume joint-head run" className="col-span-3">
              <select
                value={trainForm.resume_run}
                onChange={(event) =>
                  setTrain('resume_run', event.target.value)
                }
                className={cn(fieldCls, 'cursor-pointer appearance-none')}
              >
                <option value="">— New fusion run —</option>
                {(status?.resumable_runs ?? []).map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field
              label="Feature"
              className={
                isResuming ? 'pointer-events-none opacity-50' : undefined
              }
            >
              <SelectArch
                value={trainForm.feature_arch}
                options={['rny008_gsm', 'rny002_gsm', 'rny008', 'rny002']}
                onChange={(value) => setTrain('feature_arch', value)}
              />
            </Field>
            <Field
              label="Temporal"
              className={
                isResuming ? 'pointer-events-none opacity-50' : undefined
              }
            >
              <SelectArch
                value={trainForm.temporal_arch}
                options={['gru', 'deeper_gru', 'mingru']}
                onChange={(value) => setTrain('temporal_arch', value)}
              />
            </Field>
            {TRAIN_NUMBERS.map((field) => (
              <Field key={field.key} label={field.label}>
                <input
                  type="number"
                  value={trainForm[field.key] as number}
                  min={field.min}
                  max={field.max}
                  step={field.step ?? 1}
                  disabled={
                    isResuming &&
                    (field.key === 'clip_len' || field.key === 'sample_fps')
                  }
                  onChange={(event) =>
                    setTrain(
                      field.key,
                      Number(event.target.value) as never,
                    )
                  }
                  className={cn(
                    fieldCls,
                    'font-mono tabular-nums',
                    isResuming &&
                      (field.key === 'clip_len' ||
                        field.key === 'sample_fps') &&
                      'opacity-50',
                  )}
                />
              </Field>
            ))}
          </div>
          {!isResuming ? (
            <div className="mt-5 border-t border-border pt-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <SectionLabel className="mb-0">Validation split</SectionLabel>
                <div className="flex gap-2">
                  {(['manual', 'ratio'] as const).map((mode) => (
                    <Button
                      key={mode}
                      size="sm"
                      intent={validationMode === mode ? 'primary' : 'default'}
                      onClick={() => setValidationMode(mode)}
                    >
                      {mode === 'manual' ? 'Select videos' : 'Random ratio'}
                    </Button>
                  ))}
                </div>
              </div>
              {validationMode === 'manual' ? (
                <>
                  <div className="mb-2 flex items-center justify-between text-[11px] text-text-muted">
                    <span>
                      {selectedValidationCount} validation /{' '}
                      {validationChoices.length} in this camera view
                    </span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() =>
                          setValidationVideos(
                            new Set(
                              (annotations?.per_video ?? [])
                                .filter((video) => video.is_val)
                                .map((video) => video.video),
                            ),
                          )
                        }
                        className="text-primary-light hover:underline"
                      >
                        Load saved val set
                      </button>
                      <button
                        type="button"
                        onClick={() => setValidationVideos(new Set())}
                        className="text-text-muted hover:underline"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                  <div className="max-h-64 space-y-1 overflow-auto rounded-lg border border-border bg-surface-50 p-2">
                    {validationChoices.map((video) => (
                      <label
                        key={video.video}
                        className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 hover:bg-ink/[0.03]"
                      >
                        <input
                          type="checkbox"
                          checked={validationVideos.has(video.video)}
                          onChange={(event) => {
                            const next = new Set(validationVideos);
                            if (event.target.checked) next.add(video.video);
                            else next.delete(video.video);
                            setValidationVideos(next);
                          }}
                          className="h-3.5 w-3.5 flex-shrink-0 accent-primary"
                        />
                        <span className="min-w-0 flex-1 truncate text-xs text-text-secondary">
                          {video.video}
                        </span>
                        <Badge tone="neutral">{video.view}</Badge>
                        {video.has_association_label ? (
                          <Badge tone="success">association</Badge>
                        ) : (
                          <Badge tone="warning">action only</Badge>
                        )}
                        <span className="font-mono text-[10px] tabular-nums text-text-muted">
                          {video.events}
                        </span>
                      </label>
                    ))}
                  </div>
                  <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
                    Selected videos are validation only; every other matching
                    action annotation trains. The exact split is frozen into
                    the run for Resume.
                  </p>
                </>
              ) : (
                <div className="rounded-lg border border-border bg-surface-50 p-3">
                  <div className="grid grid-cols-2 gap-2.5">
                    <Field label="Validation ratio">
                      <input
                        type="number"
                        value={trainForm.val_ratio}
                        min={0.01}
                        max={0.99}
                        step={0.05}
                        onChange={(event) =>
                          setTrain('val_ratio', Number(event.target.value))
                        }
                        className={cn(fieldCls, 'font-mono tabular-nums')}
                      />
                    </Field>
                    <Field label="Split seed">
                      <input
                        type="number"
                        value={trainForm.split_seed}
                        min={0}
                        step={1}
                        onChange={(event) =>
                          setTrain('split_seed', Number(event.target.value))
                        }
                        className={cn(fieldCls, 'font-mono tabular-nums')}
                      />
                    </Field>
                  </div>
                  <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
                    A deterministic {Math.round(trainForm.val_ratio * 100)}%
                    split will be generated with seed {trainForm.split_seed}.
                  </p>
                </div>
              )}
            </div>
          ) : null}
          {isResuming ? (
            <p className="mt-3 rounded-lg border border-primary/20 bg-primary/[0.06] px-3 py-2 text-[11px] leading-relaxed text-text-secondary">
              Resume preserves the run&apos;s original architecture, audio
              backend, train/validation split and actor-candidate snapshot.
              Epochs is the new total target, not an additional count.
            </p>
          ) : null}
          <label className="mt-4 flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={trainForm.stop_vllm}
              onChange={(event) =>
                setTrain('stop_vllm', event.target.checked)
              }
              className="h-3.5 w-3.5 accent-primary"
            />
            Stop vLLM first
          </label>
          <Button
            intent="primary"
            onClick={() => void startTrain()}
            disabled={!canTrain}
            className="mt-4"
          >
            {trainingRunning ? 'Training…' : 'Train fusion checkpoint'}
          </Button>
        </Card>

        <Card>
          <SectionLabel>Training dataset</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {[
              [
                'Scope',
                datasetScope === 'joint_only'
                  ? 'Action ∩ Association'
                  : 'Action union',
              ],
              [
                'View',
                trainForm.camera_view === 'all'
                  ? 'all views'
                  : trainForm.camera_view,
              ],
              [
                'Eligible',
                `${eligibleVideos} vid / ${eligibleEvents.toLocaleString()} events`,
              ],
              [
                'Validation',
                isResuming
                  ? 'frozen with run'
                  : validationMode === 'manual'
                    ? `${selectedValidationCount} vid / ${selectedValidationEvents.toLocaleString()} events`
                    : `random ${Math.round(trainForm.val_ratio * 100)}% · seed ${trainForm.split_seed}`,
              ],
              [
                'Action only',
                `${status?.supervision.action_only_videos ?? 0} videos in corpus`,
              ],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-20 flex-shrink-0 text-text-muted">
                  {label}
                </span>
                <span
                  className="min-w-0 flex-1 truncate font-mono tabular-nums text-text-secondary"
                  title={String(value)}
                >
                  {value}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-4 border-t border-border pt-3">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-text-muted">
              Checkpoint heads
            </div>
            <div className="flex flex-wrap gap-1.5">
              <Badge tone="brand">Action</Badge>
              <Badge tone="brand">Location</Badge>
              <Badge tone="brand">Association</Badge>
            </div>
          </div>
          <p className={cn(warningCls, 'mt-3')}>
            Fusion checkpoints currently select the best epoch by Action
            harmonic mAP. Actor Player Top-1 is validated and displayed as its
            own task metric, but does not yet control checkpoint selection.
          </p>
          <p className="mt-3 text-[10px] leading-relaxed text-text-muted">
            Trained checkpoints appear on Action Predict and Association
            Predict, marked as fusion checkpoints.
          </p>
        </Card>
      </div>

      <TrainJobCard
        job={trainJob}
        progressKey="fusion_model_train_progress"
        epochsFallback={trainForm.num_epochs}
        onCancel={() => void cancelTrain()}
      />
      {performanceQuery.data?.entries.length ? (
        <TrainPerfCard
          data={performanceQuery.data}
          onSelectRun={setPerfRun}
        />
      ) : null}
    </div>
  );
}
