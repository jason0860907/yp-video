import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import fusionTrainSchema from '@contracts/fusion_train_request.schema.json';
import { API, apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { useSchemaForm } from '@/lib/schemaForm';
import { useSingleJob } from '@/lib/useSingleJob';
import { fieldCls } from '@/components/form/Field';
import { SchemaForm } from '@/components/form/SchemaForm';
import {
  SchemaCheckboxField,
  SchemaNumberField,
  SchemaSearchSelectField,
  SchemaSelectField,
  SchemaTextField,
} from '@/components/form/SchemaFields';
import { FieldShell } from '@/components/form/FieldLabel';
import { useTrainPerformance } from '@/components/train/useTrainPerformance';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Collapsible } from '@/components/ui/Collapsible';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { TrainJobCard } from '@/components/train/TrainJobCard';
import { TrainPerfCard } from '@/components/train/TrainPerfCard';
import { toast } from '@/components/feedback/toast';
import type { CutKind, FusionModelStatus, FusionRecipeId, Job } from '@/types/api';
import type { FusionTrainRequest } from '@/types/contracts/fusion_train_request.schema';

type FusionForm = Required<FusionTrainRequest>;

/** Mirrors spot_training.recipe_token — the run-name token per task set. */
function recipeToken(tasks: string[]): string {
  if (tasks.includes('action') && tasks.includes('rally')) {
    return tasks.includes('winner') ? 'act_ral_win' : 'act_ral';
  }
  if (tasks.includes('action')) return tasks.includes('actor') ? 'ass_act' : 'act';
  return tasks.includes('winner') ? 'ral_win' : 'ral';
}

export function FusionTrainPage() {
  const form = useSchemaForm<FusionForm>(fusionTrainSchema);
  const { values, set } = form;
  const [validationVideos, setValidationVideos] = useState<Set<string>>(new Set());

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

  const { perf, setPerfRun } = useTrainPerformance(
    'fusion-model-performance',
    API.fusionModel.performance,
    trainingRunning,
  );

  const recipe = status?.recipes.find((item) => item.id === values.recipe);
  const tasks = recipe?.tasks ?? [];
  const visible = new Set(recipe?.fields ?? []);
  const isRally = tasks.includes('rally');
  const isManual = values.validation === 'manual';

  // Switching recipe resets the trainer knobs to that recipe's defaults —
  // rally and action runs want very different batch/epoch/LR values.
  const pickRecipe = (id: FusionRecipeId) => {
    set('recipe', id);
    const next = status?.recipes.find((item) => item.id === id);
    for (const [key, value] of Object.entries(next?.defaults ?? {})) {
      set(key as keyof FusionForm, value as FusionForm[keyof FusionForm]);
    }
    if (!next?.fields.includes('include_predictions')) set('include_predictions', false);
    setValidationVideos(new Set());
  };

  // Mirrors the server's spot_run_name default so the empty field previews
  // the actual run name.
  const now = new Date();
  const autoRunName = [
    `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`,
    values.camera_view === 'all' ? 'all_view' : values.camera_view,
    recipeToken(tasks),
    values.feature_arch.replace(/_(tsm|gsm)$/, ''),
  ].join('_');

  const actionAnnotations = status?.action_annotations;
  const rallyAnnotations = status?.rally_annotations;
  const validationSeeded = useRef(false);
  type ValidationChoice = {
    name: string;
    kind: CutKind;
    events: number;
    is_val?: boolean;
    has_association_label?: boolean;
  };
  const validationChoices: ValidationChoice[] = isRally
    ? (rallyAnnotations?.per_video ?? [])
        .filter((video) => values.camera_view === 'all' || video.view === values.camera_view)
        .map((video) => ({ name: video.video, kind: video.view as CutKind, events: 0 }))
    : (actionAnnotations?.per_video ?? [])
        .filter(
          (video) =>
            (values.camera_view === 'all' || video.view === values.camera_view) &&
            (!tasks.includes('actor') || values.dataset_scope === 'partial_labels' || video.has_association_label),
        )
        .map((video) => ({ ...video, name: video.video, kind: video.view as CutKind }));
  const eligibleVideos = validationChoices.length;
  const eligibleEvents = validationChoices.reduce((total, video) => total + video.events, 0);
  const selectedValidation = validationChoices.filter((video) => validationVideos.has(video.name));
  const canTrain =
    Boolean(status?.spot_available) &&
    Boolean(recipe) &&
    !trainingRunning &&
    (!isManual || selectedValidation.length > 0);

  const usableRally = Math.max(0, Number(rallyAnnotations?.with_video) || 0);
  const rallyTrainingVideos = values.video_limit > 0 ? Math.min(values.video_limit, usableRally) : usableRally;

  useEffect(() => {
    if (validationSeeded.current || !actionAnnotations?.per_video?.length) return;
    setValidationVideos(
      new Set(actionAnnotations.per_video.filter((video) => video.is_val).map((video) => video.video)),
    );
    validationSeeded.current = true;
  }, [actionAnnotations?.per_video]);

  const startTrain = async () => {
    if (isManual && selectedValidation.length === 0) {
      toast.warning('Select at least one validation video');
      return;
    }
    if (visible.has('acc_grad_iter') && values.batch_size % values.acc_grad_iter !== 0) {
      toast.warning('Batch must be divisible by grad accumulation steps');
      return;
    }
    try {
      const body: FusionForm = {
        ...values,
        validation_videos: isManual ? selectedValidation.map((video) => video.name) : [],
      };
      const job = await apiFetch<Job>(API.fusionModel.train, { method: 'POST', body });
      setPerfRun(undefined);
      setTrainJob(job);
      toast.success(`${recipe?.name ?? 'Fusion'} training started`);
    } catch (error) {
      toast.error(`Training failed to start: ${errMsg(error)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <p className="mb-4 text-xs leading-relaxed text-text-muted">
            {recipe?.description ?? 'One SPOT checkpoint per recipe; pick which heads it learns.'}
          </p>
          <SchemaForm form={form}>
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
              {/* Recipe names and per-recipe defaults are served by /status
                  (the contract registry) — a page-composed select. */}
              <FieldShell label="Recipe" className="col-span-3">
                <select
                  value={values.recipe}
                  onChange={(event) => pickRecipe(event.target.value as FusionRecipeId)}
                  className={cn(fieldCls, 'cursor-pointer appearance-none')}
                >
                  {(status?.recipes ?? []).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </FieldShell>
              <SchemaTextField
                name="run_name"
                label="Run name"
                placeholder={autoRunName}
                className="col-span-2"
              />
              <SchemaSelectField
                name="camera_view"
                label="Camera view"
                optionLabels={{ all: 'All Views', broadcast: 'Broadcast', sideline: 'Sideline' }}
              />
              {visible.has('dataset_scope') && (
                <FieldShell label="Dataset scope" className="col-span-3">
                  <select
                    value={values.dataset_scope}
                    onChange={(event) => {
                      const scope = event.target.value as FusionForm['dataset_scope'];
                      set('dataset_scope', scope);
                      // Predictions carry no association labels — the joint-only
                      // scope can never include them.
                      if (scope === 'joint_only') set('include_predictions', false);
                    }}
                    className={cn(fieldCls, 'cursor-pointer appearance-none')}
                  >
                    <option value="joint_only">
                      Joint supervision only — Action ∩ Association ({status?.supervision.joint_videos ?? 0} videos)
                    </option>
                    <option value="partial_labels">
                      Partial-label union — all Action videos ({status?.supervision.action_videos ?? 0} videos)
                    </option>
                  </select>
                  <span className="block text-[10px] leading-relaxed text-text-muted">
                    {values.dataset_scope === 'joint_only'
                      ? 'Every training video must produce actor targets; missing Association supervision fails the run.'
                      : `${status?.supervision.action_only_videos ?? 0} Action-only videos update the shared backbone and Action head, while their actor loss is masked.`}
                  </span>
                </FieldShell>
              )}
              <SchemaSearchSelectField
                name="init_checkpoint"
                label="Init checkpoint"
                options={status?.init_checkpoints[values.recipe] ?? []}
                placeholder="— From scratch —"
                className="col-span-2"
              />
              {visible.has('audio_backend') && (
                <SchemaSelectField
                  name="audio_backend"
                  label="Audio"
                  optionLabels={{ logmel: 'Log-mel fusion', none: 'Visual only' }}
                />
              )}
              <SchemaSelectField name="feature_arch" label="Feature" />
              <SchemaSelectField name="temporal_arch" label="Temporal" />
              {visible.has('video_limit') && <SchemaNumberField name="video_limit" label="Video limit" />}
              <SchemaNumberField name="num_epochs" label="Epochs" />
              <SchemaNumberField name="batch_size" label="Batch" />
              <SchemaNumberField name="learning_rate" label="Learning rate" />
            </div>

            <Collapsible label="Advanced" className="mt-4">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
                <SchemaNumberField name="clip_len" label="Clip len" />
                {visible.has('sample_fps') && <SchemaNumberField name="sample_fps" label="Sample fps" />}
                {visible.has('action_sample_fps') && (
                  <SchemaNumberField name="action_sample_fps" label="Action fps" />
                )}
                {visible.has('rally_sample_fps') && (
                  <SchemaNumberField name="rally_sample_fps" label="Rally fps" />
                )}
                {visible.has('winner_sample_fps') && (
                  <SchemaNumberField name="winner_sample_fps" label="Winner fps" />
                )}
                {visible.has('acc_grad_iter') && <SchemaNumberField name="acc_grad_iter" label="Grad accum" />}
                <SchemaNumberField name="warm_up_epochs" label="Warmup" />
                <SchemaNumberField name="num_workers" label="Workers" />
                <SchemaNumberField name="gpu" label="GPU" />
                <SchemaSelectField name="criterion" />
                <SchemaNumberField name="start_val_epoch" label="Start val" />
                <SchemaNumberField name="epoch_num_frames" label="Epoch frames" />
              </div>
            </Collapsible>

            <div className="mt-5 border-t border-border pt-4">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
                <SchemaSelectField
                  name="validation"
                  label="Validation"
                  optionLabels={{ ratio: 'Seeded ratio', manual: 'Pick videos', none: 'None (final fit)' }}
                />
                {values.validation === 'ratio' && (
                  <>
                    <SchemaNumberField name="val_ratio" label="Val ratio" step={0.01} />
                    <SchemaNumberField name="split_seed" label="Split seed" />
                  </>
                )}
              </div>
              {isManual && (
                <div className="mt-3">
                  <VideoMultiSelectList
                    videos={validationChoices}
                    selected={validationVideos}
                    onSelectedChange={setValidationVideos}
                    title="Validation split"
                    quickSelects={[
                      { label: 'Load saved val set', predicate: (v) => Boolean(v.is_val) },
                      { label: 'Clear', predicate: () => false },
                    ]}
                    renderMeta={(video) =>
                      isRally ? null : (
                        <>
                          {video.has_association_label ? (
                            <Badge tone="success">association</Badge>
                          ) : (
                            <Badge tone="warning">action only</Badge>
                          )}
                          <span className="font-mono text-[10px] tabular-nums text-text-muted">{video.events}</span>
                        </>
                      )
                    }
                    maxHeightClass="max-h-64"
                    emptyTitle="No eligible videos in this scope"
                    query={statusQuery}
                  />
                  <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
                    Selected videos are validation only; every other matching annotation trains.
                  </p>
                </div>
              )}
            </div>

            {isRally && (
              <p className="mt-3 text-xs text-text-secondary">
                Trains on {rallyTrainingVideos} video(s) from the shared frame cache. Video limit 0 = all annotated videos.
              </p>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-3">
              {visible.has('include_predictions') &&
                (!visible.has('dataset_scope') || values.dataset_scope === 'partial_labels') && (
                  <SchemaCheckboxField name="include_predictions" label="Include predictions" />
                )}
              <SchemaCheckboxField name="stop_vllm" label="Stop vLLM first" />
            </div>
          </SchemaForm>
          <div className="mt-4 flex items-center gap-2">
            <Button intent="primary" onClick={() => void startTrain()} disabled={!canTrain} className="flex-1">
              {trainingRunning ? 'Training…' : `Train ${recipe?.name ?? ''}`}
            </Button>
            {trainingRunning && <Button onClick={() => void cancelTrain()}>Cancel</Button>}
          </div>
        </Card>

        <Card>
          <SectionLabel>Training dataset</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {(isRally
              ? [
                  ['Source', 'Rally annotations'],
                  ['View', values.camera_view === 'all' ? 'all views' : values.camera_view],
                  ['Annotated', `${rallyAnnotations?.videos ?? 0} vid / ${(rallyAnnotations?.rallies ?? 0).toLocaleString()} rallies`],
                  ['Cuts', `${usableRally} vid (${rallyAnnotations?.missing_videos ?? 0} missing)`],
                ]
              : [
                  ['Scope', tasks.includes('actor') && values.dataset_scope === 'joint_only' ? 'Action ∩ Association' : 'Action union'],
                  ['View', values.camera_view === 'all' ? 'all views' : values.camera_view],
                  ['Eligible', `${eligibleVideos} vid / ${eligibleEvents.toLocaleString()} events`],
                  ['Validation', isManual ? `${selectedValidation.length} vid` : values.validation],
                  ['Action only', `${status?.supervision.action_only_videos ?? 0} videos in corpus`],
                ]
            ).map(([label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-20 flex-shrink-0 text-text-muted">{label}</span>
                <span className="min-w-0 flex-1 truncate font-mono tabular-nums text-text-secondary" title={String(value)}>
                  {value}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-4 border-t border-border pt-3">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-text-muted">Checkpoint heads</div>
            <div className="flex flex-wrap gap-1.5">
              {tasks.map((task) => (
                <Badge key={task} tone="brand">
                  {status?.task_labels[task] ?? task}
                </Badge>
              ))}
            </div>
          </div>
          <p className="mt-3 text-[10px] leading-relaxed text-text-muted">
            The package keeps a best epoch per task; each predict surface loads the head it needs at that
            head's own best. Serveable on their own:{' '}
            {(recipe?.serveable_tasks ?? []).map((task) => status?.task_labels[task] ?? task).join(', ') || '—'}.
          </p>
        </Card>
      </div>

      <TrainJobCard
        job={trainJob}
        progressKey="spot_train_progress"
        epochsFallback={values.num_epochs}
        onCancel={() => void cancelTrain()}
        mapLabel={isRally ? 'Seg mAP' : 'Last mAP'}
        eventNoun={isRally ? 'rallies' : 'events'}
      />
      {perf && perf.entries.length > 0 ? <TrainPerfCard data={perf} onSelectRun={setPerfRun} /> : null}
    </div>
  );
}
