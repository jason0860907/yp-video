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
import { CheckpointsCard } from '@/components/train/CheckpointsCard';
import { Collapsible } from '@/components/ui/Collapsible';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { TrainJobCard } from '@/components/train/TrainJobCard';
import { TrainPerfCard } from '@/components/train/TrainPerfCard';
import { toast } from '@/components/feedback/toast';
import type { CutKind, FusionModelStatus, FusionRecipeId, Job } from '@/types/api';
import type { FusionTrainRequest } from '@/types/contracts/fusion_train_request.schema';

type FusionForm = Required<FusionTrainRequest>;

const warningCls =
  'rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-[11px] leading-relaxed text-amber-400';

export function FusionTrainPage() {
  const form = useSchemaForm<FusionForm>(fusionTrainSchema, { validation_mode: 'manual' });
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

  const chosenRecipe = status?.recipes.find((item) => item.id === values.recipe);
  const recipeBlocked = Boolean(chosenRecipe && !chosenRecipe.available);
  const annotations = status?.action_annotations;
  const validationSeeded = useRef(false);
  const validationChoices = (annotations?.per_video ?? [])
    .filter(
      (video) =>
        (values.camera_view === 'all' || video.view === values.camera_view) &&
        (values.dataset_scope === 'partial_labels' || video.has_association_label),
    )
    .map((video) => ({ ...video, name: video.video, kind: video.view as CutKind }));
  const eligibleVideos = validationChoices.length;
  const eligibleEvents = validationChoices.reduce(
    (total, video) => total + video.events,
    0,
  );
  const selectedValidation = validationChoices.filter((video) =>
    validationVideos.has(video.name),
  );
  const selectedValidationEvents = selectedValidation.reduce(
    (total, video) => total + video.events,
    0,
  );
  const canTrain =
    Boolean(status?.spot_available) &&
    !recipeBlocked &&
    !trainingRunning &&
    selectedValidation.length > 0;

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

  const startTrain = async () => {
    if (selectedValidation.length === 0) {
      toast.warning(
        'Select at least one validation video for this camera view',
      );
      return;
    }
    if (values.batch_size % values.acc_grad_iter !== 0) {
      toast.warning('Batch must be divisible by grad accumulation steps');
      return;
    }
    try {
      const body: FusionForm = {
        ...values,
        validation_videos: selectedValidation.map((video) => `${video.name}_actions.jsonl`),
      };
      const job = await apiFetch<Job>(API.fusionModel.train, { method: 'POST', body });
      setPerfRun(undefined);
      setTrainJob(job);
      toast.success('Fusion training started');
    } catch (error) {
      toast.error(`Fusion training failed to start: ${errMsg(error)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader />

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
          <SchemaForm form={form}>
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
              {/* Recipe availability (name, blocked_on) is runtime status the
                  contract's id enum can't carry — a page-composed select. */}
              <FieldShell label="Recipe" className="col-span-3">
                <select
                  value={values.recipe}
                  onChange={(event) => set('recipe', event.target.value as FusionRecipeId)}
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
              </FieldShell>
              <SchemaTextField
                name="run_name"
                label="Run name"
                placeholder="yp_fusion_association_action_..."
                className="col-span-2"
              />
              <SchemaSelectField
                name="camera_view"
                label="Camera view"
                optionLabels={{ all: 'All Views', broadcast: 'Broadcast', sideline: 'Sideline' }}
              />
              <FieldShell label="Dataset scope" className="col-span-3">
                <select
                  value={values.dataset_scope}
                  onChange={(event) =>
                    set('dataset_scope', event.target.value as FusionForm['dataset_scope'])
                  }
                  className={cn(fieldCls, 'cursor-pointer appearance-none')}
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
                  {values.dataset_scope === 'joint_only'
                    ? 'Every training video must produce actor targets; missing Association supervision fails the run.'
                    : `${status?.supervision.action_only_videos ?? 0} Action-only videos update the shared backbone and Action head, while their actor loss is masked.`}
                </span>
              </FieldShell>
              <SchemaSearchSelectField
                name="init_checkpoint"
                label="Init checkpoint"
                options={status?.init_checkpoints ?? []}
                placeholder="— From scratch —"
                className="col-span-2"
              />
              <SchemaSelectField
                name="audio_backend"
                label="Audio"
                optionLabels={{ logmel: 'Log-mel fusion', none: 'Visual only' }}
              />
              <SchemaSelectField name="feature_arch" label="Feature" />
              <SchemaSelectField name="temporal_arch" label="Temporal" />
              <SchemaNumberField name="num_epochs" label="Epochs" />
              <SchemaNumberField name="batch_size" label="Batch" />
              <SchemaNumberField name="learning_rate" label="Learning rate" />
            </div>

            <Collapsible label="Advanced" className="mt-4">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
                <SchemaNumberField name="clip_len" label="Clip len" />
                <SchemaNumberField name="sample_fps" label="Sample fps" />
                <SchemaNumberField name="acc_grad_iter" label="Grad accum" />
                <SchemaNumberField name="warm_up_epochs" label="Warmup" />
                <SchemaNumberField name="num_workers" label="Workers" />
                <SchemaNumberField name="gpu" label="GPU" />
                <SchemaSelectField name="criterion" />
                <SchemaNumberField name="start_val_epoch" label="Start val" />
                <SchemaNumberField name="epoch_num_frames" label="Epoch frames" />
              </div>
            </Collapsible>

            <div className="mt-5 border-t border-border pt-4">
              <VideoMultiSelectList
                videos={validationChoices}
                selected={validationVideos}
                onSelectedChange={setValidationVideos}
                title="Validation split"
                quickSelects={[
                  { label: 'Load saved val set', predicate: (v) => Boolean(v.is_val) },
                  { label: 'Clear', predicate: () => false },
                ]}
                renderMeta={(video) => (
                  <>
                    {video.has_association_label ? (
                      <Badge tone="success">association</Badge>
                    ) : (
                      <Badge tone="warning">action only</Badge>
                    )}
                    <span className="font-mono text-[10px] tabular-nums text-text-muted">
                      {video.events}
                    </span>
                  </>
                )}
                maxHeightClass="max-h-64"
                emptyTitle="No eligible videos in this scope"
                query={statusQuery}
              />
              <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
                Selected videos are validation only; every other matching
                action annotation trains.
              </p>
            </div>

            <div className="mt-4">
              <SchemaCheckboxField name="stop_vllm" label="Stop vLLM first" />
            </div>
          </SchemaForm>
          <div className="mt-4 flex items-center gap-2">
            <Button
              intent="primary"
              onClick={() => void startTrain()}
              disabled={!canTrain}
              className="flex-1"
            >
              {trainingRunning ? 'Training…' : 'Train fusion checkpoint'}
            </Button>
            {trainingRunning && (
              <Button onClick={() => void cancelTrain()}>Cancel</Button>
            )}
          </div>
        </Card>

        <Card>
          <SectionLabel>Training dataset</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {[
              [
                'Scope',
                values.dataset_scope === 'joint_only'
                  ? 'Action ∩ Association'
                  : 'Action union',
              ],
              [
                'View',
                values.camera_view === 'all' ? 'all views' : values.camera_view,
              ],
              [
                'Eligible',
                `${eligibleVideos} vid / ${eligibleEvents.toLocaleString()} events`,
              ],
              [
                'Validation',
                `${selectedValidation.length} vid / ${selectedValidationEvents.toLocaleString()} events`,
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
          <p className="mt-3 text-[10px] leading-relaxed text-text-muted">
            The package keeps a best epoch per task: Action Predict loads the
            Action-mAP best, Association Predict loads the actor head at its
            own Player Top-1 best. Trained checkpoints appear on both pages,
            marked as fusion checkpoints.
          </p>
        </Card>
      </div>

      <TrainJobCard
        job={trainJob}
        progressKey="fusion_model_train_progress"
        epochsFallback={values.num_epochs}
        onCancel={() => void cancelTrain()}
      />
      {perf && perf.entries.length > 0 ? (
        <TrainPerfCard data={perf} onSelectRun={setPerfRun} />
      ) : null}
      <CheckpointsCard
        title="Fusion checkpoints"
        checkpoints={status?.checkpoints ?? []}
      />
    </div>
  );
}
