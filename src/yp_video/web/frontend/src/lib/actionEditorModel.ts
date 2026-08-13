import type { ActionAnnotationData, ActionEvent, ActionRally, ActionVideo } from '@/types/api';

export const DEFAULT_ACTION_LABELS = ['serve', 'receive', 'set', 'spike', 'block', 'score'];
export const OUTSIDE_RALLY_KEY = '__outside__';

export interface ActionEditor {
  video: string;
  duration: number;
  fps: number;
  numFrames: number;
  rallies: ActionRally[];
  events: ActionEvent[];
  dirty: boolean;
}

export const EMPTY_ACTION_EDITOR: ActionEditor = {
  video: '',
  duration: 0,
  fps: 30,
  numFrames: 0,
  rallies: [],
  events: [],
  dirty: false,
};

export const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, value));

export const round4 = (value: number) => Math.round(value * 1e4) / 1e4;

export const formatActionTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '00:00';
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, '0')}:${String(Math.floor(seconds - minutes * 60)).padStart(2, '0')}`;
};

export const makeActionId = () =>
  `act_${(crypto.randomUUID?.() ?? `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`)
    .replace(/-/g, '')
    .slice(0, 16)}`;

const normalizeRallyId = (value: unknown): number | null => {
  const id = Number(value);
  return Number.isInteger(id) && id > 0 ? id : null;
};

export const findActionRally = (frame: number, editor: ActionEditor): ActionRally | null => {
  const time = frame / (editor.fps || 30);
  return editor.rallies.find((rally) => time >= rally.start && time < rally.end) ?? null;
};

export const withActionRally = (event: ActionEvent, editor: ActionEditor): ActionEvent => {
  const frame = Math.max(0, Math.round(event.frame || 0));
  const time = frame / (editor.fps || 30);
  const rally = findActionRally(frame, editor);
  return {
    ...event,
    frame,
    time: round4(time),
    rally_id: rally?.rally_id ?? null,
    relative_frame: rally
      ? Math.max(0, Math.round((time - rally.start) * (editor.fps || 30)))
      : null,
  };
};

export const sortActionEvents = (events: ActionEvent[]) =>
  [...events].sort(
    (left, right) =>
      left.frame - right.frame ||
      left.label.localeCompare(right.label) ||
      left.id.localeCompare(right.id),
  );

export function normalizeActionEditor(
  data: ActionAnnotationData,
  labels: string[],
): ActionEditor {
  const fps = Number(data.fps) || 30;
  const rallies: ActionRally[] = (data.rallies ?? [])
    // The server contract (core/rallies.load_rallies) always supplies ids; a
    // row without one is unjoinable, and inventing a positional id here could
    // collide with a real one.
    .flatMap((rally) => {
      const id = normalizeRallyId(rally.rally_id);
      if (id === null) return [];
      return [{
        rally_id: id,
        start: Number(rally.start) || 0,
        end: Number(rally.end) || 0,
        label: rally.label || 'rally',
      }];
    })
    .sort(
      (left, right) =>
        left.start - right.start ||
        left.end - right.end ||
        left.rally_id - right.rally_id,
    );
  const editor: ActionEditor = {
    video: data.source_video || data.video || '',
    duration: Number(data.duration) || 0,
    fps,
    numFrames: Number(data.num_frames) || 0,
    rallies,
    events: [],
    dirty: false,
  };
  editor.events = sortActionEvents(
    (data.events ?? []).map((event) => {
      const raw = event as Record<string, unknown>;
      const xy =
        (raw.xy as number[] | undefined) ??
        [Number(raw.x ?? 0.5), Number(raw.y ?? 0.5)];
      return withActionRally(
        {
          id: (raw.id as string) || makeActionId(),
          // rally_id / time / relative_frame are derived state —
          // withActionRally recomputes all three from frame + rallies.
          rally_id: null,
          frame: Math.max(0, Math.round(Number(raw.frame) || 0)),
          time: null,
          relative_frame: null,
          label: labels.includes(raw.label as string)
            ? (raw.label as string)
            : labels[0]!,
          xy: [
            clamp(Number(xy[0] ?? 0.5), 0, 1),
            clamp(Number(xy[1] ?? 0.5), 0, 1),
          ],
          visible: raw.visible !== false,
        },
        editor,
      );
    }),
  );
  return editor;
}

export const hasActiveActionAnnotation = (video: ActionVideo) =>
  Boolean(video.has_action_annotation);
