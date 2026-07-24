import { useCallback, useRef, useState } from 'react';
import { API, ApiError, apiFetch } from '@/lib/api';
import { clamp } from '@/lib/actionEditorModel';
import type { WaveformData } from '@/types/api';

const POINTS_PER_SECOND = 120;
const MIN_POINTS = 2400;
const MAX_POINTS = 96000;

const EMPTY_WAVEFORM: WaveformData = {
  video: '',
  loading: false,
  error: '',
  hasAudio: false,
  duration: 0,
  peaks: [],
  rms: [],
};

const errorMessage = (error: unknown) =>
  error instanceof ApiError
    ? error.body
    : error instanceof Error
      ? error.message
      : String(error);

const pointCount = (durationSeconds: number) =>
  clamp(
    Math.ceil(Math.max(0, durationSeconds) * POINTS_PER_SECOND) || MIN_POINTS,
    MIN_POINTS,
    MAX_POINTS,
  );

export function useActionWaveform() {
  const [waveform, setWaveform] = useState<WaveformData>(EMPTY_WAVEFORM);
  const requestRevision = useRef(0);

  const loadWaveform = useCallback(async (video: string, duration: number) => {
    const revision = ++requestRevision.current;
    setWaveform({ ...EMPTY_WAVEFORM, video, loading: true });
    try {
      const points = pointCount(duration);
      const data = await apiFetch<{
        has_audio?: boolean;
        duration?: number;
        peaks?: number[];
        rms?: number[];
      }>(`${API.actionAnnotate.waveform(video)}?points=${points}`);
      if (revision !== requestRevision.current) return;
      const peaks = Array.isArray(data.peaks)
        ? data.peaks.map((value) => clamp(Number(value) || 0, 0, 1))
        : [];
      const rms =
        Array.isArray(data.rms) && data.rms.length === peaks.length
          ? data.rms.map((value) => clamp(Number(value) || 0, 0, 1))
          : peaks;
      setWaveform({
        video,
        loading: false,
        error: '',
        hasAudio: Boolean(data.has_audio),
        duration: Number(data.duration) || duration || 0,
        peaks,
        rms,
      });
    } catch (error) {
      if (revision !== requestRevision.current) return;
      setWaveform({ ...EMPTY_WAVEFORM, video, error: errorMessage(error) });
    }
  }, []);

  return { waveform, loadWaveform };
}
