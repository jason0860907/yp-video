import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
import type { Tracker } from '@/types/api';

export const TRACKER_LABEL: Record<Tracker, string> = {
  bytetrack: 'ByteTrack',
  mcbyte: 'McByte++',
};

/** Which association step links detections into tracklets. McByte++ runs in
 *  the yp-track package, so it is offered only where the server has it. */
export function TrackerSelect({ value, onChange }: { value: Tracker; onChange: (t: Tracker) => void }) {
  const { data } = useQuery({
    queryKey: ['trackers'],
    queryFn: () => apiFetch<Record<Tracker, boolean>>(API.tracklets.trackers),
  });
  return (
    <div>
      <label className="mb-1 block text-[10px] uppercase tracking-wide text-text-muted">Tracker</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as Tracker)}
        className={cn(fieldCls, 'cursor-pointer appearance-none')}
      >
        {(Object.keys(TRACKER_LABEL) as Tracker[]).map((t) => (
          <option key={t} value={t} disabled={data ? !data[t] : t !== 'bytetrack'}>
            {TRACKER_LABEL[t]}
            {data && !data[t] ? ' (yp-track not installed)' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}
