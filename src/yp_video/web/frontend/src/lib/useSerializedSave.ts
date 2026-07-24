import { useCallback, useRef, type MutableRefObject } from 'react';

interface SerializedSaveContext {
  /** Edit revision captured immediately before this save started. */
  revision: number;
  /** Background/queued saves should not emit success UI. */
  silent: boolean;
}

interface SerializedSaveOptions {
  /** Monotonically incremented by every mutation of the persisted value. */
  revision: MutableRefObject<number>;
  /** Always called through the latest render closure. */
  save: (context: SerializedSaveContext) => Promise<void>;
  onError?: (error: unknown, context: SerializedSaveContext) => void;
}

/**
 * Serialize writes for an autosaved editor.
 *
 * A call made while a request is in flight is coalesced into one follow-up
 * save. If the edit revision changes during the request, a follow-up is also
 * queued automatically. The caller decides whether a completed revision is
 * still current before clearing dirty state or a local draft.
 */
export function useSerializedSave({ revision, save, onError }: SerializedSaveOptions) {
  const savingRef = useRef(false);
  const queuedRef = useRef(false);
  const saveImplRef = useRef(save);
  const onErrorRef = useRef(onError);
  const runRef = useRef<(silent?: boolean) => Promise<boolean>>(async () => false);
  saveImplRef.current = save;
  onErrorRef.current = onError;

  const run = useCallback(
    async (silent = false): Promise<boolean> => {
      if (savingRef.current) {
        queuedRef.current = true;
        return false;
      }

      savingRef.current = true;
      const context = { revision: revision.current, silent };
      try {
        await saveImplRef.current(context);
        return true;
      } catch (error) {
        onErrorRef.current?.(error, context);
        return false;
      } finally {
        savingRef.current = false;
        if (queuedRef.current || revision.current !== context.revision) {
          queuedRef.current = false;
          void runRef.current(true);
        }
      }
    },
    [revision],
  );
  runRef.current = run;
  return run;
}
