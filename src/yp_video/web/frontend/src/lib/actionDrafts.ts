import type { ActionEditor } from '@/lib/actionEditorModel';

export const ACTION_AUTOSAVE_MS = 2000;
export const ACTION_DRAFT_DEBOUNCE_MS = 300;

const ACTION_DRAFT_PREFIX = 'vq:action-draft';
const actionDraftKey = (video: string) => `${ACTION_DRAFT_PREFIX}:${video}`;

export const readActionDraft = (video: string): ActionEditor | null => {
  try {
    const raw = localStorage.getItem(actionDraftKey(video));
    if (!raw) return null;
    const draft = JSON.parse(raw) as ActionEditor;
    return Array.isArray(draft.events) ? draft : null;
  } catch {
    return null;
  }
};

export const writeActionDraft = (editor: ActionEditor): void => {
  try {
    localStorage.setItem(actionDraftKey(editor.video), JSON.stringify(editor));
  } catch {
    // Draft persistence is best-effort in privacy mode or when quota is full.
  }
};

export const clearActionDraft = (video: string): void => {
  try {
    localStorage.removeItem(actionDraftKey(video));
  } catch {
    // Draft cleanup is best-effort for browsers that block localStorage.
  }
};
