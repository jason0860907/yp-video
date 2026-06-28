import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { formatBytes } from '@/lib/format';
import { isTerminal } from '@/lib/job';
import { useSSE } from '@/lib/useSSE';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { JobProgress } from '@/components/job/JobProgress';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { Job } from '@/types/api';

interface Category {
  key: string;
  label: string;
  localOnly?: boolean;
}
const CATEGORIES: Category[] = [
  { key: 'videos', label: 'Videos', localOnly: true },
  { key: 'cuts-broadcast', label: 'Cuts (Broadcast)' },
  { key: 'cuts-sideline', label: 'Cuts (Sideline)' },
  { key: 'rally-pre-annotations', label: 'Rally Predictions' },
  { key: 'tad-predictions', label: 'TAD-Predictions' },
  { key: 'tad-features', label: 'TAD-Features' },
  { key: 'tad-checkpoints', label: 'TAD-Checkpoints' },
  { key: 'rally-annotations', label: 'Annotations' },
  { key: 'action-pre-annotations', label: 'Action Pre-Annotations' },
  { key: 'action-annotations', label: 'Action Annotations' },
  { key: 'action-checkpoints', label: 'Action Checkpoints' },
  { key: 'rally_clips', label: 'Rally Clips' },
];

interface FileRow {
  name: string;
  path: string;
  size: number;
  group?: string;
  uploaded?: boolean;
  local?: boolean;
  selected: boolean;
}
type Mode = 'upload' | 'download';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

const subPathOf = (f: FileRow) => (f.group && f.path.startsWith(`${f.group}/`) ? f.path.slice(f.group.length + 1) : f.path || f.name);
const subDirOf = (f: FileRow) => {
  const sp = subPathOf(f);
  const i = sp.lastIndexOf('/');
  return i === -1 ? '' : sp.slice(0, i);
};
const folderKeyOf = (f: FileRow) => `${f.group || ''}/${subDirOf(f)}`;

export function UploadPage() {
  const [mode, setMode] = useState<Mode>('upload');
  const [category, setCategory] = useState('cuts-broadcast');
  const [files, setFiles] = useState<FileRow[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [filesError, setFilesError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);

  const statusQuery = useQuery({
    queryKey: ['upload-status'],
    queryFn: () => apiFetch<{ configured: boolean; bucket?: string }>(API.upload.status),
  });
  const configured = statusQuery.data?.configured ?? false;

  const cat = CATEGORIES.find((c) => c.key === category)!;
  const localOnly = !!cat.localOnly;
  const effectiveMode: Mode = localOnly ? 'upload' : mode;
  const isUpload = effectiveMode === 'upload';

  const loadFiles = async () => {
    if (!configured) return;
    setLoadingFiles(true);
    setFilesError(null);
    try {
      const path = isUpload ? API.upload.files(category) : API.upload.r2Files(category);
      const list = await apiFetch<Omit<FileRow, 'selected'>[]>(path);
      setFiles(list.map((f) => ({ ...f, selected: false })));
      setExpanded({});
    } catch (e) {
      setFilesError(errMsg(e));
      setFiles([]);
    } finally {
      setLoadingFiles(false);
    }
  };

  // (Re)load whenever the configured state, category, or mode changes.
  useEffect(() => {
    void loadFiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configured, category, effectiveMode]);

  useSSE<Job>(job && !isTerminal(job.status) ? API.jobs.eventsSSE(job.id) : null, (data) => {
    setJob(data);
    if (isTerminal(data.status)) {
      setBusy(false);
      if (data.status === 'failed') toast.error(`Transfer failed: ${data.error || 'Unknown error'}`);
      else toast.success(data.message || 'Transfer complete!');
      void loadFiles();
    }
  });

  const setSelection = (pred: (f: FileRow) => boolean) => setFiles((prev) => prev.map((f) => ({ ...f, selected: pred(f) })));
  const toggleFolder = (key: string, on: boolean) => setFiles((prev) => prev.map((f) => (folderKeyOf(f) === key ? { ...f, selected: on } : f)));

  const selected = files.filter((f) => f.selected);
  const selectedSize = selected.reduce((s, f) => s + f.size, 0);
  const syncedCount = files.filter((f) => (isUpload ? f.uploaded : f.local)).length;

  const startTransfer = async (kind: 'upload' | 'download') => {
    const paths = selected.map((f) => f.path);
    if (!paths.length) return toast.warning('No files selected');
    setBusy(true);
    try {
      const endpoint = kind === 'upload' ? API.upload.start : API.upload.download;
      const started = await apiFetch<Job>(endpoint, { method: 'POST', body: { category, files: paths } });
      setJob(started);
    } catch (e) {
      setBusy(false);
      toast.error(`Failed to start ${kind}: ${errMsg(e)}`);
    }
  };

  const deleteR2 = async () => {
    if (!selected.length) return toast.warning('No files selected');
    const ok = await confirm({
      title: `Delete ${selected.length} files from R2?`,
      body: 'This only removes them from cloud storage — local copies are not touched.',
      confirmText: 'Delete from R2',
      variant: 'danger',
    });
    if (!ok) return;
    try {
      const res = await apiFetch<{ deleted: number }>(API.upload.deleteR2, { method: 'POST', body: { category, files: selected.map((f) => f.path) } });
      toast.success(`Deleted ${res.deleted} files from R2`);
      void loadFiles();
    } catch (e) {
      toast.error(`Delete failed: ${errMsg(e)}`);
    }
  };

  const deleteLocal = async () => {
    if (!selected.length) return toast.warning('No files selected');
    const notUploaded = localOnly ? [] : selected.filter((f) => !f.uploaded);
    let body: string;
    let variant: 'warning' | 'danger' = 'warning';
    if (localOnly) {
      body = 'These files only exist locally and are not backed up to R2.';
      variant = 'danger';
    } else if (notUploaded.length === 0) {
      body = 'They are already on R2, so this is recoverable.';
    } else {
      body = `⚠️ ${notUploaded.length} of them are NOT on R2 and will be lost permanently.`;
      variant = 'danger';
    }
    const ok = await confirm({ title: `Delete ${selected.length} local files?`, body, confirmText: 'Delete locally', variant });
    if (!ok) return;
    const force = localOnly || notUploaded.length > 0;
    try {
      const res = await apiFetch<{ deleted?: string[]; skipped?: string[] }>(API.upload.deleteLocal, {
        method: 'POST',
        body: { category, files: selected.map((f) => f.path), force },
      });
      const count = res.deleted?.length || 0;
      const skipped = res.skipped?.length || 0;
      if (count > 0) toast.success(`Deleted ${count} local files${skipped ? `, ${skipped} skipped` : ''}`);
      else toast.info('No files deleted');
      void loadFiles();
    } catch (e) {
      toast.error(`Delete failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          configured ? (
            <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2.5 py-1 text-[11px] font-medium text-emerald-400 ring-1 ring-emerald-500/20">
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {statusQuery.data?.bucket || 'configured'}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 rounded-full bg-amber-500/15 px-2.5 py-1 text-[11px] font-medium text-amber-400 ring-1 ring-amber-500/20">
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              not configured
            </span>
          )
        }
      />

      {!configured && (
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.06] px-4 py-3 text-sm text-amber-300">
          R2 not configured — fill in <code className="rounded bg-surface-200 px-1.5 py-0.5 text-text-secondary">r2.env</code> with your Cloudflare R2 credentials.
        </div>
      )}

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Files" value={files.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.length} tintClass="text-accent" />
        <StatTile label="Size" value={formatBytes(selectedSize)} tintClass="text-text-primary" />
        <StatTile label={isUpload ? 'Uploaded' : 'Local'} value={syncedCount} tintClass="text-emerald-400" />
      </div>

      {/* Category + mode controls */}
      <Card>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          {CATEGORIES.map((c) => (
            <button
              key={c.key}
              type="button"
              onClick={() => setCategory(c.key)}
              className={cn(
                'rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
                c.key === category ? 'border border-primary/25 bg-primary/15 text-primary-light' : 'border border-transparent bg-surface-50 text-text-muted hover:text-text-secondary',
              )}
            >
              {c.label}
            </button>
          ))}
        </div>
        {!localOnly && (
          <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
            {(['upload', 'download'] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn('rounded-md px-3.5 py-1.5 text-xs font-medium transition-colors', mode === m ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]')}
              >
                {m === 'upload' ? 'Upload' : 'Download from R2'}
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* File browser */}
      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <SectionLabel className="mb-0">
            {cat.label} · {files.length} files
          </SectionLabel>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={() => setSelection(() => true)}>
              Select All
            </Button>
            {!localOnly && isUpload && (
              <Button size="sm" intent="default" onClick={() => setSelection((f) => !!f.uploaded)}>
                Uploaded
              </Button>
            )}
            {!localOnly && (
              <Button size="sm" intent="primary" onClick={() => setSelection((f) => (isUpload ? !f.uploaded : !f.local))}>
                Un-synced
              </Button>
            )}
          </div>
        </div>

        <div className="max-h-[32rem] space-y-0.5 overflow-auto pr-1">
          {loadingFiles ? (
            <div className="py-8 text-center text-xs text-text-muted">Loading…</div>
          ) : filesError ? (
            <EmptyState
              icon={
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              }
              title="Failed to load files"
              subtitle={filesError}
            />
          ) : files.length === 0 ? (
            <EmptyState
              icon={
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v8.25m0 0l-3-3m3 3l3-3M2.25 13.5h3.86a2.25 2.25 0 012.012 1.244l.256.512a2.25 2.25 0 002.013 1.244h3.218a2.25 2.25 0 002.013-1.244l.256-.512a2.25 2.25 0 012.013-1.244h3.859" />
                </svg>
              }
              title={isUpload ? 'No local files' : 'No files on R2'}
              subtitle={`No files found in ${category}`}
            />
          ) : (
            <FileTree
              files={files}
              expanded={expanded}
              localOnly={localOnly}
              isUpload={isUpload}
              onToggleFile={(idx, on) => setFiles((prev) => prev.map((f, i) => (i === idx ? { ...f, selected: on } : f)))}
              onToggleFolder={toggleFolder}
              onExpand={(key) => setExpanded((e) => ({ ...e, [key]: !e[key] }))}
            />
          )}
        </div>

        <div className="mt-3 flex items-center gap-3 border-t border-border pt-3">
          {localOnly ? (
            <Button intent="danger" onClick={deleteLocal} disabled={busy}>
              Delete Local
            </Button>
          ) : isUpload ? (
            <>
              <Button intent="primary" onClick={() => startTransfer('upload')} disabled={busy}>
                Upload Selected
              </Button>
              <Button intent="danger" onClick={deleteLocal} disabled={busy}>
                Delete Local
              </Button>
            </>
          ) : (
            <>
              <Button onClick={() => startTransfer('download')} disabled={busy}>
                Download Selected
              </Button>
              <Button intent="danger" onClick={deleteR2} disabled={busy}>
                Delete on R2
              </Button>
            </>
          )}
          <span className="ml-auto font-mono text-xs tabular-nums text-text-muted">
            {selected.length > 0 ? `${selected.length} selected (${formatBytes(selectedSize)})` : ''}
          </span>
        </div>
      </Card>

      {job && (
        <Card>
          <SectionLabel>Progress</SectionLabel>
          <JobProgress job={job} detail={transferDetail(job)} showLogs truncateMsg={false} />
        </Card>
      )}
    </div>
  );
}

function transferDetail(job: Job): string {
  const p = job.params ?? {};
  const bytesDone = Number(p.bytes_done);
  const bytesTotal = Number(p.bytes_total);
  const speed = Number(p.speed);
  const eta = Number(p.eta);
  const running = job.status === 'running';
  return [
    bytesDone && bytesTotal ? `${formatBytes(bytesDone)}/${formatBytes(bytesTotal)}` : '',
    running && speed ? `${formatBytes(speed)}/s` : '',
    running && eta ? `ETA ${eta}s` : '',
  ]
    .filter(Boolean)
    .join(' · ');
}

interface FileTreeProps {
  files: FileRow[];
  expanded: Record<string, boolean>;
  localOnly: boolean;
  isUpload: boolean;
  onToggleFile: (idx: number, on: boolean) => void;
  onToggleFolder: (key: string, on: boolean) => void;
  onExpand: (key: string) => void;
}

/** Grouped, collapsible file list. Files sort by (group → sub-dir → name);
 *  nested files collapse under a folder row with an aggregate checkbox. */
function FileTree({ files, expanded, localOnly, isUpload, onToggleFile, onToggleFolder, onExpand }: FileTreeProps) {
  const { order, groupCounts, subTotal, subSelected } = useMemo(() => {
    const order = files.map((_, i) => i).sort((a, b) => {
      const fa = files[a]!;
      const fb = files[b]!;
      return (fa.group || '').localeCompare(fb.group || '') || subDirOf(fa).localeCompare(subDirOf(fb)) || fa.name.localeCompare(fb.name);
    });
    const groupCounts: Record<string, number> = {};
    const subTotal: Record<string, number> = {};
    const subSelected: Record<string, number> = {};
    for (const f of files) {
      if (f.group) groupCounts[f.group] = (groupCounts[f.group] || 0) + 1;
      if (f.group && subDirOf(f)) {
        const k = folderKeyOf(f);
        subTotal[k] = (subTotal[k] || 0) + 1;
        if (f.selected) subSelected[k] = (subSelected[k] || 0) + 1;
      }
    }
    return { order, groupCounts, subTotal, subSelected };
  }, [files]);

  const rows: ReactNode[] = [];
  let currentGroup: string | null = null;
  let currentSub: string | null = null;

  const syncLabel = isUpload ? 'uploaded' : 'local';
  const unsyncLabel = isUpload ? 'local only' : 'R2 only';

  for (const i of order) {
    const f = files[i]!;
    if (f.group && f.group !== currentGroup) {
      currentGroup = f.group;
      currentSub = null;
      rows.push(
        <div key={`g-${f.group}`} className="flex items-center gap-2 px-2.5 pb-1 pt-3 font-heading text-[11px] font-medium uppercase tracking-wider text-text-muted">
          <span>{f.group}</span>
          <span className="tabular-nums normal-case tracking-normal text-text-muted/70">({groupCounts[f.group]} files)</span>
        </div>,
      );
    }
    const d = f.group ? subDirOf(f) : '';
    if (d !== (currentSub || '')) {
      currentSub = d;
      if (d) {
        const key = folderKeyOf(f);
        const total = subTotal[key] || 0;
        const sel = subSelected[key] || 0;
        rows.push(
          <div key={`f-${key}`} className="flex items-center gap-3 rounded-lg p-2.5 transition-colors hover:bg-surface-50">
            <FolderCheckbox checked={sel > 0 && sel === total} indeterminate={sel > 0 && sel < total} onChange={(on) => onToggleFolder(key, on)} />
            <button type="button" onClick={() => onExpand(key)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
              <svg className={cn('h-3.5 w-3.5 shrink-0 text-text-muted transition-transform', expanded[key] && 'rotate-90')} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
              </svg>
              <span className="truncate text-sm text-text-secondary">{d}/</span>
              <span className="shrink-0 tabular-nums text-[11px] text-text-muted/70">({total})</span>
            </button>
          </div>,
        );
      }
    }
    if (d && !expanded[folderKeyOf(f)]) continue;

    const isSynced = isUpload ? f.uploaded : f.local;
    rows.push(
      <div key={`r-${i}`} className={cn('group flex w-max min-w-full items-center gap-3 rounded-lg p-2.5 transition-colors hover:bg-surface-50', d && 'pl-9')}>
        <input type="checkbox" checked={f.selected} onChange={(e) => onToggleFile(i, e.target.checked)} className="h-3.5 w-3.5 flex-shrink-0 cursor-pointer accent-primary" />
        <span className="flex-1 whitespace-nowrap text-sm text-text-primary">{f.name}</span>
        <span className="tabular-nums text-[11px] text-text-muted">{formatBytes(f.size)}</span>
        {!localOnly &&
          (isSynced ? (
            <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-400 ring-1 ring-emerald-500/20">
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {syncLabel}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 rounded-full bg-ink/5 px-2.5 py-0.5 text-[11px] font-medium text-text-muted ring-1 ring-ink/10">
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {unsyncLabel}
            </span>
          ))}
      </div>,
    );
  }

  return <>{rows}</>;
}

function FolderCheckbox({ checked, indeterminate, onChange }: { checked: boolean; indeterminate: boolean; onChange: (on: boolean) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = indeterminate;
  }, [indeterminate]);
  return <input ref={ref} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="h-3.5 w-3.5 flex-shrink-0 cursor-pointer accent-primary" />;
}
