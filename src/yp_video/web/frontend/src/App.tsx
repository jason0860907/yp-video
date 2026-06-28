import type { ReactElement } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { DEFAULT_PATH, NAV_ITEMS } from '@/components/layout/nav';
import { Placeholder } from '@/components/Placeholder';
import { JobsPage } from '@/pages/JobsPage';
import { DownloadPage } from '@/pages/DownloadPage';
import { DetectPage } from '@/pages/DetectPage';
import { ActionPredictPage } from '@/pages/ActionPredictPage';
import { PredictPage } from '@/pages/PredictPage';
import { ActionTrainPage } from '@/pages/ActionTrainPage';
import { UploadPage } from '@/pages/UploadPage';
import { CutPage } from '@/pages/CutPage';
import { AnnotatePage } from '@/pages/AnnotatePage';
import { ReviewPage } from '@/pages/ReviewPage';

/** Migrated pages, by route. Paths absent here fall back to a Placeholder. */
const PAGES: Record<string, ReactElement> = {
  '/download': <DownloadPage />,
  '/cut': <CutPage />,
  '/detect': <DetectPage />,
  '/annotate': <AnnotatePage />,
  '/predict': <PredictPage />,
  '/review': <ReviewPage />,
  '/action-predict': <ActionPredictPage />,
  '/action-train': <ActionTrainPage />,
  '/upload': <UploadPage />,
  '/jobs': <JobsPage />,
};

/**
 * Route table. Routes are driven from the same NAV config that builds the
 * sidebar, so the two never drift; each migrated page is registered in PAGES.
 */
export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to={DEFAULT_PATH} replace />} />
        {NAV_ITEMS.map((item) => (
          <Route key={item.path} path={item.path} element={PAGES[item.path] ?? <Placeholder title={item.label} />} />
        ))}
        <Route path="*" element={<Placeholder title="Not found" />} />
      </Route>
    </Routes>
  );
}
