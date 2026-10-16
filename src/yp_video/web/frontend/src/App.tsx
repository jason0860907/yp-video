import type { ReactElement } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { DEFAULT_PATH, NAV_ITEMS } from '@/components/layout/nav';
import { Card } from '@/components/ui/Card';
import { JobsPage } from '@/pages/JobsPage';
import { DownloadPage } from '@/pages/DownloadPage';
import { UploadPage } from '@/pages/UploadPage';
import { CutPage } from '@/pages/CutPage';
import { LabelPage } from '@/pages/label/LabelPage';
import { AnalysisPage } from '@/pages/analysis/AnalysisPage';
import { ReidPredictPage } from '@/pages/ReidPredictPage';
import { PlayerDetectionPage } from '@/pages/PlayerDetectionPage';
import { TrackingPage } from '@/pages/TrackingPage';
import { FusionTrainPage } from '@/pages/FusionTrainPage';
import { InferencePage } from '@/pages/InferencePage';
import { ReidTrainPage } from '@/pages/ReidTrainPage';
import { AppReviewPage } from '@/pages/app-review/AppReviewPage';
import { AuditPage } from '@/pages/AuditPage';

/** Every NAV route's page. */
const PAGES: Record<string, ReactElement> = {
  '/download': <DownloadPage />,
  '/cut': <CutPage />,
  '/label': <LabelPage />,
  '/analysis': <AnalysisPage />,
  '/tracking': <TrackingPage />,
  '/player-detection': <PlayerDetectionPage />,
  '/reid-predict': <ReidPredictPage />,
  '/fusion-train': <FusionTrainPage />,
  '/inference': <InferencePage />,
  '/reid-train': <ReidTrainPage />,
  '/upload': <UploadPage />,
  '/jobs': <JobsPage />,
  '/audit': <AuditPage />,
  '/app-review': <AppReviewPage />,
};

function NotFound() {
  return (
    <div className="mx-auto max-w-5xl">
      <Card label="Not found">
        <p className="text-sm text-text-secondary">No page lives at this address.</p>
      </Card>
    </div>
  );
}

/**
 * Route table. Routes are driven from the same NAV config that builds the
 * sidebar, so the two never drift; each page is registered in PAGES.
 */
export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to={DEFAULT_PATH} replace />} />
        {NAV_ITEMS.map((item) => (
          <Route key={item.path} path={item.path} element={PAGES[item.path]} />
        ))}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
