import type { ReactElement } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { DEFAULT_PATH, NAV_ITEMS } from '@/components/layout/nav';
import { Placeholder } from '@/components/Placeholder';
import { JobsPage } from '@/pages/JobsPage';
import { DownloadPage } from '@/pages/DownloadPage';
import { DetectPage } from '@/pages/DetectPage';
import { ActionPredictPage } from '@/pages/ActionPredictPage';
import { ActionTrainPage } from '@/pages/ActionTrainPage';
import { ActionAnnotatePage } from '@/pages/ActionAnnotatePage';
import { SpotTrainPage } from '@/pages/SpotTrainPage';
import { SpotPredictPage } from '@/pages/SpotPredictPage';
import { UploadPage } from '@/pages/UploadPage';
import { CutPage } from '@/pages/CutPage';
import { AnnotatePage } from '@/pages/AnnotatePage';
import { ReidPredictPage } from '@/pages/ReidPredictPage';
import { PlayerDetectionPage } from '@/pages/PlayerDetectionPage';
import { TrackingPage } from '@/pages/TrackingPage';
import { AssociationLabelPage } from '@/pages/AssociationLabelPage';
import { AssociationPredictPage } from '@/pages/AssociationPredictPage';
import { AssociationTrainPage } from '@/pages/AssociationTrainPage';
import { FusionTrainPage } from '@/pages/FusionTrainPage';
import { ReidLabelPage } from '@/pages/ReidLabelPage';
import { ReidTrainPage } from '@/pages/ReidTrainPage';

/** Migrated pages, by route. Paths absent here fall back to a Placeholder. */
const PAGES: Record<string, ReactElement> = {
  '/download': <DownloadPage />,
  '/cut': <CutPage />,
  '/rally-vlm-predict': <DetectPage />,
  '/annotate': <AnnotatePage />,
  '/spot-train': <SpotTrainPage />,
  '/spot-predict': <SpotPredictPage />,
  '/action-predict': <ActionPredictPage />,
  '/action-train': <ActionTrainPage />,
  '/action-annotate': <ActionAnnotatePage />,
  '/tracking': <TrackingPage />,
  '/player-detection': <PlayerDetectionPage />,
  '/reid-predict': <ReidPredictPage />,
  '/association-predict': <AssociationPredictPage />,
  '/association-label': <AssociationLabelPage />,
  '/association-train': <AssociationTrainPage />,
  '/fusion-train': <FusionTrainPage />,
  '/reid-label': <ReidLabelPage />,
  '/reid-train': <ReidTrainPage />,
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
