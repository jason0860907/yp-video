/** Association Train — train a ranker on the reviewed events, and choose what
 *  the learned diagnostics run on.
 *
 *  Its own page rather than a card under the labeling work, for the same
 *  reason Action Train and ReID Train are: training is a batch operation on
 *  the whole corpus, and reading it next to one video's events implied it was
 *  about that video.
 */
import { AssociationTrainingCard } from '@/components/association/AssociationTrainingCard';

export function AssociationTrainPage() {
  return (
    <div className="space-y-4">
      <AssociationTrainingCard />
    </div>
  );
}
