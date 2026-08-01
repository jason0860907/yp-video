import { Button } from '@/components/ui/Button';
import { copyText } from '@/lib/download';
import { toast } from '@/components/feedback/toast';

/** The label pickers load on selection, so this is their only button:
 *  the picked video's filename, onto the clipboard. */
export function CopyFilenameButton({ name }: { name: string }) {
  return (
    <Button
      className="h-9 py-0"
      onClick={() => {
        if (!name) {
          toast.warning('No video selected');
          return;
        }
        copyText(name).then(
          () => toast.success(`Copied ${name}`),
          () => toast.error('Copy failed'),
        );
      }}
    >
      Copy Filename
    </Button>
  );
}
