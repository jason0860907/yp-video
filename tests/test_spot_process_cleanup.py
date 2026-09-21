"""Real child processes must not survive a failed streaming callback."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from yp_video.action import predict, prelabel


@pytest.mark.parametrize('callback', ['events', 'progress'])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('ignore_term', [False, True])
def test_callback_failure_reaps_inference(tmp_path, callback, error_type, ignore_term):
    spawned = []
    popen = subprocess.Popen
    def launch(*args, **kwargs):
        proc = popen(*args, **kwargs)
        spawned.append(proc)
        return proc
    script = '\n'.join([
        'import signal, time',
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)' if ignore_term else 'pass',
        'print(\'SPOT_PARTIAL {"task":"rally","events":[]}\', flush=True)',
        'time.sleep(60)',
    ])
    error = error_type('report failed')
    def fail(*args):
        raise error
    try:
        with (
            mock.patch.object(prelabel, 'spot_available', return_value=True),
            mock.patch.object(prelabel, 'build_command', return_value=[sys.executable, '-c', script]),
            mock.patch.object(predict, 'SPOT_DIR', tmp_path),
            mock.patch.object(predict, '_STOP_TIMEOUT_SECONDS', 0.1),
            mock.patch.object(predict, '_spot_progress_ratio', return_value=0.5 if callback == 'progress' else None),
            mock.patch.object(predict.subprocess, 'Popen', side_effect=launch),
        ):
            with pytest.raises(error_type) as caught:
                predict.run_spot_inference('video.mp4', checkpoint=Path('model.pt'), tasks=('rally',),
                    on_events=fail if callback == 'events' else None,
                    on_progress=fail if callback == 'progress' else None)
        assert caught.value is error
        assert spawned[0].returncode is not None
        assert spawned[0].stdout.closed
        with pytest.raises(ChildProcessError):
            os.waitpid(spawned[0].pid, os.WNOHANG)
    finally:
        for proc in spawned:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def test_callback_failure_stops_decoder_descendant(tmp_path):
    child_pid_file = tmp_path / 'decoder.pid'
    child = ('import os, signal, time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
             f'Path({str(child_pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)')
    script = ('import subprocess, sys, time; from pathlib import Path\n'
              f'subprocess.Popen([sys.executable, "-c", {child!r}])\n'
              f'while not Path({str(child_pid_file)!r}).exists(): time.sleep(.01)\n'
              'print(\'SPOT_PARTIAL {"task":"rally","events":[]}\', flush=True)\n'
              'time.sleep(60)')
    def fail(*args):
        raise RuntimeError('HTTP 500')
    try:
        with (
            mock.patch.object(prelabel, 'spot_available', return_value=True),
            mock.patch.object(prelabel, 'build_command', return_value=[sys.executable, '-c', script]),
            mock.patch.object(predict, 'SPOT_DIR', tmp_path),
            mock.patch.object(predict, '_STOP_TIMEOUT_SECONDS', 0.1),
        ):
            with pytest.raises(RuntimeError, match='HTTP 500'):
                predict.run_spot_inference('video.mp4', checkpoint=Path('model.pt'), tasks=('rally',), on_events=fail)
        pid = int(child_pid_file.read_text())
        for _ in range(100):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        else:
            pytest.fail('Decoder child is still running')
    finally:
        if child_pid_file.exists():
            try:
                os.kill(int(child_pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
