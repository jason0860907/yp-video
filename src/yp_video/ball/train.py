"""Fine-tune RF-DETR on an exported ball set (ball/dataset.py): one class.

Starts from the COCO weights, whose "sports ball" already finds an isolated
ball; the pseudo-labels teach the volleyball in a crowd and drop the clutter.
The input stays at the 1024 that made the stock detector usable. Every round
starts from COCO again: only the labels improve between rounds.

  yp-video/.venv/bin/python -m yp_video.ball.train <dataset> <run> [--epochs N] [--resume CKPT]

``--resume`` takes a run's ``checkpoint_<epoch>.ckpt`` and continues it with
optimizer, schedule and EMA state; ``--epochs`` stays the run's total.

Needs the ``train`` dependency group (``uv sync --group train --inexact``).
"""

from __future__ import annotations

from yp_video.ball.detector import RESOLUTION, VARIANT
from yp_video.config import BALL_CHECKPOINTS_DIR, BALL_DATASETS_DIR

# Batch 4 takes ~11 GB at 1024; the worker's SPOT pass needs ~14 GB of the same
# 24 GB card, so batch 2 with twice the accumulation (same effective 16).
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LR = 1e-4
NUM_WORKERS = 8


def train(dataset: str, run: str, epochs: int, resume: str | None = None) -> None:
    import rfdetr

    model = getattr(rfdetr, VARIANT)(resolution=RESOLUTION)
    model.train(
        dataset_dir=str(BALL_DATASETS_DIR / dataset),
        output_dir=str(BALL_CHECKPOINTS_DIR / run),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        num_workers=NUM_WORKERS,
        class_names=["ball"],
        checkpoint_interval=1,
        tensorboard=True,
        progress_bar="tqdm",
        resume=resume,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.dataset, args.run, args.epochs, args.resume)
