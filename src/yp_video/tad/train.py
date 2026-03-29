"""Train ActionFormer model for volleyball rally detection."""

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn

from yp_video.config import (
    ACTIONFORMER_DIR,
    PROJECT_ROOT,
    TAD_CHECKPOINTS_DIR,
    TAD_CONFIGS_DIR,
)


def _setup_actionformer():
    """Add ActionFormer to sys.path for imports."""
    af_dir = str(ACTIONFORMER_DIR)
    af_utils = str(ACTIONFORMER_DIR / "libs" / "utils")
    if af_dir not in sys.path:
        sys.path.insert(0, af_dir)
    if af_utils not in sys.path:
        sys.path.insert(0, af_utils)


def check_actionformer_installed() -> bool:
    """Check if ActionFormer submodule is present."""
    return (ACTIONFORMER_DIR / "libs" / "modeling" / "meta_archs.py").exists()


def main():
    parser = argparse.ArgumentParser(description="Train ActionFormer for volleyball")
    parser.add_argument(
        "--config",
        type=Path,
        default=TAD_CONFIGS_DIR / "volleyball_actionformer.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Print frequency (iterations)",
    )
    parser.add_argument(
        "--ckpt-freq",
        type=int,
        default=5,
        help="Checkpoint frequency (epochs)",
    )
    args = parser.parse_args()

    if not check_actionformer_installed():
        print("Error: ActionFormer not found!")
        print("Please init the submodule first:")
        print("  git submodule update --init")
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Set GPU before any CUDA operations
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    _setup_actionformer()

    from libs.core import load_config
    from libs.datasets import make_dataset, make_data_loader
    from libs.modeling import make_meta_arch
    from libs.utils import (
        ModelEma,
        fix_random_seed,
        make_optimizer,
        make_scheduler,
        save_checkpoint,
        train_one_epoch,
    )

    # Load config
    cfg = load_config(str(args.config))
    cfg["devices"] = ["cuda:0"]
    # Expand ~ in paths
    for key in ("json_file", "feat_folder"):
        if key in cfg["dataset"]:
            cfg["dataset"][key] = os.path.expanduser(cfg["dataset"][key])

    # Override output folder
    if not args.work_dir:
        today = date.today().strftime("%Y-%m%d")
        args.work_dir = TAD_CHECKPOINTS_DIR / "actionformer" / today

    ckpt_folder = str(args.work_dir)
    os.makedirs(ckpt_folder, exist_ok=True)

    pprint(cfg)

    # Fix random seed
    rng_generator = fix_random_seed(args.seed, include_cuda=True)

    # Create dataset & dataloader (run from project root for relative paths)
    os.chdir(str(PROJECT_ROOT))
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars["empty_label_ids"]

    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg["loader"]
    )

    # Create model
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=[0])

    # Optimizer & scheduler
    optimizer = make_optimizer(model, cfg["opt"])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)

    # EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and args.resume.exists():
        checkpoint = torch.load(str(args.resume), map_location="cuda:0")
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"=> Resumed from epoch {start_epoch}")
        del checkpoint

    # Save config
    with open(os.path.join(ckpt_folder, "config.txt"), "w") as fid:
        pprint(cfg, stream=fid)

    # Training loop
    max_epochs = cfg["opt"].get(
        "early_stop_epochs",
        cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"],
    )
    print(f"\nStart training ActionFormer for {max_epochs} epochs ...")

    for epoch in range(start_epoch, max_epochs):
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg["train_cfg"]["clip_grad_l2norm"],
            print_freq=args.print_freq,
        )

        # Save checkpoint periodically
        if (
            ((epoch + 1) == max_epochs)
            or (args.ckpt_freq > 0 and (epoch + 1) % args.ckpt_freq == 0)
        ):
            save_states = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "state_dict_ema": model_ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name=f"epoch_{epoch + 1:03d}.pth.tar",
            )

    print("Training complete!")


if __name__ == "__main__":
    main()
