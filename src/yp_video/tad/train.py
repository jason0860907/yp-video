"""Train ActionFormer model for volleyball rally detection."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn

from yp_video.config import (
    ACTIONFORMER_DIR,
    FEATURES_DIR,
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


def _run_validation(val_loader, model, det_eval, det_eval_by_src, val_videos_by_src):
    """Run inference once and evaluate against the overall + per-source evaluators.

    Returns:
        overall_mAP (float): mean of mAP across tIoU thresholds (matches valid_one_epoch)
        per_source (dict): {source: {"mAP": ..., "tiou_mAP": [...], "n_videos": ...}}
    """
    import pandas as pd

    model.eval()
    pieces = {"video-id": [], "t-start": [], "t-end": [], "label": [], "score": []}
    with torch.no_grad():
        for video_list in val_loader:
            output = model(video_list)
            for vid in output:
                if vid["segments"].shape[0] == 0:
                    continue
                n = vid["segments"].shape[0]
                pieces["video-id"].extend([vid["video_id"]] * n)
                pieces["t-start"].append(vid["segments"][:, 0])
                pieces["t-end"].append(vid["segments"][:, 1])
                pieces["label"].append(vid["labels"])
                pieces["score"].append(vid["scores"])

    if not pieces["t-start"]:
        # Model produced zero detections across the entire val set; nothing to score.
        return 0.0, {src: {"mAP": 0.0, "tiou_mAP": [0.0] * len(e.tiou_thresholds),
                           "n_videos": len(val_videos_by_src.get(src, []))}
                     for src, e in det_eval_by_src.items()}

    preds_df = pd.DataFrame({
        "video-id": pieces["video-id"],
        "t-start": torch.cat(pieces["t-start"]).numpy(),
        "t-end": torch.cat(pieces["t-end"]).numpy(),
        "label": torch.cat(pieces["label"]).numpy(),
        "score": torch.cat(pieces["score"]).numpy(),
    })

    # Overall — pass a copy because evaluate() mutates the 'label' column in place.
    det_eval.evaluate(preds_df.copy(), verbose=False)
    overall_mAP = float(det_eval.ap.mean())

    per_source: dict[str, dict] = {}
    for src, evaluator in det_eval_by_src.items():
        videos = val_videos_by_src.get(src, [])
        sub = preds_df[preds_df["video-id"].isin(videos)].copy()
        try:
            evaluator.evaluate(sub if len(sub) else preds_df.iloc[0:0].copy(), verbose=False)
            per_source[src] = {
                "mAP": round(float(evaluator.ap.mean()), 4),
                "tiou_mAP": [round(float(v), 4) for v in evaluator.ap.mean(axis=1)],
                "n_videos": len(videos),
                "n_preds": int(len(sub)),
            }
        except Exception as e:
            per_source[src] = {"error": str(e), "n_videos": len(videos)}

    return overall_mAP, per_source


def _build_balanced_loader(train_dataset, rng_generator, alpha: float, loader_cfg: dict):
    """Replace uniform shuffling with per-source WeightedRandomSampler.

    Why: training videos are heavily skewed toward a few broadcast styles
    (tpvl alone is ~36% of the train set). Uniform sampling lets the loss
    be dominated by majority sources and the model under-fits minority
    sources (U19, VNL). Inverse-frequency weighting equalizes the
    per-source expected count per epoch so every style contributes
    comparable gradient signal.
    """
    from collections import Counter

    from torch.utils.data import DataLoader, WeightedRandomSampler
    from libs.datasets.data_utils import trivial_batch_collator, worker_init_reset_seed

    from yp_video.tad.convert_annotations import _source_key

    sources = [_source_key(item["id"]) for item in train_dataset.data_list]
    counts = Counter(sources)
    weights = [(1.0 / counts[s]) ** alpha for s in sources]

    n = len(sources)
    total_w = sum(weights)
    per_source = []
    print(f"\nBalanced sampler (alpha={alpha}):")
    print(f"  {'source':<14s} {'count':>6s} {'weight':>10s} {'expected/epoch':>16s}")
    for src in sorted(counts):
        c = counts[src]
        w = (1.0 / c) ** alpha
        expected = (w * c / total_w) * n
        per_source.append({
            "source": src, "count": c, "weight": round(w, 6),
            "expected_per_epoch": round(expected, 2),
        })
        print(f"  {src:<14s} {c:>6d} {w:>10.4f} {expected:>16.1f}")

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=n, replacement=True, generator=rng_generator
    )
    loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        sampler=sampler,
        drop_last=True,
        generator=rng_generator,
        persistent_workers=True,
    )
    info = {"alpha": alpha, "per_source": per_source, "n_train_videos": n}
    return loader, info


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
        "--model",
        type=str,
        default=None,
        help="V-JEPA model size (base/large/giant/gigantic) — overrides feat_folder & input_dim in config",
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
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5,
        help="Validation frequency (epochs)",
    )
    parser.add_argument(
        "--balanced-sampler",
        dest="balanced_sampler",
        action="store_true",
        default=True,
        help="Use WeightedRandomSampler to equalize per-source frequency (default: on)",
    )
    parser.add_argument(
        "--no-balanced-sampler",
        dest="balanced_sampler",
        action="store_false",
        help="Disable balanced sampling; fall back to uniform shuffle",
    )
    parser.add_argument(
        "--sampler-alpha",
        type=float,
        default=0.5,
        help="Smoothing for inverse-frequency weights: weight = (1/count_in_source)**alpha. "
             "1.0 = full balance, 0.0 = uniform, 0.5 = sqrt-sampling.",
    )
    # ── Optional config overrides (None = keep YAML value) ────────────────
    parser.add_argument("--lr", type=float, default=None,
                        help="Override opt.learning_rate")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override opt.epochs (cosine length / training horizon after warmup)")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Override opt.warmup_epochs")
    parser.add_argument("--schedule", type=str, default=None,
                        choices=["cosine", "multistep", "constant"],
                        help="Override opt.schedule_type. 'constant' = flat after warmup (no decay).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override loader.batch_size")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override opt.weight_decay")
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
        ANETdetection,
        ModelEma,
        fix_random_seed,
        make_optimizer,
        make_scheduler,
        save_checkpoint,
        train_one_epoch,
        valid_one_epoch,
    )

    # Load config
    cfg = load_config(str(args.config))
    cfg["devices"] = ["cuda:0"]
    # Expand ~ in paths
    for key in ("json_file", "feat_folder"):
        if key in cfg["dataset"]:
            cfg["dataset"][key] = os.path.expanduser(cfg["dataset"][key])

    # Override opt/loader params from CLI when supplied
    if args.lr is not None:
        cfg["opt"]["learning_rate"] = args.lr
    if args.epochs is not None:
        cfg["opt"]["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        cfg["opt"]["warmup_epochs"] = args.warmup_epochs
    if args.weight_decay is not None:
        cfg["opt"]["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        cfg["loader"]["batch_size"] = args.batch_size
    if args.schedule == "constant":
        # Flat-after-warmup: multistep with a milestone past the horizon
        cfg["opt"]["schedule_type"] = "multistep"
        cfg["opt"]["schedule_steps"] = [10**9]
        cfg["opt"]["schedule_gamma"] = 0.1
    elif args.schedule is not None:
        cfg["opt"]["schedule_type"] = args.schedule

    # Override feature folder & input_dim when --model is specified
    from yp_video.tad.extract_features import MODEL_CONFIGS
    model_name = args.model or "base"
    mcfg = MODEL_CONFIGS[model_name]
    if args.model:
        feat_folder = str(FEATURES_DIR / mcfg.dir_suffix)
        cfg["dataset"]["feat_folder"] = feat_folder
        cfg["dataset"]["input_dim"] = mcfg.feat_dim
        cfg["model"]["input_dim"] = mcfg.feat_dim
        print(f"Model override: feat_folder={feat_folder}, input_dim={mcfg.feat_dim}")

    # Override output folder
    if not args.work_dir:
        stamp = datetime.now().strftime("%Y-%m%d-%H%M")
        args.work_dir = TAD_CHECKPOINTS_DIR / "actionformer" / mcfg.dir_suffix / stamp

    ckpt_folder = str(args.work_dir)
    os.makedirs(ckpt_folder, exist_ok=True)

    pprint(cfg)

    # Fix random seed
    rng_generator = fix_random_seed(args.seed, include_cuda=True)

    # Create dataset & dataloader (run from project root for relative paths)
    os.chdir(str(PROJECT_ROOT))

    # Training set
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars["empty_label_ids"]

    sampler_info: dict | None = None
    if args.balanced_sampler:
        train_loader, sampler_info = _build_balanced_loader(
            train_dataset, rng_generator, args.sampler_alpha, cfg["loader"]
        )
    else:
        train_loader = make_data_loader(
            train_dataset, True, rng_generator, **cfg["loader"]
        )

    # Validation set
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    val_db_vars = val_dataset.get_attributes()
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg["loader"]["num_workers"]
    )

    # Evaluator (tIoU 0.3 ~ 0.7)
    import numpy as np
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds=np.linspace(0.3, 0.7, 5),
    )

    # Per-source evaluators: same GT file but filtered to one source's videos.
    # Lets us see which broadcast styles the model is actually fitting vs. ignoring.
    from collections import defaultdict as _dd

    from yp_video.tad.convert_annotations import _source_key as _sk

    val_videos_by_src: dict[str, list[str]] = _dd(list)
    for _it in val_dataset.data_list:
        val_videos_by_src[_sk(_it["id"])].append(_it["id"])

    det_eval_by_src: dict[str, ANETdetection] = {}
    for _src, _vids in val_videos_by_src.items():
        _e = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=np.linspace(0.3, 0.7, 5),
        )
        _e.ground_truth = _e.ground_truth[_e.ground_truth["video-id"].isin(_vids)].reset_index(drop=True)
        if len(_e.ground_truth) > 0:
            det_eval_by_src[_src] = _e

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

    # Training log (JSONL — one JSON object per line)
    log_path = os.path.join(ckpt_folder, "train_log.jsonl")

    # Write a _meta header with the effective hyperparameters so each run is
    # self-describing (no need to read config.txt to know what produced the
    # curve). Skip when resuming so we don't duplicate headers in the same file.
    if not args.resume and not os.path.exists(log_path):
        from collections import Counter

        from yp_video.tad.convert_annotations import _source_key

        train_by_src = Counter(_source_key(it["id"]) for it in train_dataset.data_list)
        val_by_src = Counter(_source_key(it["id"]) for it in val_dataset.data_list)
        dataset_summary = []
        for src in sorted(set(train_by_src) | set(val_by_src)):
            dataset_summary.append({
                "source": src,
                "train": train_by_src.get(src, 0),
                "val": val_by_src.get(src, 0),
            })

        meta = {
            "_meta": True,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "feat_dim": cfg["dataset"]["input_dim"],
            "seed": args.seed,
            "opt": {
                "type": cfg["opt"]["type"],
                "learning_rate": cfg["opt"]["learning_rate"],
                "weight_decay": cfg["opt"]["weight_decay"],
                "warmup": cfg["opt"]["warmup"],
                "warmup_epochs": cfg["opt"]["warmup_epochs"],
                "epochs": cfg["opt"]["epochs"],
                "schedule_type": cfg["opt"]["schedule_type"],
                "schedule_steps": cfg["opt"].get("schedule_steps"),
                "schedule_gamma": cfg["opt"].get("schedule_gamma"),
            },
            "loader": {
                "batch_size": cfg["loader"]["batch_size"],
                "num_workers": cfg["loader"]["num_workers"],
            },
            "dataset": {
                "n_train_videos": len(train_dataset),
                "n_val_videos": len(val_dataset),
                "by_source": dataset_summary,
            },
            "balanced_sampler": bool(args.balanced_sampler),
            "sampler": sampler_info,
        }
        with open(log_path, "w") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # Training loop
    max_epochs = cfg["opt"].get(
        "early_stop_epochs",
        cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"],
    )
    best_mAP = 0.0
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

        # Validation
        if (
            args.eval_freq > 0
            and ((epoch + 1) % args.eval_freq == 0 or (epoch + 1) == max_epochs)
        ):
            print(f"\n[Validation] Epoch {epoch + 1}")
            avg_mAP, per_source = _run_validation(
                val_loader,
                model_ema.module,
                det_eval,
                det_eval_by_src,
                val_videos_by_src,
            )
            # Detailed per-tIoU metrics are stored on the evaluator
            mAP_per_tiou = det_eval.ap.mean(axis=1) if det_eval.ap is not None else None
            mRecall = det_eval.recall.mean(axis=2) if hasattr(det_eval, 'recall') and det_eval.recall is not None else None
            tiou_thresholds = det_eval.tiou_thresholds

            is_best = avg_mAP > best_mAP
            if is_best:
                best_mAP = avg_mAP

            # Build per-tIoU metrics
            tiou_metrics = {}
            if mAP_per_tiou is not None:
                for i, tiou in enumerate(tiou_thresholds):
                    key = f"{tiou:.2f}"
                    tiou_metrics[key] = {
                        "mAP": round(float(mAP_per_tiou[i]), 4),
                        "recall": round(float(mRecall[i][0]), 4),
                    }

            entry = {
                "epoch": epoch + 1,
                "mAP": round(float(avg_mAP), 4),
                "tiou": tiou_metrics,
                "per_source": per_source,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"[Epoch {epoch + 1}] mAP = {avg_mAP:.4f}  best = {best_mAP:.4f}")
            if per_source:
                print("  per-source mAP: " + "  ".join(
                    f"{s}={m.get('mAP', '-'):.3f}" if isinstance(m.get('mAP'), float) else f"{s}=err"
                    for s, m in sorted(per_source.items())
                ))

            # Save best checkpoint
            if is_best:
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
                    file_name="best.pth.tar",
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

    print(f"\nTraining complete! Best mAP = {best_mAP:.4f}")


if __name__ == "__main__":
    main()
