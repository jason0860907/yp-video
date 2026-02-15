"""MambaTAD configuration for volleyball rally detection.

This config is designed for:
- I3D/R3D features (512-dimensional from R3D-18)
- Single class: "rally"
- Variable length videos (typical volleyball match: 30-60 min)

Reference: MambaTAD paper and official configs
"""

# Model settings
model = dict(
    type="MambaTAD",
    backbone=dict(
        type="MambaBackbone",
        in_channels=512,  # R3D-18 feature dimension
        hidden_dim=256,
        n_layers=4,
        dropout=0.1,
        bidirectional=True,
    ),
    neck=dict(
        type="FPN",
        in_channels=[256],
        out_channels=256,
        num_outs=6,
    ),
    head=dict(
        type="TADHead",
        num_classes=1,  # Single class: rally
        in_channels=256,
        feat_channels=256,
        num_convs=2,
        cls_prior_prob=0.01,
        regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        center_sampling=True,
        center_sampling_radius=1.5,
        loss_cls=dict(
            type="FocalLoss",
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
        ),
        loss_reg=dict(
            type="DIoULoss",
            loss_weight=1.0,
        ),
    ),
)

# Dataset settings
dataset = dict(
    type="TADDataset",
    anno_file="tad/data/annotations/volleyball_anno.json",
    data_prefix="tad/data/features",
    pipeline=[
        dict(type="LoadFeatures"),
        dict(type="Normalize", mean=0.0, std=1.0),
        dict(type="Pad", size_divisor=32),
    ],
    test_pipeline=[
        dict(type="LoadFeatures"),
        dict(type="Normalize", mean=0.0, std=1.0),
        dict(type="Pad", size_divisor=32),
    ],
)

# Training settings
train_cfg = dict(
    assigner=dict(
        type="MaxIoUAssigner",
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0.0,
    ),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(type="soft_nms", iou_thr=0.5, min_score=0.001),
    max_per_video=100,
)

# Optimizer
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    betas=(0.9, 0.999),
)

optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=0.01,
)

# Runtime settings
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=["mAP"])
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

# Distributed settings
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# Data loader
data = dict(
    samples_per_gpu=2,  # Batch size per GPU
    workers_per_gpu=4,
    train=dict(
        type=dataset["type"],
        anno_file=dataset["anno_file"],
        data_prefix=dataset["data_prefix"],
        pipeline=dataset["pipeline"],
        subset="train",
    ),
    val=dict(
        type=dataset["type"],
        anno_file=dataset["anno_file"],
        data_prefix=dataset["data_prefix"],
        pipeline=dataset["test_pipeline"],
        subset="test",
    ),
    test=dict(
        type=dataset["type"],
        anno_file=dataset["anno_file"],
        data_prefix=dataset["data_prefix"],
        pipeline=dataset["test_pipeline"],
        subset="test",
    ),
)

# Label mapping
label_map = {
    0: "rally",
}
