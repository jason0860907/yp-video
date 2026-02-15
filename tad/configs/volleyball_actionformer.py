"""ActionFormer configuration for volleyball rally detection.

This config uses:
- R3D-18 features (512-dimensional)
- Single class: "rally"
- OpenTAD framework
"""

_base_ = [
    "../../OpenTAD/configs/_base_/models/actionformer.py",
]

# Override model settings for volleyball
model = dict(
    projection=dict(
        in_channels=512,  # R3D-18 feature dimension
        out_channels=256,
        arch=(2, 2, 5),
        input_pdrop=0.2,
    ),
    neck=dict(
        in_channels=256,
        out_channels=256,
    ),
    rpn_head=dict(
        num_classes=1,  # Single class: rally
        in_channels=256,
        feat_channels=256,
    ),
)

# Dataset settings
dataset_type = "ThumosPaddingDataset"
annotation_path = "tad/data/annotations/volleyball_anno.json"
class_map = "tad/data/annotations/category_idx.txt"
data_path = "tad/data/features/"

trunc_len = 2304

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=64,  # R3D extraction stride (64 frames at 60fps)
        sample_stride=1,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=trunc_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=64,
        sample_stride=1,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=64,
        sample_stride=1,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

# Solver settings
solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=300)

# Evaluation
evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

# Post processing
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=200,
        iou_threshold=0.1,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.7,
    ),
    save_dict=False,
)

# Workflow
workflow = dict(
    logging_interval=20,
    checkpoint_interval=5,
    val_loss_interval=1,
    val_eval_interval=5,
    val_start_epoch=10,
)

work_dir = "tad/checkpoints/actionformer"
