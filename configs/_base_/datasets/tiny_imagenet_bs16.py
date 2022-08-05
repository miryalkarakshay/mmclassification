# dataset settings
dataset_type = "TinyImageNet"

img_norm_cfg = dict(
    mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821], to_rgb=False
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomCrop", size=64, padding=4),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix="data/tiny-imagenet-200/train",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix="data/tiny-imagenet-200/val/images",
        ann_file="data/tiny-imagenet-200/val/annotations.txt",
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_prefix="data/tiny-imagenet-200/val/images",
        ann_file="data/tiny-imagenet-200/val/annotations.txt",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
evaluation = dict(interval=1, metric="accuracy")
