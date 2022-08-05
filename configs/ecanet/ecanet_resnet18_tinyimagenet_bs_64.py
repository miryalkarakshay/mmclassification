_base_ = [
    "../_base_/models/ecanet/ecanet_resnet18_tinyimagenet.py",
    "../_base_/datasets/tiny_imagenet_bs16.py",
    "../_base_/schedules/cifar10_bs128.py",
    "../_base_/default_runtime.py",
]

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
    samples_per_gpu=64,
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

checkpoint_config = dict(interval=5)

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", step=[30, 60, 90])
runner = dict(type="EpochBasedRunner", max_epochs=100)

# "auto" means automatically select the metrics to compare.
# You can also use a specific key like "accuracy_top-1".
evaluation = dict(
    interval=5, save_best="auto", metric="accuracy", metric_options={"topk": (1, 5)}
)

log_config = dict(
    interval=100,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
