_base_ = [
    "../_base_/models/ecanet/ecanet_resnet18_cifar.py",
    "../_base_/datasets/cifar10_bs16.py",
    "../_base_/schedules/cifar10_bs128.py",
    "../_base_/default_runtime.py",
]

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", step=[30, 60, 90])
runner = dict(type="EpochBasedRunner", max_epochs=100)


checkpoint_config = dict(interval=5)

# "auto" means automatically select the metrics to compare.
# You can also use a specific key like "accuracy_top-1".
evaluation = dict(
    interval=5, save_best="auto", metric="accuracy", metric_options={"topk": (1, 5)}
)

log_config = dict(
    interval=100,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
