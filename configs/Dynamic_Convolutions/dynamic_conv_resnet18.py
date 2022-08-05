_base_ = [
    "../_base_/models/Dynamic_Convolutions/dynamic_conv_resnet18_cifar.py",
    "../_base_/datasets/cifar10_bs16.py",
    "../_base_/schedules/cifar10_bs128.py",
    "../_base_/default_runtime.py",
]

dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/cifar10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True))


checkpoint_config = dict(interval=5)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)


# "auto" means automatically select the metrics to compare.
# You can also use a specific key like "accuracy_top-1".
evaluation = dict(interval=5, save_best="auto", metric='accuracy', metric_options={'topk': (1, 5)})

custom_hooks = [
    dict(type='TemperatureUpdateHook')
]

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]) 