# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="Dynamic_Resnet_imagenet",
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=200,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
