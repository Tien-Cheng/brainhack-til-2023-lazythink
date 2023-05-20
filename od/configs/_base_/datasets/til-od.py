# dataset settings
dataset_type = "CocoDataset"
data_root = "data/"
classes = "plushie"


# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/segmentation/cityscapes/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#          'data/': 's3://openmmlab/datasets/segmentation/'
#      }))
backend_args = None

# Import custom noise pipeline
custom_imports = dict(
    imports=["configs.custom_transform"],
    allow_failed_imports=False,
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomGaussian", prob=0.2),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file="labels-train.json",
        data_prefix=dict(img="images/train"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file="labels-val.json",
        data_prefix=dict(img="images/validation"),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file="labels-test.json",
        data_prefix=dict(img="images/test"),
        pipeline=test_pipeline,
    ),
)


val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "/labels-val.json",
    metric="bbox",
    backend_args=backend_args,
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "/labels-test.json",
    metric="bbox",
    format_only=True,
    outfile_prefix="./work_dirs/coco_detection/test",
)
