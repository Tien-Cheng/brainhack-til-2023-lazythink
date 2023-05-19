# dataset settings
dataset_type = "CocoDataset"
data_root = "data/"
classes = (
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "class_8",
    "class_9",
    "class_10",
    "class_11",
    "class_12",
    "class_13",
    "class_14",
    "class_15",
    "class_16",
    "class_17",
    "class_18",
    "class_19",
    "class_20",
    "class_21",
    "class_22",
    "class_23",
    "class_24",
    "class_25",
    "class_26",
    "class_27",
    "class_28",
    "class_29",
    "class_30",
    "class_31",
    "class_32",
    "class_33",
    "class_34",
    "class_35",
    "class_36",
    "class_37",
    "class_38",
    "class_39",
    "class_40",
    "class_41",
    "class_42",
    "class_43",
    "class_44",
    "class_45",
    "class_46",
    "class_47",
    "class_48",
    "class_49",
    "class_50",
    "class_51",
    "class_52",
    "class_53",
    "class_54",
    "class_55",
    "class_56",
    "class_57",
    "class_58",
    "class_59",
    "class_60",
    "class_61",
    "class_62",
    "class_63",
    "class_64",
    "class_65",
    "class_66",
    "class_67",
    "class_68",
    "class_69",
    "class_70",
    "class_71",
    "class_72",
    "class_73",
    "class_74",
    "class_75",
    "class_76",
    "class_77",
    "class_78",
    "class_79",
    "class_80",
    "class_81",
    "class_82",
    "class_83",
    "class_84",
    "class_85",
    "class_86",
    "class_87",
    "class_88",
    "class_89",
    "class_90",
    "class_91",
    "class_92",
    "class_93",
    "class_94",
    "class_95",
    "class_96",
    "class_97",
    "class_98",
    "class_99",
    "class_100",
    "class_101",
    "class_102",
    "class_103",
    "class_104",
    "class_105",
    "class_106",
    "class_107",
    "class_108",
    "class_109",
    "class_110",
    "class_111",
    "class_112",
    "class_113",
    "class_114",
    "class_115",
    "class_116",
    "class_117",
    "class_118",
    "class_119",
    "class_120",
    "class_121",
    "class_122",
    "class_123",
    "class_124",
    "class_125",
    "class_126",
    "class_127",
    "class_128",
    "class_129",
    "class_130",
    "class_131",
    "class_132",
    "class_133",
    "class_134",
    "class_135",
    "class_136",
    "class_137",
    "class_138",
    "class_139",
    "class_140",
    "class_141",
    "class_142",
    "class_143",
    "class_144",
    "class_145",
    "class_146",
    "class_147",
    "class_148",
    "class_149",
    "class_150",
    "class_151",
    "class_152",
    "class_153",
    "class_154",
    "class_155",
    "class_156",
    "class_157",
    "class_158",
    "class_159",
    "class_160",
    "class_161",
    "class_162",
    "class_163",
    "class_164",
    "class_165",
    "class_166",
    "class_167",
    "class_168",
    "class_169",
    "class_170",
    "class_171",
    "class_172",
    "class_173",
    "class_174",
    "class_175",
    "class_176",
    "class_177",
    "class_178",
    "class_179",
    "class_180",
    "class_181",
    "class_182",
    "class_183",
    "class_184",
    "class_185",
    "class_186",
    "class_187",
    "class_188",
    "class_189",
    "class_190",
    "class_191",
    "class_192",
    "class_193",
    "class_194",
    "class_195",
    "class_196",
    "class_197",
    "class_198",
    "class_199",
)


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

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
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
    batch_size=1,
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
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file="test.json",
        data_prefix=dict(img="images/validation"),
        pipeline=test_pipeline,
    ),
)


val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "/labels-val.json",
    metric="bbox",
    backend_args=backend_args,
)

test_evaluator = val_evaluator
