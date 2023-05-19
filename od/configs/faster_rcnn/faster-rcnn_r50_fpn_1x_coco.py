_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../_base_/datasets/til-od.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

# We also need to change the num_classes in head to match the dataset's
model = dict(roi_head=dict(bbox_head=dict(num_classes=200)))

# We can use the pre-trained Faster-RCNN model to obtain higher performance
load_from = (
    "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
    "faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP"
    "-0.384_20200504_210434-a5d8aa15.pth"
)
