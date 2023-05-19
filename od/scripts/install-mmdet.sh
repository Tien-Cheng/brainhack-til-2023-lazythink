#!/bin/sh

# Install mim engine
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install mmdet
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..
