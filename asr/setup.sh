#!/bin/bash
python -m venv env
source env/bin/activate
pip install librosa jiwer protobuf==3.20.3
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchtext==0.14.1 torchaudio==0.13.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade numba>=0.54.1
BRANCH = 'r1.18.0'
python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]
pip freeze > requirements-nemo.txt