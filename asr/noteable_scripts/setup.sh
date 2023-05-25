#!/bin/bash
pip install librosa
mamba create -n env python==3.9 -y
mamba run -n env pip install pyloudnorm Cython nemo_text_processing
mamba run -n env pip install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba run -n env pip install cudatoolkit -y
mamba run -n env pip install nemo_toolkit['all']
mamba run -n env pip install -y -c conda-forge libstdcxx-ng=12