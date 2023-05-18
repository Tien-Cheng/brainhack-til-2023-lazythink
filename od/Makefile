.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Name of the Conda environment
CONDA_ENV_NAME := brainhack-od

# Create or update the Conda environment
install-cpu:
	conda env update --name $(CONDA_ENV_NAME) --file environment.yaml --prune
	conda install pytorch torchvision cpuonly -c pytorch
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) ; bash scripts/install-mmdet.sh

install-gpu:
	conda env update --name $(CONDA_ENV_NAME) --file environment.yaml --prune
	conda install pytorch torchvision -c pytorch
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) ; bash scripts/install-mmdet.sh

# Default target: create or update the environment and activate it
default: install-cpu
