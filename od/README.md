### Getting Started
#### Prerequisites
0. Ensure that [conda]() is installed on your system.
1. Create a conda environment based on the environment.yaml file by running the following command:
    ```bash
    conda env create -f environment.yaml
    ```
2. Depending on availability of GPU, you may run the following command to install the required packages:
    ```bash
    make install-gpu
    ```
    or
    ```bash
    make install-cpu
    ```
