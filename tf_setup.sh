#!/bin/bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
cp -rf ${CONDA_PREFIX}/nvvm ${CONDA_PREFIX}/lib/ || echo "nvvm folder already exists or not found"
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
