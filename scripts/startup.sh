source ~/.bashrc

conda activate indiProj3
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=/usr/local/bin/singularity:$PATH
export PYTHONPATH=~/DEHB:$PYTHONPATH
