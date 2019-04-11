export PYTHONPATH=$PWD/analysis:$PWD/search/pybinds:$PYTHONPATH
# Disable python multiprocessing
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS="N"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
#setup pyephem
