export PYTHONPATH=$PWD/analysis:$PWD/search/pybinds:$PYTHONPATH

# Disable python multiprocessing
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1

# Set up pyOrbfit for Orbit Fitting
export PYORBFIT_HOME=/astro/store/epyc/users/brycek/dirac-pyOrbfit

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYORBFIT_HOME
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$PYORBFIT_HOME
export PATH=$PATH:$PYORBFIT_HOME
export PYTHONPATH=$PYTHONPATH:$PYORBFIT_HOME

export ORBIT_EPHEMERIS=$PYORBFIT_HOME/binEphem.423
export ORBIT_OBSERVATORIES=$PYORBFIT_HOME/observatories.dat
