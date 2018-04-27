#!/bin/sh
echo "Installing... Be aware of your current python environment"
git submodule init
git submodule update
cd search/pybinds/

# Check for cmake version 3 under "cmake" and "cmake3"
if hash cmake 2>/dev/null; then
    cmake ./
elif hash cmake3 2>/dev/null; then
    cmake3 ./
else
    echo >&2 "Cannot find cmake. Aborting..."
    exit 1
fi
make
cd ../../
source setup.bash
cd tests/
./run_tests.sh
cd ../
