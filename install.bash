#!/bin/sh
echo "Installing... Be aware of your current python environment"
git submodule init
git submodule update
cd search/pybinds/

if hash python 2>/dev/null; then
    if python -c 'import sys; print(sys.version_info[0])' | grep -q 3; then
        echo "Python version 3 confirmed"
    else
        echo "Python detected, but it must be version 3.x";
        exit 1;
    fi
else
    echo "Python not found"
    exit 1;
fi


# Check for cmake version 3 under "cmake" and "cmake3"
if hash cmake3 2>/dev/null; then
    cmake3 ./
elif hash cmake 2>/dev/null; then
    cmake ./
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
