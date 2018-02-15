#!/bin/sh
echo "Installing... Be aware of your current python environment"
git submodule init
git submodule update
cd search/pybinds/
cmake ./
make
cd ../../
source setup.bash
cd tests/
./run_tests.sh
cd ../
