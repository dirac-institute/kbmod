#!/bin/sh
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
