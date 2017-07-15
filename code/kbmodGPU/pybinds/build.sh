c++ -O3 -shared -std=c++11 -fPIC -I ../include/pybind11/ -I ../include/ -L /home/kbmod-usr/anaconda2/lib/python2.7/config/ -L ../lib/ -o kbmod.so psfBind.cpp `python-config --cflags --ldflags`


