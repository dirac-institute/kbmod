# Bare bones testing framework
.PHONY: all test

all : tests/hello.cpp
#	g++ -I/Users/Bryce/lsst/DarwinX86/boost/1.60.lsst1+1/include/ -L/Users/Bryce/lsst/DarwinX86/boost/1.60.lsst1+1/lib -otests/hello -lboost_unit_test_framework tests/hello.cpp
	g++ -otests/hello -lboost_unit_test_framework tests/hello.cpp

test : all
	./tests/hello