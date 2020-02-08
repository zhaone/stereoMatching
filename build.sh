#!/bin/bash

# for eval
chmod a+x ./run

# build
mkdir build
cd ./build
cmake ..
make