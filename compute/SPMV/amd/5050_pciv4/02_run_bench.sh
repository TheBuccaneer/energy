#!/bin/bash

nvcc -O3 -std=c++17 -lcusparse -lnvidia-ml -o main main.cu
./main
