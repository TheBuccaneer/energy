#!/bin/bash

nvcc -O3 -std=c++17 -lnvidia-ml main.cu -o main
./main