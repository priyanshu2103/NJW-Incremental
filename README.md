# NJW-Incremental
Incremental implementation of NJW algorithm presented in paper "On Spectral Clustering: Analysis and an algorithm"

Steps to run the algorithm: 
1. `sudo apt-get install libeigen3-dev`
2. `g++ njw.cpp -O3`
3. `./a.out <input.txt> <no. of clusters>` (Ex: `./a.out ip_spiral.txt 3`)

For visualization of datasets and clusters, refer to https://colab.research.google.com/drive/1RhV9VIv1U6HJ70YfSHOIcnemElSbnVaU?usp=sharing
