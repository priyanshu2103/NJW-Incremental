# NJW-Incremental
Incremental implementation of NJW algorithm presented in paper "On Spectral Clustering: Analysis and an algorithm"
The final code is available in `njw_final.cpp`.

Steps to run the code: 

Install eigen3 library `sudo apt-get install libeigen3-dev`

To compile: `g++ njw_final.cpp -O3`

For updation: `./a.out <input.txt> <no. of clusters> <order of approximation(currently only order 1 supported)> update <index of point to update(0-indexed)> <point to be updated>`(e.g. update point at index 5 to (6,16) -> `./a.out ip_jain.txt 2 1 update 5 6 16`)

For insertion: `./a.out <input.txt> <no. of clusters> <order of approximation(currently only order 1 supported)> insert <point to be inserted>` (e.g add point (6,16) -> `./a.out ip_jain.txt 2 1 insert 6 16`)

For deletion: `./a.out <input.txt> <no. of clusters> <order of approximation(currently only order 1 supported)> remove <index of point to delete(0-indexed)>` (e.g. remove point at index 5 -> `./a.out ip_jain.txt 2 1 remove 5`)

For visualization of datasets and clusters, refer to `Final_submission.ipynb`

Datasets used are available at http://cs.uef.fi/sipu/datasets/
