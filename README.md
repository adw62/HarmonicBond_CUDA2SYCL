# HarmonicBond_CUDA2SYCL
This repo contains CUDA code extracted from OpenMM (https://github.com/openmm/) for computing harmonic bonds and the SYCL equivalent of this code written using DPC++ Compatibility Tool. 

The CUDA code can be compiled with: `nvcc bond.cu -o bond`

The SYCL code can be complied with: `dpcpp bond.cpp -o bond`

These can be timed by running: `python time.py` 

Running the CUDA code on Nvidia M1000M and SYCL code on Intel DevCloud targeting XeMax yeilds the following results:

![openmm2](https://user-images.githubusercontent.com/38112687/159540167-69e85cb0-beba-492b-9090-3f534a80dea7.png)

