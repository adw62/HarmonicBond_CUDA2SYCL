# HarmonicBond_CUDA2SYCL
This repo contains CUDA code extracted from OpenMM (https://github.com/openmm/) for computing harmonic bonds and the SYCL equivalent of this code written using DPC++ Compatibility Tool. 

The CUDA code can be compiled with: `nvcc bond.cu -o bond`

The SYCL code can be complied with: `dpcpp bond.cpp -o bond`

These can be timed by running: `python time.py` 

Running the CUDA code on Nvidia M1000M and SYCL code on Intel DevCloud targeting DG1(XeMax) yeilds the following results:

![OpenMM](https://user-images.githubusercontent.com/38112687/159527297-28096dff-7a52-43ac-8731-e18680d913a8.png)
