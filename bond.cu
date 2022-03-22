#include<iostream>
#include <cassert>  

#define KERNEL extern "C" __global__

inline __device__ float3 make_float3(float a) {
    return make_float3(a, a, a);
}

inline __device__ void operator*=(float3& a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(float3& a, float3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ void operator+=(float3& a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ inline long long realToFixedPoint(float x) {
    return static_cast<long long>(x * 0x100000000);
}

extern "C" __global__ void computeBondedForces(unsigned long long* __restrict__ forceBuffer,
 float* __restrict__ energyBuffer, const float4* __restrict__ posq,
  const uint2* __restrict__ atomIndices0_0, float2* params, int num_atoms) {
float energy = 0;
for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < num_atoms-1; index += blockDim.x*gridDim.x) {
    uint2 atoms0 = atomIndices0_0[index];
    unsigned int atom1 = atoms0.x;
    float4 pos1 = posq[atom1];
    unsigned int atom2 = atoms0.y;
    float4 pos2 = posq[atom2];
float3 delta = make_float3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if 0
APPLY_PERIODIC_TO_DELTA(delta)
#endif
float r = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
float2 bondParams = params[index];
float deltaIdeal = r-bondParams.x;
energy += 0.5f * bondParams.y*deltaIdeal*deltaIdeal;
float dEdR = bondParams.y * deltaIdeal;

dEdR = (r > 0) ? (dEdR / r) : 0;
delta *= dEdR;
float3 force1 = delta;
float3 force2 = -delta;

    atomicAdd(&forceBuffer[atom1], static_cast<unsigned long long>(realToFixedPoint(force1.x)));
    atomicAdd(&forceBuffer[atom1+num_atoms], static_cast<unsigned long long>(realToFixedPoint(force1.y)));
    atomicAdd(&forceBuffer[atom1+num_atoms*2], static_cast<unsigned long long>(realToFixedPoint(force1.z)));
    __threadfence_block();
    atomicAdd(&forceBuffer[atom2], static_cast<unsigned long long>(realToFixedPoint(force2.x)));
    atomicAdd(&forceBuffer[atom2+num_atoms], static_cast<unsigned long long>(realToFixedPoint(force2.y)));
    atomicAdd(&forceBuffer[atom2+num_atoms*2], static_cast<unsigned long long>(realToFixedPoint(force2.z)));
    __threadfence_block();
}
energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
//printf("%6.4lf\n", energy);
}


int main(int argc, char** argv) {

if (argc != 2) {
    printf("Usage: %s <NINT>\n", argv[0]);
    exit(1);
}

int num_atoms = (atoi(argv[1])*200)+1;
int num_bonds = num_atoms-1;

//x, y, z for atoms which have a spring connected to 0,0,0
float p = 5.0;
//spring constant for all springs
float k = 2;
//equilibrium separation for all springs
float r0 = 1;

//make bond idxs (these ids denote what atoms have springs between them)
uint2* atomIndices0_0 = new uint2[num_bonds]();
uint2* atomIndices0_0_h = new uint2[num_bonds]();
for (unsigned int i = 0; i < num_bonds; i++) {
  atomIndices0_0_h[i] = {0, i+1};
}

cudaMalloc((void **) &atomIndices0_0, num_bonds*sizeof(uint2));
cudaMemcpy(atomIndices0_0, atomIndices0_0_h, num_bonds*sizeof(uint2), cudaMemcpyHostToDevice);

//make bond params, (r0, k)
float2* params = new float2[num_bonds]();
float2* params_h = new float2[num_bonds]();
for (int i = 0; i < num_bonds; i++) {
  params_h[i] = {r0, k};
}

cudaMalloc((void **) &params, num_bonds*sizeof(float2));
cudaMemcpy(params, params_h, num_bonds*sizeof(float2), cudaMemcpyHostToDevice);

//make energy running total
float* energyBuffer = new float[num_bonds]();
float* energyBuffer_h = new float[num_bonds]();
for (int i = 0; i < num_bonds; i++) {
  energyBuffer_h[i] = 0.0;
}

cudaMalloc((void **) &energyBuffer, num_bonds*sizeof(float));
cudaMemcpy(energyBuffer, energyBuffer_h, num_bonds*sizeof(float), cudaMemcpyHostToDevice);

//make force running total
unsigned long long* forceBuffer = new unsigned long long[num_atoms*3]();
unsigned long long* forceBuffer_h = new unsigned long long[num_atoms*3]();
for (int i = 0; i < num_atoms*3; i++) {
  forceBuffer_h[i] = 0.0;
}

cudaMalloc((void **) &forceBuffer, 3*num_atoms*sizeof(unsigned long long));
cudaMemcpy(forceBuffer, forceBuffer_h, 3*num_atoms*sizeof(unsigned long long), cudaMemcpyHostToDevice);

//make positions (Place one atom at 0,0,0 and the rest at some user defined x,y,z)
float4* posq = new float4[num_atoms]();
float4* posq_h = new float4[num_atoms]();
posq_h[0] = {0.0,0.0,0.0,0.0};
for (int i = 1; i < num_atoms; i++) {
  posq_h[i] = {p,p,p,0.0};
}

cudaMalloc((void **) &posq, num_atoms*sizeof(float4));
cudaMemcpy(posq, posq_h, num_atoms*sizeof(float4), cudaMemcpyHostToDevice);

computeBondedForces<<<num_bonds/200, 200>>>(forceBuffer, energyBuffer, posq, atomIndices0_0, params, num_atoms);


cudaMemcpy(energyBuffer_h, energyBuffer, num_bonds*sizeof(float), cudaMemcpyDeviceToHost);
for (int i = 0; i < num_bonds; i++) {
    //0.5*k*(sqrt(3*p**2)-r0)**2
    //0.5*2*(sqrt(75)-1)**2
    assert(abs(energyBuffer_h[i]-58.6795) <= 0.0001);
    //printf("%6.4lf\n", energyBuffer_h[i]); 
}

cudaError_t cudaerr = cudaDeviceSynchronize();
if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
        cudaGetErrorString(cudaerr));
}

