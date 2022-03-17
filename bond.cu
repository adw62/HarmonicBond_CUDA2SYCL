#include<iostream>

#define KERNEL extern "C" __global__
#define PADDED_NUM_ATOMS 32

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

extern "C" __global__ void computeBondedForces() {

int groups = 1;
int num_atoms = 2;

//make idxs
uint2* atomIndices0_0 = new uint2[num_atoms]();
atomIndices0_0[0] = {0, 1};
atomIndices0_0[1] = {1, 0};

//make force running total
unsigned long long* forceBuffer = new unsigned long long[num_atoms]();
forceBuffer[0] = 0.0;
forceBuffer[1] = 0.0;

//make energy running total
float* energyBuffer = new float[num_atoms]();
energyBuffer[0] = 0.0;
energyBuffer[1] = 0.0;

//make positions
float4* posq = new float4[num_atoms]();
posq[0] = {0.0,0.0,0.0,0.0};
posq[1] = {5.0,5.0,5.0,0.0};

//make bond params, (r0, k)
float2* params = new float2[num_atoms]();
params[0] = {1, 2};
params[1] = {1, 0};

float energy = 0;
if ((groups&1) != 0)
for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < 2; index += blockDim.x*gridDim.x) {
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
printf("%6.4lf\n", r);

    atomicAdd(&forceBuffer[atom1], static_cast<unsigned long long>(realToFixedPoint(force1.x)));
    atomicAdd(&forceBuffer[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>(realToFixedPoint(force1.y)));
    atomicAdd(&forceBuffer[atom1+PADDED_NUM_ATOMS*2], static_cast<unsigned long long>(realToFixedPoint(force1.z)));
    __threadfence_block();
    atomicAdd(&forceBuffer[atom2], static_cast<unsigned long long>(realToFixedPoint(force2.x)));
    atomicAdd(&forceBuffer[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>(realToFixedPoint(force2.y)));
    atomicAdd(&forceBuffer[atom2+PADDED_NUM_ATOMS*2], static_cast<unsigned long long>(realToFixedPoint(force2.z)));
    __threadfence_block();
}
energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
printf("%6.4lf\n", energy);
}


int main() {

computeBondedForces<<<1, 2>>>();

cudaError_t cudaerr = cudaDeviceSynchronize();
if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
        cudaGetErrorString(cudaerr));
}

