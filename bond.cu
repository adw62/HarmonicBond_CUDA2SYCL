#include<iostream>


// Compilation Options: --use_fast_math

#define ACOS acosf
#define APPLY_PERIODIC_TO_DELTA(delta) {delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x; \
delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y; \
delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;}
#define APPLY_PERIODIC_TO_POS(pos) {pos.x -= floor(pos.x*invPeriodicBoxSize.x)*periodicBoxSize.x; \
pos.y -= floor(pos.y*invPeriodicBoxSize.y)*periodicBoxSize.y; \
pos.z -= floor(pos.z*invPeriodicBoxSize.z)*periodicBoxSize.z;}
#define APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center) {pos.x -= floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x; \
pos.y -= floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y; \
pos.z -= floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;}
#define ASIN asinf
#define ATAN atanf
#define BALLOT(var) __ballot_sync(0xffffffff, var);
#define COS cosf
#define ERF erff
#define ERFC erfcf
#define EXP expf
#define LOG logf
#define POW powf
#define RECIP 1.0f/
#define RSQRT rsqrtf
#define SHFL(var, srcLane) __shfl_sync(0xffffffff, var, srcLane);
#define SIN sinf
#define SQRT sqrtf
#define SYNC_WARPS __syncwarp();
#define TAN tanf
#define make_mixed2 make_float2
#define make_mixed3 make_float3
#define make_mixed4 make_float4
#define make_real2 make_float2
#define make_real3 make_float3
#define make_real4 make_float4

typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
typedef float mixed;
typedef float2 mixed2;
typedef float3 mixed3;
typedef float4 mixed4;
typedef unsigned int tileflags;
/**
 * This file contains CUDA definitions for the macros and functions needed for the
 * common compute framework.
 */

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define LOCAL __shared__
#define LOCAL_ARG
#define GLOBAL
#define RESTRICT __restrict__
#define LOCAL_ID threadIdx.x
#define LOCAL_SIZE blockDim.x
#define GLOBAL_ID (blockIdx.x*blockDim.x+threadIdx.x)
#define GLOBAL_SIZE (blockDim.x*gridDim.x)
#define GROUP_ID blockIdx.x
#define NUM_GROUPS gridDim.x
#define SYNC_THREADS __syncthreads();
#define MEM_FENCE __threadfence_block();
#define ATOMIC_ADD(dest, value) atomicAdd(dest, value)

typedef long long mm_long;
typedef unsigned long long mm_ulong;

#define SUPPORTS_64_BIT_ATOMICS 1
#define SUPPORTS_DOUBLE_PRECISION 1

__device__ inline long long realToFixedPoint(real x) {
    return static_cast<long long>(x * 0x100000000);
}

#define PADDED_NUM_ATOMS 32

/**
 * This file defines vector operations to simplify code elsewhere.
 */

// Versions of make_x() that take a single value and set all components to that.

inline __device__ int2 make_int2(int a) {
    return make_int2(a, a);
}

inline __device__ int3 make_int3(int a) {
    return make_int3(a, a, a);
}

inline __device__ int4 make_int4(int a) {
    return make_int4(a, a, a, a);
}

inline __device__ float2 make_float2(float a) {
    return make_float2(a, a);
}

inline __device__ float3 make_float3(float a) {
    return make_float3(a, a, a);
}

inline __device__ float4 make_float4(float a) {
    return make_float4(a, a, a, a);
}

inline __device__ double2 make_double2(double a) {
    return make_double2(a, a);
}

inline __device__ double3 make_double3(double a) {
    return make_double3(a, a, a);
}

inline __device__ double4 make_double4(double a) {
    return make_double4(a, a, a, a);
}

// Negate a vector.

inline __device__ int2 operator-(int2 a) {
    return make_int2(-a.x, -a.y);
}

inline __device__ int3 operator-(int3 a) {
    return make_int3(-a.x, -a.y, -a.z);
}

inline __device__ int4 operator-(int4 a) {
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

inline __device__ float2 operator-(float2 a) {
    return make_float2(-a.x, -a.y);
}

inline __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __device__ float4 operator-(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

inline __device__ double2 operator-(double2 a) {
    return make_double2(-a.x, -a.y);
}

inline __device__ double3 operator-(double3 a) {
    return make_double3(-a.x, -a.y, -a.z);
}

inline __device__ double4 operator-(double4 a) {
    return make_double4(-a.x, -a.y, -a.z, -a.w);
}

// Add two vectors.

inline __device__ int2 operator+(int2 a, int2 b) {
    return make_int2(a.x+b.x, a.y+b.y);
}

inline __device__ int3 operator+(int3 a, int3 b) {
    return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ int4 operator+(int4 a, int4 b) {
    return make_int4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x+b.x, a.y+b.y);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __device__ double2 operator+(double2 a, double2 b) {
    return make_double2(a.x+b.x, a.y+b.y);
}

inline __device__ double3 operator+(double3 a, double3 b) {
    return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ double4 operator+(double4 a, double4 b) {
    return make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

// Subtract two vectors.

inline __device__ int2 operator-(int2 a, int2 b) {
    return make_int2(a.x-b.x, a.y-b.y);
}

inline __device__ int3 operator-(int3 a, int3 b) {
    return make_int3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ int4 operator-(int4 a, int4 b) {
    return make_int4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x-b.x, a.y-b.y);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __device__ double2 operator-(double2 a, double2 b) {
    return make_double2(a.x-b.x, a.y-b.y);
}

inline __device__ double3 operator-(double3 a, double3 b) {
    return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ double4 operator-(double4 a, double4 b) {
    return make_double4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

// Multiply two vectors.

inline __device__ int2 operator*(int2 a, int2 b) {
    return make_int2(a.x*b.x, a.y*b.y);
}

inline __device__ int3 operator*(int3 a, int3 b) {
    return make_int3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ int4 operator*(int4 a, int4 b) {
    return make_int4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x*b.x, a.y*b.y);
}

inline __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline __device__ double2 operator*(double2 a, double2 b) {
    return make_double2(a.x*b.x, a.y*b.y);
}

inline __device__ double3 operator*(double3 a, double3 b) {
    return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ double4 operator*(double4 a, double4 b) {
    return make_double4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

// Divide two vectors.

inline __device__ int2 operator/(int2 a, int2 b) {
    return make_int2(a.x/b.x, a.y/b.y);
}

inline __device__ int3 operator/(int3 a, int3 b) {
    return make_int3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ int4 operator/(int4 a, int4 b) {
    return make_int4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

inline __device__ float2 operator/(float2 a, float2 b) {
    return make_float2(a.x/b.x, a.y/b.y);
}

inline __device__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

inline __device__ double2 operator/(double2 a, double2 b) {
    return make_double2(a.x/b.x, a.y/b.y);
}

inline __device__ double3 operator/(double3 a, double3 b) {
    return make_double3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ double4 operator/(double4 a, double4 b) {
    return make_double4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

// += operator

inline __device__ void operator+=(int2& a, int2 b) {
    a.x += b.x; a.y += b.y;
}

inline __device__ void operator+=(int3& a, int3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ void operator+=(int4& a, int4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __device__ void operator+=(float2& a, float2 b) {
    a.x += b.x; a.y += b.y;
}

inline __device__ void operator+=(float3& a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ void operator+=(float4& a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __device__ void operator+=(double2& a, double2 b) {
    a.x += b.x; a.y += b.y;
}

inline __device__ void operator+=(double3& a, double3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ void operator+=(double4& a, double4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// -= operator

inline __device__ void operator-=(int2& a, int2 b) {
    a.x -= b.x; a.y -= b.y;
}

inline __device__ void operator-=(int3& a, int3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator-=(int4& a, int4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

inline __device__ void operator-=(float2& a, float2 b) {
    a.x -= b.x; a.y -= b.y;
}

inline __device__ void operator-=(float3& a, float3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator-=(float4& a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

inline __device__ void operator-=(double2& a, double2 b) {
    a.x -= b.x; a.y -= b.y;
}

inline __device__ void operator-=(double3& a, double3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator-=(double4& a, double4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// *= operator

inline __device__ void operator*=(int2& a, int2 b) {
    a.x *= b.x; a.y *= b.y;
}

inline __device__ void operator*=(int3& a, int3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ void operator*=(int4& a, int4 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

inline __device__ void operator*=(float2& a, float2 b) {
    a.x *= b.x; a.y *= b.y;
}

inline __device__ void operator*=(float3& a, float3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ void operator*=(float4& a, float4 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

inline __device__ void operator*=(double2& a, double2 b) {
    a.x *= b.x; a.y *= b.y;
}

inline __device__ void operator*=(double3& a, double3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ void operator*=(double4& a, double4 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

// /= operator

inline __device__ void operator/=(int2& a, int2 b) {
    a.x /= b.x; a.y /= b.y;
}

inline __device__ void operator/=(int3& a, int3 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}

inline __device__ void operator/=(int4& a, int4 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}

inline __device__ void operator/=(float2& a, float2 b) {
    a.x /= b.x; a.y /= b.y;
}

inline __device__ void operator/=(float3& a, float3 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}

inline __device__ void operator/=(float4& a, float4 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}

inline __device__ void operator/=(double2& a, double2 b) {
    a.x /= b.x; a.y /= b.y;
}

inline __device__ void operator/=(double3& a, double3 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}

inline __device__ void operator/=(double4& a, double4 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}

// Multiply a vector by a constant.

inline __device__ int2 operator*(int2 a, int b) {
    return make_int2(a.x*b, a.y*b);
}

inline __device__ int3 operator*(int3 a, int b) {
    return make_int3(a.x*b, a.y*b, a.z*b);
}

inline __device__ int4 operator*(int4 a, int b) {
    return make_int4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ int2 operator*(int a, int2 b) {
    return make_int2(a*b.x, a*b.y);
}

inline __device__ int3 operator*(int a, int3 b) {
    return make_int3(a*b.x, a*b.y, a*b.z);
}

inline __device__ int4 operator*(int a, int4 b) {
    return make_int4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x*b, a.y*b);
}

inline __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ float2 operator*(float a, float2 b) {
    return make_float2(a*b.x, a*b.y);
}

inline __device__ float3 operator*(float a, float3 b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float4 operator*(float a, float4 b) {
    return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline __device__ double2 operator*(double2 a, double b) {
    return make_double2(a.x*b, a.y*b);
}

inline __device__ double3 operator*(double3 a, double b) {
    return make_double3(a.x*b, a.y*b, a.z*b);
}

inline __device__ double4 operator*(double4 a, double b) {
    return make_double4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ double2 operator*(double a, double2 b) {
    return make_double2(a*b.x, a*b.y);
}

inline __device__ double3 operator*(double a, double3 b) {
    return make_double3(a*b.x, a*b.y, a*b.z);
}

inline __device__ double4 operator*(double a, double4 b) {
    return make_double4(a*b.x, a*b.y, a*b.z, a*b.w);
}

// Divide a vector by a constant.

inline __device__ int2 operator/(int2 a, int b) {
    return make_int2(a.x/b, a.y/b);
}

inline __device__ int3 operator/(int3 a, int b) {
    return make_int3(a.x/b, a.y/b, a.z/b);
}

inline __device__ int4 operator/(int4 a, int b) {
    return make_int4(a.x/b, a.y/b, a.z/b, a.w/b);
}

inline __device__ float2 operator/(float2 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ float3 operator/(float3 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ float4 operator/(float4 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ double2 operator/(double2 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

inline __device__ double3 operator/(double3 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

inline __device__ double4 operator/(double4 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

// *= operator (multiply vector by constant)

inline __device__ void operator*=(int2& a, int b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(int3& a, int b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(int4& a, int b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

inline __device__ void operator*=(float2& a, float b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(float3& a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(float4& a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

inline __device__ void operator*=(double2& a, double b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(double3& a, double b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(double4& a, double b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

// Dot product

inline __device__ float dot(float3 a, float3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

inline __device__ double dot(double3 a, double3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

// Cross product

inline __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

inline __device__ float4 cross(float4 a, float4 b) {
    return make_float4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0f);
}

inline __device__ double3 cross(double3 a, double3 b) {
    return make_double3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

inline __device__ double4 cross(double4 a, double4 b) {
    return make_double4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0);
}

// Normalize a vector

inline __device__ float2 normalize(float2 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y);
}

inline __device__ float3 normalize(float3 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
}

inline __device__ float4 normalize(float4 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}

inline __device__ double2 normalize(double2 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y);
}

inline __device__ double3 normalize(double3 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}

inline __device__ double4 normalize(double4 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}

// Strip off the fourth component of a vector.

inline __device__ short3 trimTo3(short4 v) {
    return make_short3(v.x, v.y, v.z);
}

inline __device__ int3 trimTo3(int4 v) {
    return make_int3(v.x, v.y, v.z);
}

inline __device__ float3 trimTo3(float4 v) {
    return make_float3(v.x, v.y, v.z);
}

inline __device__ double3 trimTo3(double4 v) {
    return make_double3(v.x, v.y, v.z);
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
mixed* energyBuffer = new mixed[num_atoms]();
energyBuffer[0] = 0.0;
energyBuffer[1] = 0.0;

//make psoitions
float4* posq = new float4[num_atoms]();
posq[0] = {0.0,0.0,0.0,0.0};
posq[1] = {5.0,5.0,5.0,0.0};

//make bond params, (r0, k)
float2* params = new float2[num_atoms]();
params[0] = {1, 2};
params[1] = {1, 0};

mixed energy = 0;
if ((groups&1) != 0)
for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < 2; index += blockDim.x*gridDim.x) {
    uint2 atoms0 = atomIndices0_0[index];
    unsigned int atom1 = atoms0.x;
    real4 pos1 = posq[atom1];
    unsigned int atom2 = atoms0.y;
    real4 pos2 = posq[atom2];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if 0
APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r = SQRT(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
float2 bondParams = params[index];
real deltaIdeal = r-bondParams.x;
energy += 0.5f * bondParams.y*deltaIdeal*deltaIdeal;
real dEdR = bondParams.y * deltaIdeal;

dEdR = (r > 0) ? (dEdR / r) : 0;
delta *= dEdR;
real3 force1 = delta;
real3 force2 = -delta;
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

