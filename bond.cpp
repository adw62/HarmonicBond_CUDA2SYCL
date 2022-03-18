#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

#define KERNEL extern "C"
#define PADDED_NUM_ATOMS 32

inline sycl::float3 make_float3(float a) {
    return sycl::float3(a, a, a);
}

/*
DPCT1011:0: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 1.2.1
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float3 &a, float b) {
    a.x() *= b; a.y() *= b; a.z() *= b;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:1: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 1.2.1
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float3 &a, sycl::float3 b) {
    a.x() *= b.x(); a.y() *= b.y(); a.z() *= b.z();
}
} // namespace dpct_operator_overloading

/*
DPCT1011:2: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 1.2.1
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float3 &a, sycl::float3 b) {
    a.x() += b.x(); a.y() += b.y(); a.z() += b.z();
}
} // namespace dpct_operator_overloading

/*
DPCT1011:3: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 1.2.1
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator-(sycl::float3 a) {
    return sycl::float3(-a.x(), -a.y(), -a.z());
}
} // namespace dpct_operator_overloading

inline long long realToFixedPoint(float x) {
    return static_cast<long long>(x * 0x100000000);
}

extern "C" void computeBondedForces(
    unsigned long long *__restrict__ forceBuffer,
    float *__restrict__ energyBuffer, const sycl::float4 *__restrict__ posq,
    const sycl::uint2 *__restrict__ atomIndices0_0, sycl::float2 *params,
    int num_atoms, sycl::nd_item<3> item_ct1, const sycl::stream &stream_ct1) {

float energy = 0;
for (unsigned int index =
         item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
         item_ct1.get_local_id(2);
     index < num_atoms;
     index += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    sycl::uint2 atoms0 = atomIndices0_0[index];
    unsigned int atom1 = atoms0.x();
    sycl::float4 pos1 = posq[atom1];
    unsigned int atom2 = atoms0.y();
    sycl::float4 pos2 = posq[atom2];
sycl::float3 delta =
    sycl::float3(pos2.x() - pos1.x(), pos2.y() - pos1.y(), pos2.z() - pos1.z());
#if 0
APPLY_PERIODIC_TO_DELTA(delta)
#endif
float r = sycl::sqrt(delta.x() * delta.x() + delta.y() * delta.y() +
                     delta.z() * delta.z());
sycl::float2 bondParams = params[index];
float deltaIdeal = r - bondParams.x();
energy += 0.5f * bondParams.y() * deltaIdeal * deltaIdeal;
float dEdR = bondParams.y() * deltaIdeal;

dEdR = (r > 0) ? (dEdR / r) : 0;
dpct_operator_overloading::operator*=(delta, dEdR);
sycl::float3 force1 = delta;
sycl::float3 force2 = dpct_operator_overloading::operator-(delta);

    /*
    DPCT1039:6: The generated code assumes that "&forceBuffer[atom1]" points to
    the global memory address space. If it points to a local memory address
    space, replace "sycl::global_ptr" with "sycl::local_ptr".
    */
    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(&forceBuffer[atom1]))
        .fetch_add(
            static_cast<unsigned long long>(realToFixedPoint(force1.x())));
    /*
    DPCT1039:7: The generated code assumes that
    "&forceBuffer[atom1+PADDED_NUM_ATOMS]" points to the global memory address
    space. If it points to a local memory address space, replace
    "sycl::global_ptr" with "sycl::local_ptr".
    */
    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(
            &forceBuffer[atom1 + PADDED_NUM_ATOMS]))
        .fetch_add(
            static_cast<unsigned long long>(realToFixedPoint(force1.y())));
    /*
    DPCT1039:8: The generated code assumes that
    "&forceBuffer[atom1+PADDED_NUM_ATOMS*2]" points to the global memory address
    space. If it points to a local memory address space, replace
    "sycl::global_ptr" with "sycl::local_ptr".
    */
    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(
            &forceBuffer[atom1 + PADDED_NUM_ATOMS * 2]))
        .fetch_add(
            static_cast<unsigned long long>(realToFixedPoint(force1.z())));
    /*
    DPCT1078:9: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::ext::oneapi::atomic_fence(
        sycl::ext::oneapi::memory_order::acq_rel,
        sycl::ext::oneapi::memory_scope::work_group);
    /*
    DPCT1039:10: The generated code assumes that "&forceBuffer[atom2]" points to
    the global memory address space. If it points to a local memory address
    space, replace "sycl::global_ptr" with "sycl::local_ptr".
    */
    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(&forceBuffer[atom2]))
        .fetch_add(
            static_cast<unsigned long long>(realToFixedPoint(force2.x())));
    /*
    DPCT1039:11: The generated code assumes that
    "&forceBuffer[atom2+PADDED_NUM_ATOMS]" points to the global memory address
    space. If it points to a local memory address space, replace
    "sycl::global_ptr" with "sycl::local_ptr".
    */
    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(
            &forceBuffer[atom2 + PADDED_NUM_ATOMS]))
        .fetch_add(
            static_cast<unsigned long long>(realToFixedPoint(force2.y())));
    /*
    DPCT1039:12: The generated code assumes that
    "&forceBuffer[atom2+PADDED_NUM_ATOMS*2]" points to the global memory address
    space. If it points to a local memory address space, replace
    "sycl::global_ptr" with "sycl::local_ptr".
    */
    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(
            &forceBuffer[atom2 + PADDED_NUM_ATOMS * 2]))
        .fetch_add(
            static_cast<unsigned long long>(realToFixedPoint(force2.z())));
    /*
    DPCT1078:13: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::ext::oneapi::atomic_fence(
        sycl::ext::oneapi::memory_order::acq_rel,
        sycl::ext::oneapi::memory_scope::work_group);
}
energyBuffer[item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
             item_ct1.get_local_id(2)] += energy;
}

int main(int argc, char **argv) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

if (argc != 2) {
    printf("Usage: %s <NINT>\n", argv[0]);
    exit(1);
}
int num_atoms = atoi(argv[1]);
//int num_atoms = 10000;

//make idxs
sycl::uint2 *atomIndices0_0 = new sycl::uint2[num_atoms]();
sycl::uint2 *atomIndices0_0_h = new sycl::uint2[num_atoms]();
for (unsigned int i = 0; i < num_atoms; i++) {
  atomIndices0_0_h[i] = {0, i};
}

atomIndices0_0 = sycl::malloc_device<sycl::uint2>(num_atoms, q_ct1);
q_ct1.memcpy(atomIndices0_0, atomIndices0_0_h, num_atoms * sizeof(sycl::uint2))
    .wait();

//make force running total
unsigned long long* forceBuffer = new unsigned long long[num_atoms]();
unsigned long long* forceBuffer_h = new unsigned long long[num_atoms]();
for (int i = 0; i < num_atoms; i++) {
  forceBuffer_h[i] = 0.0;
}

forceBuffer = sycl::malloc_device<unsigned long long>(num_atoms, q_ct1);
q_ct1.memcpy(forceBuffer, forceBuffer_h, num_atoms * sizeof(unsigned long long))
    .wait();

//make energy running total
float* energyBuffer = new float[num_atoms]();
float* energyBuffer_h = new float[num_atoms]();
for (int i = 0; i < num_atoms; i++) {
  energyBuffer_h[i] = 0.0;
}

energyBuffer = sycl::malloc_device<float>(num_atoms, q_ct1);
q_ct1.memcpy(energyBuffer, energyBuffer_h, num_atoms * sizeof(float)).wait();

//make positions
sycl::float4 *posq = new sycl::float4[num_atoms]();
sycl::float4 *posq_h = new sycl::float4[num_atoms]();
posq_h[0] = {0.0,0.0,0.0,0.0};
for (int i = 1; i < num_atoms; i++) {
  posq_h[i] = {5.0,5.0,5.0,0.0};
}

posq = sycl::malloc_device<sycl::float4>(num_atoms, q_ct1);
q_ct1.memcpy(posq, posq_h, num_atoms * sizeof(sycl::float4)).wait();

//make bond params, (r0, k)
sycl::float2 *params = new sycl::float2[num_atoms]();
sycl::float2 *params_h = new sycl::float2[num_atoms]();
params_h[0] = {1, 0};
for (int i = 1; i < num_atoms; i++) {
  params_h[i] = {1, 2};
}

params = sycl::malloc_device<sycl::float2>(num_atoms, q_ct1);
q_ct1.memcpy(params, params_h, num_atoms * sizeof(sycl::float2)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_atoms / 200) *
                                           sycl::range<3>(1, 1, 200),
                                       sycl::range<3>(1, 1, 200)),
                     [=](sycl::nd_item<3> item_ct1) {
                       computeBondedForces(forceBuffer, energyBuffer, posq,
                                           atomIndices0_0, params, num_atoms,
                                           item_ct1, stream_ct1);
                     });
  });

/*
DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
int cudaerr = (dev_ct1.queues_wait_and_throw(), 0);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}