// v4.2
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "copy.h"
#include "iterate.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

#define DEVICE_ID 1
#define ITERATIONS 3500000

using namespace std;
using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;


cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, int res);

int main()
{
    const unsigned int res = 256;
    const unsigned int size = res * res;
    float* u = new float[size];
    float* v = new float[size];
    float* u_sol = new float[size];

    text_to_array(u, v);

    // Add vectors in parallel.
    cudaError_t cudaStatus = iterate_prep(u, v, u_sol, size, res);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    array_to_text(u, res);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete[] u;
    delete[] u_sol;
    delete[] v;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, int res)
{
    //const unsigned int res = 128;
    float* dev_ua = 0;
    float* dev_ub = 0;
    float* dev_v = 0;
    float* dev_2_u = 0;
    float* u_2 = 0;

    //measure at this iteration
    int iteration_limit_per_2 = ITERATIONS / 2;
    int measure_at = iteration_limit_per_2 - 1;

    // cooperative kernel launch parameters
    void* kernelArgs[] = { (void*)&dev_ua, (void*)&dev_ub, (void*)&dev_v, (void*)&res, (void*)&iteration_limit_per_2 };
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(8, 8, 1);

    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    int average_iteration_runtime;

    // declare persisting L2 cache variables
    cudaDeviceProp prop;
    cudaStream_t stream_1;
    size_t L2_size;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(DEVICE_ID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&dev_2_u, 2 * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!0");
        goto Error;
    }

    // splitting the allcated memory for u
    dev_ua = dev_2_u;
    dev_ub = dev_2_u + size;

    cudaStatus = cudaMalloc((void**)&dev_v, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!2");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ua, u, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!3");

        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ub, u, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!4");

        goto Error;
    }


    cudaStatus = cudaMemcpy(dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!5");
        goto Error;
    }

    // ----- Set up persisting cache -----

    /*
    // set persisting cache maximum limit
    cudaGetDeviceProperties(&prop, DEVICE_ID);
    L2_size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, L2_size);
    // setting stream parameters
    cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
    cudaStreamCreate(&stream_1);
    stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(dev_2_u); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes =  2 * res * res * sizeof(float);                    // Number of bytes for persistence access.
    // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    stream_attribute.accessPolicyWindow.hitRatio = 1;                          // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    //Set the attributes to a CUDA stream of type cudaStream_t
    cudaStreamSetAttribute(stream_1, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    */
    cudaDeviceSynchronize();
    // ---------- Launch a kernel on the GPU with one thread for each element. ----------
    begin = std::chrono::steady_clock::now();

    cudaLaunchCooperativeKernel((void*)iterateKernel, dimGrid, dimBlock, kernelArgs);
    cudaDeviceSynchronize();

    end = std::chrono::steady_clock::now();
    std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[us]" << endl;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[ns]" << endl;

    // destroy stream
    //cudaStreamDestroy(stream_1);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaFree(dev_ua);
    cudaFree(dev_ub);
    cudaFree(dev_v);

Error:
    cudaFree(dev_ua);
    cudaFree(dev_ub);
    cudaFree(dev_v);
    //cudaFree(dev_u_sol);

    return cudaStatus;
}

