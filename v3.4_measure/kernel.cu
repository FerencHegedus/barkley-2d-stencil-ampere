//3.4_measure
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
//measuring time
#include <chrono>

#define DEVICE_ID 1
#define ITERATIONS 3500000

__const__ float repsilon = 1 / 0.02;	// Inkább: repsilon = 1.0/0.02
__const__ float ra = 1 / 0.64;		// Inkább: ra = 1.0/0.64
__const__ float b = 0.05;
__const__ float dt = 0.003051800000000;

float dx = 0.195312500000000;
//float dt = 0.003051800000000;
// float ccc = dt / 6 / dx / dx;
// const float ccc = 0.0133335171;
__const__ float ccc = 0.013333517126739025115966796875000000000000000000000000000000;

using namespace std;

cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, unsigned int save_n_stepps, int res);
void text_to_array(float* u, float* v);
void array_to_text(float* u_sol, int res);


// copy from global memory to the registers
static __device__ __forceinline__ void copy_g_to_r(float* g_ua, float* g_v, float* r_ua, float* r_v, int ThreadOffset)
{
    *r_ua = g_ua[ThreadOffset];
    *r_v = g_v[ThreadOffset];
}

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(float* r_ua, float* r_v, float* r_un, float* r_vn)
{
    float u_th;

    // ----- constants -----
    // Ezeket vagy a constant vagy a shared memóriába raknám.
    // Constant memóriába úgy is be tudod "rakni" ha beégeted a számot ahol használod.
    // Ha változtatni akarod, akkor lehet #define -t is használni (a global függvényen kívül természetesen).
    // DE NEM KRITIKUS
    /*
    const float epsilon = 0.02;	// Inkább: repsilon = 1.0/0.02
    const float a = 0.64;		// Inkább: ra = 1.0/0.64
    */


    // Az osztásokat kicserélnémreciprokkal való szorzásra! Lásd a kommenteket fent.
    // Az osztások nagyon lassúak, kb 50-70 szer lassabb mint egy szorzás vagy összeadás/kivonás
    u_th = ((*r_v) + b) * ra; // ( r_v + b ) * ra // ez a sor összevonható lenne a +2. sorral a +2. sorban
    *r_vn = (*r_v) + dt * ((*r_ua) - (*r_v));
    *r_un = (*r_ua) + dt * repsilon * (*r_ua) * (1 - (*r_ua)) * ((*r_ua) - u_th); // ... dt * repsilon * r_ua ...
}

// deal with laplace
static __device__ __forceinline__ void laplace(float* g_ua, float* r_un, int res, int i, int j, int ThreadOffset)
{
    if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
    {

        float r_L = 0;
        // Nem nagy a stencil, kilehet bontani teljesen:
        float* g_ua_act = g_ua + ThreadOffset; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD
        r_L += -20 * g_ua_act[0];
        r_L += 4 * g_ua_act[+1];
        r_L += 4 * g_ua_act[-1];

        g_ua_act = g_ua_act - res; // Eltoljuk a pointert 1 sorral feljebb, 1 INT ADD
        r_L += 4 * g_ua_act[0];
        r_L += 1 * g_ua_act[+1];
        r_L += 1 * g_ua_act[-1];

        g_ua_act = g_ua_act + 2 * res; // Eltoljuk a pointert 2 sorral lejjebb, 1 INT FMA
        r_L += 4 * g_ua_act[0];
        r_L += 1 * g_ua_act[+1];
        r_L += 1 * g_ua_act[-1];

        r_L = ccc * r_L; // Ez nem tudom miért kell ide de ha kell, akkor kell :)
        // Ebben az esetben a d_coeff nem kell. A beégetett számok a constant memóriába kerülnek automatikusan.

        //Műveletek összesen: //ezt módosítottam, változhatott az arány, idk
        // 9 float FMA, 1 float MUL
        // 2 int FMA, 2 int ADD
        // Arány: 10/4

        *r_un = (*r_un) + r_L;
    }
}

// copy from registers to global memory
static __device__ __forceinline__ void copy_r_to_g(float* g_ub, float* g_v, float r_un, float r_vn, int ThreadOffset)
{
    // Integer spórolás ha már van ThreadOffset
    g_ub[ThreadOffset] = r_un;
    g_v[ThreadOffset] = r_vn;
}

// no flux boundary condition
static __device__ __forceinline__ void no_flux(float* g_ub, int res, int i, int j, int ThreadOffset)
{
    // ----- No Flux BC -----

    // Hasonló módon az integer spórólások.
    // Továbbá nem kell az else, mert az egyértelmű. Ha i==0, akkor nem lehet i==res-1
    // Egy probléma van, ez a sarkok kezelése. Lehet azokat külön kéne venni. Az is lehet hogy nem számít honna vesszük a sarokról az adatot.
    // Alkalmaztam üres mezőket, hogy átláthatóbb legyen a dolog.
    if (i == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + 1]; }
    //__syncthreads();
    if (i == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - 1]; }
    __syncthreads();
    if (j == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + res]; }
    //__syncthreads();
    if (j == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - res]; }
}

__global__ void iterateKernel(float* g_ua, float* g_ub, float* g_v, int res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < res && j < res)
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;

        // ----- Load Data -----
        copy_g_to_r(g_ua, g_v, &r_ua, &r_v, ThreadOffset);
        __syncthreads();

        // ----- Iteration step -----
        iterate_without_laplace(&r_ua, &r_v, &r_un, &r_vn);
        // __syncthreads();

        // ----- Laplace -----
        laplace(g_ua, &r_un, res, i, j, ThreadOffset);
        // __syncthreads();

        // ----- "Unload" Data -----
        copy_r_to_g(g_ub, g_v, r_un, r_vn, ThreadOffset);
        // __syncthreads();
    }
}

__global__ void iterateKernelExp(float* g_ua, float* g_ub, float* g_v, int res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float u_th;

    // ----- constants -----
    const float repsilon = 1 / 0.02;
    const float ra = 1 / 0.64;	
    const float b = 0.05;
    const float dt = 0.003051800000000;

    if (i < res && j < res)
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;
        float r_L;
        const float dx = 0.195312500000000;
        const float dt = 0.003051800000000;
        // float ccc = dt / 6 / dx / dx;
        //const float ccc = 0.0133335171;
        const float ccc = 0.013333517126739025115966796875000000000000000000000000000000;

        // ----- Load Data -----
        //copy_g_to_r(g_ua, g_v, &r_ua, &r_v, ThreadOffset);
        r_ua = g_ua[ThreadOffset];
        r_v = g_v[ThreadOffset];    
        __syncthreads();
        // ----- Iteration step -----
        //iterate_without_laplace(&r_ua, &r_v, &r_un, &r_vn);
        u_th = (r_v + b) * ra;
        r_vn = r_v + dt * (r_ua - r_v);
        r_un = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th);
        __syncthreads();
        // ----- Laplace -----
        //laplace(g_ua, &r_un, res, i, j, ThreadOffset);
        if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
        {
            r_L = 0;
            // Nem nagy a stencil, kilehet bontani teljesen:
            float* g_ua_act = g_ua + ThreadOffset; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD
            r_L += -20 * g_ua_act[0];
            r_L += 4 * g_ua_act[+1];
            r_L += 4 * g_ua_act[-1];

            g_ua_act = g_ua_act - res; // Eltoljuk a pointert 1 sorral feljebb, 1 INT ADD
            r_L += 4 * g_ua_act[0];
            r_L += 1 * g_ua_act[+1];
            r_L += 1 * g_ua_act[-1];

            g_ua_act = g_ua_act + 2 * res; // Eltoljuk a pointert 2 sorral lejjebb, 1 INT FMA
            r_L += 4 * g_ua_act[0];
            r_L += 1 * g_ua_act[+1];
            r_L += 1 * g_ua_act[-1];

            r_L = ccc * r_L; 

            r_un += r_L;
        }
        __syncthreads();
        // ----- "Unload" Data -----
        //copy_r_to_g(g_ub, g_v, r_un, r_vn, ThreadOffset);
        g_ub[ThreadOffset] = r_un;
        g_v[ThreadOffset] = r_vn;
        __syncthreads();
    }
}

__global__ void no_flux_kernel(float* g_ub, int res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < res && j < res)
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)

        // ----- No Flux Boundary Condition -----
        no_flux(g_ub, res, i, j, ThreadOffset);
    }
}

int main()
{
    const unsigned int res = 256;
    const unsigned int s = res * res;
    const unsigned int save_n_stepps = 200 * 2;
    float* u = new float[s];
    float* v = new float[s];
    float* u_sol = new float[s * save_n_stepps];

    text_to_array(u, v);

    // Add vectors in parallel.
    cudaError_t cudaStatus = iterate_prep(u, v, u_sol, s, save_n_stepps, res);
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
cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, unsigned int save_n_stepps, int res)
{
    //const unsigned int res = 128;
    float* dev_ua = 0;
    float* dev_ub = 0;
    float* dev_v = 0;
    dim3 dimGrid(8, 8, 1);
    dim3 dimBlock(32, 32, 1);

    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
	int average_iteration_runtime;
    //measure at this iteration
    


    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(DEVICE_ID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&dev_ua, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!0");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ub, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!1");
        goto Error;
    }

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

    int iteration_limit_per_2;
	iteration_limit_per_2 = ITERATIONS / 2;
    int measure_at;
	measure_at = iteration_limit_per_2 - 1;



    cudaDeviceSynchronize();
	begin = std::chrono::steady_clock::now();
    // Launch a kernel on the GPU with one thread for each element.
    for (size_t i = 0; i < iteration_limit_per_2; i++)
    {
        
        iterateKernel << <dimGrid, dimBlock >> > (dev_ua, dev_ub, dev_v, res);
        cudaDeviceSynchronize();

        no_flux_kernel << < dimGrid, dimBlock >> > (dev_ub, res);
        cudaDeviceSynchronize();

        iterateKernel << < dimGrid, dimBlock >> > (dev_ub, dev_ua, dev_v, res);
        cudaDeviceSynchronize();

        no_flux_kernel << < dimGrid, dimBlock >> > (dev_ua, res);
        cudaDeviceSynchronize();
        //cout << i << "\n";
    }
	end = std::chrono::steady_clock::now();
	std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
	std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
	average_iteration_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() / 2 / iteration_limit_per_2;
	cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[ns]" << endl;
	average_iteration_runtime = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 2 / iteration_limit_per_2;
	cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[ns]" << endl;
	
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

void text_to_array(float* u, float* v) {
    //const unsigned int s = 10000;
    fstream newfile;

    newfile.open("u_ff.txt", ios::in); //open a file to perform read operation using file object
    if (newfile.is_open())  //checking whether the file is open
    {
        string tp;
        int i = 0;
        while (getline(newfile, tp))  //read data from file object and put it into string.
        {
            //cout << tp << "\n"; //print the data of the string
            u[i++] = stof(tp);
        }
        newfile.close(); //close the file object.
    }
    newfile.open("v_ff.txt", ios::in); //open a file to perform read operation using file object
    if (newfile.is_open())  //checking whether the file is open
    {
        string tp;
        int i = 0;
        while (getline(newfile, tp))  //read data from file object and put it into string.
        {
            //cout << tp << "\n"; //print the data of the string
            v[i++] = stof(tp);
        }
        //cout << "v: " << v[0] << "\n";
        newfile.close(); //close the file object.
    }
}

void array_to_text(float* u_sol, int res)
{
    const unsigned int s = res * res;
    fstream newfile;
    newfile.open("u_sol_3_4_m.txt", ios::out);  // open a file to perform write operation using file object
    if (newfile.is_open()) //checking whether the file is open
    {
        for (int i = 0; i < s; i++)
        {
            newfile << (u_sol[i]) << "\n"; //inserting text
        }
        newfile << "\n"; //inserting text

        newfile.close(); //close the file object
    }
};