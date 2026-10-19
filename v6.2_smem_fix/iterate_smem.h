// v6.2
#include "cuda_runtime.h"

#define PADDING 0

__const__ float repsilon = 1 / 0.02;
__const__ float ra = 1 / 0.64;
__const__ float b = 0.05;
__const__ float dt = 0.0030518;
__const__ float ccc = 0.013333517126739025115966796875;

static __device__ __forceinline__ void no_flux(float* g_ub, int res, int i, int j, int ThreadOffset);
static __device__ __forceinline__ void no_flux_smem(float tile[][32], int res, int i, int j, int tidx, int tidy);

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(const float r_ua, const float r_v, float* r_un, float* r_vn)
{
    float u_th;


    u_th = (r_v + b) * ra; 
    *r_vn = r_v + dt * (r_ua - r_v);
    *r_un = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th); 
}

// deal with laplace
static __device__ __forceinline__ void laplace(float* g_ua, float tile[][32], float* r_un, int res, int i, int j, int ThreadOffset, int tidx, int tidy)
{
    if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
    {
        
        //float dx = 0.195312500000000;
        //float dt = 0.003051800000000;
        //float ccc = dt / 6 / dx / dx;
        float r_L = 0;

        //bool N = tidy == 0;
        //bool E = tidx == 31;
        //bool S = tidy == 31;
        //bool W = tidx == 0;

        if ((tidx == 0) || (tidx == 31) || (tidy == 0) || (tidy == 31))
        {
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

        }
        else
        {
            r_L += -20 * tile[tidy][tidx];
            r_L += 4 * tile[tidy][tidx + 1];
            r_L += 4 * tile[tidy][tidx - 1];

            r_L += 4 * tile[tidy - 1][tidx];
            r_L += 1 * tile[tidy - 1][tidx + 1];
            r_L += 1 * tile[tidy - 1][tidx - 1];

            r_L += 4 * tile[tidy + 1][tidx];
            r_L += 1 * tile[tidy + 1][tidx + 1];
            r_L += 1 * tile[tidy + 1][tidx - 1];
        }

        r_L = ccc * r_L;

        *r_un += r_L;
    }
}

__global__ void iterate_kernel_smem(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    if (i < res && j < res)
    {
        __shared__ float tile_a[32][32];
        __shared__ float tile_b[32][32];
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_ub;
        float r_vn;

        // ----- Load Data -----
        copy_g_to_smem(g_ua, tile_a, ThreadOffset);
        copy_smem_to_r(tile_a, &r_ua);
        copy_g_to_r(g_v, &r_v, ThreadOffset);
        // idk why but works without sync
        // whole_grid.sync();
        
#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            iterate_without_laplace(r_ua, r_v, &r_ub, &r_vn);

            whole_grid.sync();
            // ----- Laplace -----
            laplace(g_ua, tile_a, &r_ub, res, i, j, ThreadOffset, tidx, tidy);
            
            // ----- "Unload" Data -----
            copy_r_to_smem(r_ub, tile_b);
            
            // ----- No Flux Bc -----
            no_flux_smem(tile_b, res, i, j, tidx, tidy);
            
            copy_smem_to_g(tile_b, g_ub, ThreadOffset, tidx, tidy);

            // ----- Iteration step 2 -----
            iterate_without_laplace(r_ub, r_vn, &r_ua, &r_v);

            whole_grid.sync();
            // ----- Laplace -----
            laplace(g_ub, tile_b, &r_ua, res, i, j, ThreadOffset, tidx, tidy);
            
            // ----- "Unload" Data -----
            copy_r_to_smem(r_ua, tile_a);
            
            // ----- No Flux Bc -----
            no_flux_smem(tile_a, res, i, j, tidx, tidy);
            
            copy_smem_to_g(tile_a, g_ua, ThreadOffset, tidx, tidy);
        }
    }
}

// no flux boundary condition (shared memory)
static __device__ __forceinline__ void no_flux_smem(float tile[32][32], int res, int i, int j, int tidx, int tidy)
{
    // ----- No Flux BC -----
    if (i == 0) { tile[tidy][0] = tile[tidy][1]; }
    if (i == res - 1) { tile[tidy][31] = tile[tidy][30]; }
    __syncthreads();
    if (j == 0) { tile[0][tidx] = tile[1][tidx]; }
    if (j == res - 1) { tile[31][tidx] = tile[30][tidx]; }
}

/*
__global__ void iterate_kernel_smem_exp(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    if (i < res && j < res)
    {
        __shared__ alignas(alignof(float4)) float tile_a[32][32];
        __shared__ alignas(alignof(float4)) float tile_b[32][32];
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_ub;
        float r_vn;

        float u_th;

        // ----- constants -----
        const float repsilon = 1 / 0.02;	// Inkább: repsilon = 1.0/0.02
        const float ra = 1 / 0.64;		// Inkább: ra = 1.0/0.64
        const float b = 0.05;
        const float dt = 0.0030518;

        float dx = 0.195312500000000;
        // float dt = 0.003051800000000;
        float ccc = dt / 6 / dx / dx;
        float r_L = 0;
        // float r_T = 0;

        bool N = tidy == 0;
        bool E = tidx == 31;
        bool S = tidy == 31;
        bool W = tidx == 0;

        // ----- Load Data -----
        tile_a[tidy][tidx] = g_ua[ThreadOffset];
        r_ua = tile_a[tidy][tidx];
        r_v = g_v[ThreadOffset];
        // idk why but works without sync
        //whole_grid.sync();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            u_th = (r_v + b) * ra;
            r_vn = r_v + dt * (r_ua - r_v);
            r_ub = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th);

            // ----- Laplace -----
            if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
            {
                r_L = 0;
                if ((tidx == 0) || (tidx == 31) || (tidy == 0) || (tidy == 31))
                {
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
                }
                else
                {
                    r_L += -20 * tile_a[tidy][tidx];
                    r_L += 4 * tile_a[tidy][tidx + 1];
                    r_L += 4 * tile_a[tidy][tidx - 1];

                    r_L += 4 * tile_a[tidy - 1][tidx];
                    r_L += 1 * tile_a[tidy - 1][tidx + 1];
                    r_L += 1 * tile_a[tidy - 1][tidx - 1];

                    r_L += 4 * tile_a[tidy + 1][tidx];
                    r_L += 1 * tile_a[tidy + 1][tidx + 1];
                    r_L += 1 * tile_a[tidy + 1][tidx - 1];
                }
                r_L = ccc * r_L;
                r_ub += r_L;
            }
            __syncthreads();

            // ----- "Unload" Data -----
            tile_b[tidy][tidx] = r_ub;
            __syncthreads();

            // ----- No Flux Bc -----
            if (i == 0) { tile_b[tidy][tidx] = tile_b[tidy][tidx + 1]; }
            __syncthreads();
            if (i == res - 1) { tile_b[tidy][tidx] = tile_b[tidy][tidx - 1]; }
            __syncthreads();
            if (j == 0) { tile_b[tidy][tidx] = tile_b[tidy + 1][tidx]; }
            __syncthreads();
            if (j == res - 1) { tile_b[tidy][tidx] = tile_b[tidy - 1][tidx]; }
            __syncthreads();

            g_ub[ThreadOffset] = tile_b[tidy][tidx];
            //__syncthreads();
            whole_grid.sync();

            // ----- Iteration step 2 -----
            u_th = (r_vn + b) * ra;
            r_v = r_vn + dt * (r_ub - r_vn);
            r_ua = r_ub + dt * repsilon * r_ub * (1 - r_ub) * (r_ub - u_th);

            // ----- Laplace -----
            if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
            {
                r_L = 0;

                if ((tidx == 0) || (tidx == 31) || (tidy == 0) || (tidy == 31))
                {
                    float* g_ua_act = g_ub + ThreadOffset; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD
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
                }
                else
                {
                    r_L += -20 * tile_b[tidy][tidx];
                    r_L += 4 * tile_b[tidy][tidx + 1];
                    r_L += 4 * tile_b[tidy][tidx - 1];

                    r_L += 4 * tile_b[tidy - 1][tidx];
                    r_L += 1 * tile_b[tidy - 1][tidx + 1];
                    r_L += 1 * tile_b[tidy - 1][tidx - 1];

                    r_L += 4 * tile_b[tidy + 1][tidx];
                    r_L += 1 * tile_b[tidy + 1][tidx + 1];
                    r_L += 1 * tile_b[tidy + 1][tidx - 1];
                }
                r_L = ccc * r_L;
                r_ua += r_L;
            }

            __syncthreads();

            // ----- "Unload" Data -----
            tile_a[tidy][tidx] = r_ua;
            __syncthreads();

            // ----- No Flux Bc -----
            if (i == 0) { tile_a[tidy][tidx] = tile_a[tidy][tidx + 1]; }
            __syncthreads();
            if (i == res - 1) { tile_a[tidy][tidx] = tile_a[tidy][tidx - 1]; }
            __syncthreads();
            if (j == 0) { tile_a[tidy][tidx] = tile_a[tidy + 1][tidx]; }
            __syncthreads();
            if (j == res - 1) { tile_a[tidy][tidx] = tile_a[tidy - 1][tidx]; }
            __syncthreads();
            g_ua[ThreadOffset] = tile_a[tidy][tidx];
            whole_grid.sync();
        } // end of iterations
    } // end of inner of grid
}
*/