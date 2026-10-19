#include "cuda_runtime.h"

#define PADDING 0

static __device__ __forceinline__ void no_flux(float* g_ub, int res, int i, int j, int ThreadOffset);
static __device__ __forceinline__ void no_flux_smem(float tile[][32], int res, int i, int j, int ThreadOffset);
//static __device__ __forceinline__ void swap_c(float* a, float* b);
//static __device__ __forceinline__ void swap_c(float a[][32], float b[][32], float tmp[][32]);

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(const float r_ua, const float r_v, float* r_un, float* r_vn)
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
    const float repsilon = 1 / 0.02;	// Inkább: repsilon = 1.0/0.02
    const float ra = 1 / 0.64;		// Inkább: ra = 1.0/0.64
    const float b = 0.05;
    const float dt = 0.0030518;

    // Az osztásokat kicserélnémreciprokkal való szorzásra! Lásd a kommenteket fent.
    // Az osztások nagyon lassúak, kb 50-70 szer lassabb mint egy szorzás vagy összeadás/kivonás
    u_th = (r_v + b) * ra; // ( r_v + b ) * ra // ez a sor összevonható lenne a +2. sorral a +2. sorban
    *r_vn = r_v + dt * (r_ua - r_v);
    *r_un = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th); // ... dt * repsilon * r_ua ...
}

static __device__ __forceinline__ void laplace_old(float* g_ua, float tile[][32], float* r_un, int res, int i, int j, int ThreadOffset)
{
    if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
    {
        float dx = 0.195312500000000;
        float dt = 0.003051800000000;
        float ccc = dt / 6 / dx / dx;
        float r_L = 0;

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

        *r_un += r_L;
    }
}
// deal with laplace
static __device__ __forceinline__ void laplace(float* g_ua, float tile[][32], float* r_un, int res, int i, int j, int ThreadOffset)
{
    if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
    {
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        float dx = 0.195312500000000;
        float dt = 0.003051800000000;
        float ccc = dt / 6 / dx / dx;
        float r_L = 0;
        // float r_T = 0;

        bool N = tidy == 0;
        bool E = tidx == 31;
        bool S = tidy == 31;
        bool W = tidx == 0;
        


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
            //else if ((tidx != 0) && (tidx != 31) && (tidy != 0) && (tidy != 31))
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
        //__threadfence_block();
        /*
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
        */

        /*
        // Nem nagy a stencil, kilehet bontani teljesen:
        float* g_ua_act = g_ua + ThreadOffset; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD
        r_L += -20 * g_ua_act[0] * 0;
        r_L += 4 * g_ua_act[+1] * (int)E;
        r_L += 4 * g_ua_act[-1] * (int)W;

        g_ua_act = g_ua_act - res; // Eltoljuk a pointert 1 sorral feljebb, 1 INT ADD
        r_L += 4 * g_ua_act[0] * (int)N;
        r_L += 1 * g_ua_act[+1] * (int)(N || E);
        r_L += 1 * g_ua_act[-1] * (int)(N || W);

        g_ua_act = g_ua_act + 2 * res; // Eltoljuk a pointert 2 sorral lejjebb, 1 INT FMA
        r_L += 4 * g_ua_act[0] * (int)S;
        r_L += 1 * g_ua_act[+1] * (int)(S || E);
        r_L += 1 * g_ua_act[-1] * (int)(S || W);

        //__threadfence();

        if ( (tidx != 0) && (tidx != 31) && (tidy != 0) && (tidy != 31) )
        {
            r_L += - 20 * tile[tidy][tidx] * 1;
            r_L += 4 * tile[tidy][tidx+1] * (int)(!(N || E));
            r_L += 4 * tile[tidy][tidx-1] * (int)(!(N || W));

            r_L += 4 * tile[tidy-1][tidx] * (int)(!S);
            r_L += 1 * tile[tidy-1][tidx+1] * (int)(!(S || E));
            r_L += 1 * tile[tidy-1][tidx-1] * (int)(!(N || W));

            r_L += 4 * tile[tidy+1][tidx] * (int)(!S);
            r_L += 1 * tile[tidy+1][tidx+1] * (int)(!(S || E));
            r_L += 1 * tile[tidy+1][tidx-1] * (int)(!(S || W));

        }
        */
        //__threadfence_block();
        
        //__syncthreads();

        r_L = ccc * r_L; // Ez nem tudom miért kell ide de ha kell, akkor kell :)
        // Ebben az esetben a d_coeff nem kell. A beégetett számok a constant memóriába kerülnek automatikusan.

        //Műveletek összesen: //ezt módosítottam, változhatott az arány, idk
        // 9 float FMA, 1 float MUL
        // 2 int FMA, 2 int ADD
        // Arány: 10/4

        *r_un += r_L;
    }
}

__global__ void iterate_kernel(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();

    if (i < res && j < res)
    {
        __shared__ float tile_a[32][32];
        __shared__ float tile_b[32][32];
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;

        // ----- Load Data -----
        //copy_g_to_r(g_ua, &r_ua, ThreadOffset);
        //copy_g_to_r(g_v, &r_v, ThreadOffset);
        copy_g_to_smem(g_ua, tile_a, ThreadOffset);
        copy_smem_to_r(tile_a, &r_ua);
        copy_g_to_r(g_v, &r_v, ThreadOffset);
        whole_grid.sync();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            //r_un = 0;
            iterate_without_laplace(r_ua, r_v, &r_un, &r_vn);

            // ----- Laplace -----
            laplace_old(g_ua, tile_a, &r_un, res, i, j, ThreadOffset);
            

            // ----- "Unload" Data -----
            copy_r_to_g(g_ub, r_un, ThreadOffset);
            //copy_r_to_g(g_v, r_vn, ThreadOffset);
            whole_grid.sync();

            // ----- No Flux Bc -----
            no_flux(g_ub, res, i, j, ThreadOffset);
            whole_grid.sync();

            // ----- Iteration step 2 -----
            //r_ua = 0;
            iterate_without_laplace(r_un, r_vn, &r_ua, &r_v);

            // ----- Laplace -----
            laplace_old(g_ub, tile_b, &r_ua, res, i, j, ThreadOffset);

            // ----- "Unload" Data -----
            copy_r_to_g(g_ua, r_ua, ThreadOffset);
            //copy_r_to_g(g_v, r_v, ThreadOffset);
            whole_grid.sync();

            // ----- No Flux Bc -----
            no_flux(g_ua, res, i, j, ThreadOffset);
            whole_grid.sync();
        }
        copy_g_to_smem(g_ua, tile_a, ThreadOffset);
        whole_grid.sync();
        copy_smem_to_g(tile_a, g_ua, ThreadOffset);
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

            // ----- Laplace -----
            laplace(g_ua, tile_a, &r_ub, res, i, j, ThreadOffset);
            
            // ----- "Unload" Data -----
            copy_r_to_smem(r_ub, tile_b);
            whole_grid.sync();
            __threadfence_block();
            // ----- No Flux Bc -----
            no_flux_smem(tile_b, res, i, j, ThreadOffset);
            whole_grid.sync();
            copy_smem_to_g(tile_b, g_ub, ThreadOffset);
            //copy_smem_to_g_boundary(tile_b, g_ub, i, j, ThreadOffset);
            whole_grid.sync();

            // ----- Iteration step 2 -----
            iterate_without_laplace(r_ub, r_vn, &r_ua, &r_v);
            
            // ----- Laplace -----
            laplace(g_ub, tile_b, &r_ua, res, i, j, ThreadOffset);
            
            // ----- "Unload" Data -----
            copy_r_to_smem(r_ua, tile_a);
            //tile_a[tidy][tidx] = r_ua;
            whole_grid.sync();
            __threadfence_block();
            // ----- No Flux Bc -----
            no_flux_smem(tile_a, res, i, j, ThreadOffset);
            whole_grid.sync();
            copy_smem_to_g(tile_a, g_ua, ThreadOffset);
            //copy_smem_to_g_boundary(tile_a, g_ua, i, j, ThreadOffset);
            whole_grid.sync();

        }
        copy_smem_to_g(tile_a, g_ua, ThreadOffset);
        copy_smem_to_g(tile_b, g_ub, ThreadOffset);
        whole_grid.sync();

    }
}


__global__ void iterate_kernel_smem_test(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
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

            // ----- Laplace -----
            laplace(g_ua, tile_a, &r_ub, res, i, j, ThreadOffset);

            // ----- "Unload" Data -----
            copy_r_to_g(g_ub, r_ub, ThreadOffset);
            whole_grid.sync();

            // ----- No Flux Bc -----
            no_flux(g_ub, res, i, j, ThreadOffset);
            whole_grid.sync();
            copy_g_to_smem(g_ub, tile_b, ThreadOffset);
            //copy_smem_to_g(tile_b, g_ub, ThreadOffset);
            //copy_smem_to_g_boundary(tile_b, g_ub, i, j, ThreadOffset);
            whole_grid.sync();

            // ----- Iteration step 2 -----
            iterate_without_laplace(r_ub, r_vn, &r_ua, &r_v);

            // ----- Laplace -----
            laplace(g_ub, tile_b, &r_ua, res, i, j, ThreadOffset);

            // ----- "Unload" Data -----
            //copy_r_to_smem(r_ua, tile_a);
            copy_r_to_g(g_ua, r_ua, ThreadOffset);
            whole_grid.sync();

            // ----- No Flux Bc -----
            no_flux(g_ua, res, i, j, ThreadOffset);
            whole_grid.sync();
            copy_g_to_smem(g_ua, tile_a, ThreadOffset);
            //copy_smem_to_g(tile_a, g_ua, ThreadOffset);
            //copy_smem_to_g_boundary(tile_a, g_ua, i, j, ThreadOffset);
            whole_grid.sync();

        }
        copy_smem_to_g(tile_a, g_ua, ThreadOffset);
        copy_smem_to_g(tile_b, g_ub, ThreadOffset);
        whole_grid.sync();

    }
}



// no flux boundary condition (global memory)
static __device__ __forceinline__ void no_flux(float* g_ub, int res, int i, int j, int ThreadOffset)
{
    // ----- No Flux BC -----
    grid_group whole_grid = this_grid();
    // Hasonló módon az integer spórólások.
    // Továbbá nem kell az else, mert az egyértelmű. Ha i==0, akkor nem lehet i==res-1
    // Egy probléma van, ez a sarkok kezelése. Lehet azokat külön kéne venni. Az is lehet hogy nem számít honna vesszük a sarokról az adatot.
    // Alkalmaztam üres mezőket, hogy átláthatóbb legyen a dolog.
    if (i == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + 1]; }
    whole_grid.sync();
    if (i == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - 1]; }
    whole_grid.sync();
    if (j == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + res]; }
    whole_grid.sync();
    if (j == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - res]; }
}

// no flux boundary condition (shared memory)
static __device__ __forceinline__ void no_flux_smem(float tile[32][32], int res, int i, int j, int ThreadOffset)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    // ----- No Flux BC -----
    grid_group whole_grid = this_grid();
    // Hasonló módon az integer spórólások.
    // Továbbá nem kell az else, mert az egyértelmű. Ha i==0, akkor nem lehet i==res-1
    // Egy probléma van, ez a sarkok kezelése. Lehet azokat külön kéne venni. Az is lehet hogy nem számít honna vesszük a sarokról az adatot.
    // Alkalmaztam üres mezőket, hogy átláthatóbb legyen a dolog.
    if (i == 0) {               tile[tidy][0] =  tile[tidy][1];}
    whole_grid.sync();
    if (i == res - 1) {  tile[tidy][31] =  tile[tidy][30]; }
    whole_grid.sync();
    if (j == 0) {               tile[0][tidx] =  tile[1][tidx]; }
    whole_grid.sync();
    if (j == res - 1) {  tile[31][tidx] =  tile[30][tidx]; }
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
        whole_grid.sync();
        __threadfence();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            // iterate_without_laplace(r_ua, r_v, &r_ub, &r_vn);
            u_th = (r_v + b) * ra;
            whole_grid.sync();
            r_vn = r_v + dt * (r_ua - r_v);
            whole_grid.sync();
            r_ub = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th);
            whole_grid.sync();

            // ----- Laplace -----
            // laplace(g_ua, tile_a, &r_ub, res, i, j, ThreadOffset);

            if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
            {
                r_L = 0;
                //whole_grid.sync();
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
                    //else if ((tidx != 0) && (tidx != 31) && (tidy != 0) && (tidy != 31))
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

                __threadfence();

                r_L = ccc * r_L;
                r_ub += r_L;
            }


            // ----- "Unload" Data -----
            // copy_r_to_smem(r_ub, tile_b);
            tile_b[tidy][tidx] = r_ub;
            __threadfence();
            whole_grid.sync();

            // ----- No Flux Bc -----
            //no_flux_smem(tile_b, res, i, j, ThreadOffset);
            if (i == 0) { tile_b[tidy][tidx] = tile_b[tidy][tidx + 1]; }
            whole_grid.sync();
            __threadfence();
            if (i == blockDim.x - 1) { tile_b[tidy][tidx] = tile_b[tidy][tidx - 1]; }
            whole_grid.sync();
            __threadfence();
            if (j == 0) { tile_b[tidy][tidx] = tile_b[tidy + 1][tidx]; }
            whole_grid.sync();
            __threadfence();
            if (j == blockDim.y - 1) { tile_b[tidy][tidx] = tile_b[tidy - 1][tidx]; }
            __threadfence();
            whole_grid.sync();
            // copy_smem_to_g(tile_b, g_ub, ThreadOffset);
            g_ub[ThreadOffset] = tile_b[tidy][tidx];
            __threadfence();
            //copy_smem_to_g_boundary(tile_b, g_ub, i, j, ThreadOffset);
            whole_grid.sync();

            // ----- Iteration step 2 -----
            // iterate_without_laplace(r_ub, r_vn, &r_ua, &r_v);

            u_th = (r_vn + b) * ra;
            r_v = r_vn + dt * (r_ub - r_vn);
            r_ua = r_ub + dt * repsilon * r_ub * (1 - r_ub) * (r_ub - u_th);

            // ----- Laplace -----
            // laplace(g_ub, tile_b, &r_ua, res, i, j, ThreadOffset);

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
                    //else if ((tidx != 0) && (tidx != 31) && (tidy != 0) && (tidy != 31))
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

                __threadfence();

                r_L = ccc * r_L;
                r_ua += r_L;
            }

            whole_grid.sync();

            // ----- "Unload" Data -----
            // copy_r_to_smem(r_ua, tile_a);
            tile_a[tidy][tidx] = r_ua;
            whole_grid.sync();

            // ----- No Flux Bc -----
            //no_flux_smem(tile_a, res, i, j, ThreadOffset);

            if (i == 0) { tile_a[tidy][tidx] = tile_a[tidy][tidx + 1]; }
            whole_grid.sync();
            __threadfence();
            if (i == blockDim.x - 1) { tile_a[tidy][tidx] = tile_a[tidy][tidx - 1]; }
            whole_grid.sync();
            __threadfence();
            if (j == 0) { tile_a[tidy][tidx] = tile_a[tidy + 1][tidx]; }
            whole_grid.sync();
            __threadfence();
            if (j == blockDim.y - 1) { tile_a[tidy][tidx] = tile_a[tidy - 1][tidx]; }
            whole_grid.sync();
            __threadfence();
            // copy_smem_to_g(tile_a, g_ua, ThreadOffset);
            g_ua[ThreadOffset] = tile_a[tidy][tidx];
            __threadfence();
            //copy_smem_to_g_boundary(tile_a, g_ua, i, j, ThreadOffset);
            whole_grid.sync();




        }
        //copy_smem_to_g(tile_a, g_ua, ThreadOffset);
        //g_ua[ThreadOffset] = tile_a[tidy][tidx];
        //copy_smem_to_g(tile_b, g_ub, ThreadOffset);
        //g_ub[ThreadOffset] = tile_b[tidy][tidx];
        whole_grid.sync();

    }
}

/*
__global__ void iterate_kernel_swap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();

    if (i < res && j < res)
    {
        __shared__ float tile_a[32][32];
        __shared__ float tile_b[32][32];
        float tile_tmp[32][32];
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_ub;
        float r_vn;
        
        float* tmp;
        float* p_ua = &r_ua;
        float* p_ub = &r_ub;
        float* p_v = &r_v;
        float* p_vn = &r_vn;

        

        // ----- Load Data -----
        //copy_g_to_r(g_ua, &r_ua, ThreadOffset);
        //copy_g_to_r(g_v, &r_v, ThreadOffset);
        copy_g_to_smem(g_ua, tile_a, ThreadOffset);
        copy_smem_to_r(tile_a, &r_ua);
        copy_g_to_r(g_v, &r_v, ThreadOffset);
        whole_grid.sync();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            //r_un = 0;
            iterate_without_laplace(r_ua, r_v, &r_ub, &r_vn);

            // ----- Laplace -----
            laplace_old(g_ua, tile_a, &r_ub, res, i, j, ThreadOffset);


            // ----- "Unload" Data -----
            copy_r_to_g(g_ub, r_ub, ThreadOffset);
            //copy_r_to_g(g_v, r_vn, ThreadOffset);
            whole_grid.sync();

            // ----- No Flux Bc -----
            no_flux(g_ub, res, i, j, ThreadOffset);
            whole_grid.sync();


            tmp = &r_ub;
            &r_ub = &r_ua;
            &r_ua = tmp;

            tmp = &r_v;
            &r_v = &r_vn;
            &r_vn = tmp;

            tmp = g_ub;
            g_ub = g_ua;
            g_ua = tmp;

            tile_tmp = &tile_b;
            tile_b = tile_a;
            tile_a = tile_tmp;

        }
    }
}
*/
/*
static __device__ __forceinline__ void swap_c(float* a, float* b)
{
    float* tmp = b;
    b = a;
    a = tmp;
}
static __device__ __forceinline__ void swap_c(float a[][32], float b[][32], float tmp[][32])
{
    b = a;
    a = tmp;
}
*/