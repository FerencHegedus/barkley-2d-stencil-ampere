#include "cuda_runtime.h"

static __device__ __forceinline__ void no_flux(float* g_ub, int res, int i, int j, int ThreadOffset);

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

// deal with laplace
static __device__ __forceinline__ void laplace(float* g_ua, float* r_un, int res, int i, int j, int ThreadOffset)
{
    if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
    {

        float dx = 0.195312500000000;
        float dt = 0.003051800000000;
        float ccc = dt / 6 / dx / dx;
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

        //__threadfence();
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
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;

        // ----- Load Data -----
        copy_g_to_r(g_ua, &r_ua, ThreadOffset);
        copy_g_to_r(g_v, &r_v, ThreadOffset);
        whole_grid.sync();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            //r_un = 0;
            iterate_without_laplace(r_ua, r_v, &r_un, &r_vn);
            //whole_grid.sync();

            // ----- Laplace -----
            laplace(g_ua, &r_un, res, i, j, ThreadOffset);
            //whole_grid.sync();

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
            //whole_grid.sync();

            // ----- Laplace -----
            laplace(g_ub, &r_ua, res, i, j, ThreadOffset);
            //whole_grid.sync();

            // ----- "Unload" Data -----
            copy_r_to_g(g_ua, r_ua, ThreadOffset);
            //copy_r_to_g(g_v, r_v, ThreadOffset);
            whole_grid.sync();

            // ----- No Flux Bc -----
            no_flux(g_ua, res, i, j, ThreadOffset);
            whole_grid.sync();
        }
    }
}

// no flux boundary condition
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
    //whole_grid.sync();
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

/*
* for debuging of v6.2
* result: sochastic behaviour stopped around NE corner
* 
__global__ void iterate_kernel_exp(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    if (i < res && j < res)
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;

        float u_th = 0;
        const float repsilon = 1 / 0.02;	// Inkább: repsilon = 1.0/0.02
        const float ra = 1 / 0.64;		// Inkább: ra = 1.0/0.64
        const float b = 0.05;
        // const float dt = 0.0030518;

        float* g_ua_act;
        const float dx = 0.195312500000000;
        const float dt = 0.003051800000000;
        float ccc = dt / 6 / dx / dx;
        float r_L = 0;

        // ----- Load Data -----
        //copy_g_to_r(g_ua, &r_ua, ThreadOffset);
        r_ua = g_ua[ThreadOffset];
        //copy_g_to_r(g_v, &r_v, ThreadOffset);
        r_v = g_v[ThreadOffset];
        whole_grid.sync();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            //r_un = 0;
            // iterate_without_laplace(r_ua, r_v, &r_un, &r_vn);
            u_th = (r_v + b) * ra;
            r_vn = r_v + dt * (r_ua - r_v);
            r_un = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th);
            whole_grid.sync();

            // ----- Laplace -----
            // laplace(g_ua, &r_un, res, i, j, ThreadOffset);
            if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
            {
                r_L = 0;

                // Nem nagy a stencil, kilehet bontani teljesen:
                g_ua_act = g_ua + ThreadOffset; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD
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

                //__threadfence();
                //__syncthreads();

                r_L = ccc * r_L; // Ez nem tudom miért kell ide de ha kell, akkor kell :)
                // Ebben az esetben a d_coeff nem kell. A beégetett számok a constant memóriába kerülnek automatikusan.

                //Műveletek összesen: //ezt módosítottam, változhatott az arány, idk
                // 9 float FMA, 1 float MUL
                // 2 int FMA, 2 int ADD
                // Arány: 10/4

                r_un += r_L;
            }
            whole_grid.sync();

            // ----- "Unload" Data -----
            // copy_r_to_g(g_ub, r_un, ThreadOffset);
            // copy_r_to_g(g_v, r_vn, ThreadOffset);
            g_ub[ThreadOffset] = r_un;
            whole_grid.sync();
            g_v[ThreadOffset] = r_vn;
            whole_grid.sync();

            // ----- No Flux Bc -----
            // no_flux(g_ub, res, i, j, ThreadOffset);
            if (i == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + 1]; }
            whole_grid.sync();
            if (i == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - 1]; }
            whole_grid.sync();
            if (j == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + res]; }
            whole_grid.sync();
            if (j == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - res]; }
            whole_grid.sync();

            // ----- Iteration step 2 -----
            //r_ua = 0;
            iterate_without_laplace(r_un, r_vn, &r_ua, &r_v);
            //whole_grid.sync();

            // ----- Laplace -----
            laplace(g_ub, &r_ua, res, i, j, ThreadOffset);
            whole_grid.sync();

            // ----- "Unload" Data -----
            copy_r_to_g(g_ua, r_ua, ThreadOffset);
            copy_r_to_g(g_v, r_v, ThreadOffset);
            whole_grid.sync();

            // ----- No Flux Bc -----
            no_flux(g_ua, res, i, j, ThreadOffset);
            whole_grid.sync();
        }
    }
}
*/