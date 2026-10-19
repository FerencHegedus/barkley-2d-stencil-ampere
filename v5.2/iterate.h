#include "cuda_runtime.h"

//__constant__ float dx = 0.1953125;
__constant__ float dt = 0.0030518;
__constant__ float ccc = 0.013333517126739025115966796875;
__constant__ float repsilon = 1 / 0.02;
__constant__ float ra = 1 / 0.64;
__constant__ float b = 0.05;

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
    //const float repsilon = 1 / 0.02;	// Inkább: repsilon = 1.0/0.02
    //const float ra = 1 / 0.64;		// Inkább: ra = 1.0/0.64
    //const float b = 0.05;
    //const float dt = 0.003051800000000;

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

        //const float dx = 0.195312500000000;
        //const float dt = 0.003051800000000;
        //float ccc = dt / 6 / dx / dx;
        //const float ccc = 0.013333517126739025115966796875000000000000000000000000000000;
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

        
        //__threadfence_block();

        r_L = ccc * r_L; // Ez nem tudom miért kell ide de ha kell, akkor kell :)
        // Ebben az esetben a d_coeff nem kell. A beégetett számok a constant memóriába kerülnek automatikusan.

        //Műveletek összesen: //ezt módosítottam, változhatott az arány, idk
        // 9 float FMA, 1 float MUL
        // 2 int FMA, 2 int ADD
        // Arány: 10/4

        *r_un += r_L;
    }
}

__global__ void iterateKernel(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
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
        copy_g_to_r(g_ua, g_v, &r_ua, &r_v, ThreadOffset);
        //whole_grid.sync();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            iterate_without_laplace(r_ua, r_v, &r_un, &r_vn);
            
            // ----- Laplace -----
            laplace(g_ua, &r_un, res, i, j, ThreadOffset);

            // ----- "Unload" Data -----
            copy_r_to_g(g_ub, g_v, r_un, r_vn, ThreadOffset);
            __syncthreads();

            // ----- No Flux Bc -----
            no_flux(g_ub, res, i, j, ThreadOffset);
            whole_grid.sync();

            // ----- Iteration step 2 -----
            iterate_without_laplace(r_un, r_vn, &r_ua, &r_v);
            
            // ----- Laplace -----
            laplace(g_ub, &r_ua, res, i, j, ThreadOffset);
            
            // ----- "Unload" Data -----
            copy_r_to_g(g_ua, g_v, r_ua, r_v, ThreadOffset);
            __syncthreads();

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

    // Hasonló módon az integer spórólások.
    // Továbbá nem kell az else, mert az egyértelmű. Ha i==0, akkor nem lehet i==res-1
    // Egy probléma van, ez a sarkok kezelése. Lehet azokat külön kéne venni. Az is lehet hogy nem számít honna vesszük a sarokról az adatot.
    // Alkalmaztam üres mezőket, hogy átláthatóbb legyen a dolog.
    if (i == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + 1]; }
    
    if (i == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - 1]; }
    __syncthreads();
    if (j == 0) { g_ub[ThreadOffset] = g_ub[ThreadOffset + res]; }
    
    if (j == res - 1) { g_ub[ThreadOffset] = g_ub[ThreadOffset - res]; }
    
}

