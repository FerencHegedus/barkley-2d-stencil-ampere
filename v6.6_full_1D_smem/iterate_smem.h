// v6.6
// Ezeket vagy a constant vagy a shared memóriába raknám.
// Constant memóriába úgy is be tudod "rakni" ha beégeted a számot ahol használod.
// Ha változtatni akarod, akkor lehet #define -t is használni (a global függvényen kívül természetesen).
// DE NEM KRITIKUS

// Az osztásokat kicserélnémreciprokkal való szorzásra! Lásd a kommenteket fent.
// Az osztások nagyon lassúak, kb 50-70 szer lassabb mint egy szorzás vagy összeadás/kivonás

// Ez nem tudom miért kell ide de ha kell, akkor kell :)
// Ebben az esetben a d_coeff nem kell. A beégetett számok a constant memóriába kerülnek automatikusan.

//Műveletek összesen: //ezt módosítottam, változhatott az arány, idk
// 9 float FMA, 1 float MUL
// 2 int FMA, 2 int ADD
// Arány: 10/4

// 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)

// Hasonló módon az integer spórólások.
// Továbbá nem kell az else, mert az egyértelmű. Ha i==0, akkor nem lehet i==res-1
// Egy probléma van, ez a sarkok kezelése. Lehet azokat külön kéne venni. Az is lehet hogy nem számít honna vesszük a sarokról az adatot.
// Alkalmaztam üres mezőket, hogy átláthatóbb legyen a dolog.

#include "cuda_runtime.h"

#define GDIM 8
#define BDIM 32
#define PADDING 1
#define TILEDIM 34
#define TILEDIM_Y 34
// #define TILEDIM_X 35
#define TILEDIM_X 35
/*
#define TILEDIM ( BDIM + 2 )
#define TILEDIM_Y ( BDIM + 2 )
#define TILEDIM_X ( BDIM + 2 + PADDING)
*/

__constant__ float dt = 0.0030518;
__constant__ float ccc = 0.013333517126739025115966796875;
__constant__ float repsilon = 1 / 0.02;
__constant__ float ra = 1 / 0.64;
__constant__ float b = 0.05;

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(const float r_ua, const float r_v, float* r_un, float* r_vn)
{
    float u_th;

    u_th = (r_v + b) * ra;
    *r_vn = r_v + dt * (r_ua - r_v);
    *r_un = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th);
}

// deal with laplace
static __device__ __forceinline__ void laplace(float* tile_act, float r_u, float* r_un, int midx, int midy, int inner_inner)
{
    if (inner_inner)
    {
        float r_L = 0;

        // float* tile_act = &tile[0][0] + midy * TILEDIM_X + midx; // use register or use increase the calculation time ???
        r_L += -20 * r_u;	// fma 
        r_L += 4 * tile_act[+1];
        r_L += 4 * tile_act[-1];

        tile_act -= TILEDIM_X;
        r_L += 4 * tile_act[0];
        r_L += 1 * tile_act[+1];
        r_L += 1 * tile_act[-1];

        tile_act += 2 * TILEDIM_X;
        r_L += 4 * tile_act[0];
        r_L += 1 * tile_act[+1];
        r_L += 1 * tile_act[-1];

        r_L = ccc * r_L;

        *r_un += r_L;
    }
}

// rename variables
__global__ void iterate_kernel_smem(float* g_ua, float* g_ub, float* g_v, g_E g_grid_e_a, g_E g_grid_e_b, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    int tidx = threadIdx.x;
    // int tidy = threadIdx.y;
    int tidx_res = tidx * RES;
    int midx = threadIdx.x + 1;
    int midy = threadIdx.y + 1;
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    if (i < RES && j < RES)
    {
        /*
        __shared__ float tile_a_2[TILEDIM_Y][TILEDIM_X];
        __shared__ float tile_b_2[TILEDIM_Y][TILEDIM_X];
        float* tile_a;
        float* tile_b;
        tile_a = &tile_a_2[0][0];
        tile_b = &tile_b_2[0][0];
        */
        __shared__ float tile_a[TILEDIM_Y * TILEDIM_X];
        __shared__ float tile_b[TILEDIM_Y * TILEDIM_X];
        float* tile_act_a = tile_a + midy * TILEDIM_X + midx; // use register or use increase the calculation time ???
        float* tile_act_b = tile_b + midy * TILEDIM_X + midx;
        int ThreadOffset = i + j * RES;
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_ub;
        //float r_vn;

        /*
        1D tile regs
        s   [0][midx]
        n   [33][midx]
        w   [midx][0]
        e   [midx][33]

        s   [1][midx]
        n   [32][midx]
        w   [midx][1]
        e   [midx][32]
        */

        g_E tile_e_a_in, tile_e_b_in, tile_e_a_out, tile_e_b_out, tile_e_a_in_nf, tile_e_b_in_nf;

        /*
        case 0: // South Edge
            if (b_direction.s_not) { tile[0 * TILEDIM_X + midx] = *g_ec.s; }
            break;
        case 1: // West Edge
            if (b_direction.w_not) { tile[midx * TILEDIM_X + 0] = *g_ec.w; }
            break;
        case 2: // East Edge
            if (b_direction.e_not) { tile[midx * TILEDIM_X + 33] = *g_ec.e; }
            break;
        case 3: // North Edge
            if (b_direction.n_not) { tile[33 * TILEDIM_X + midx] = *g_ec.n; }
            break;
        */

        tile_e_a_out.s = tile_a + midx;
        tile_e_a_out.w = tile_a + midx * TILEDIM_X;
        tile_e_a_out.e = tile_a + midx * TILEDIM_X + 33;
        tile_e_a_out.n = tile_a + 33 * TILEDIM_X + midx;

        tile_e_b_out.s = tile_b + midx;
        tile_e_b_out.w = tile_b + midx * TILEDIM_X;
        tile_e_b_out.e = tile_b + midx * TILEDIM_X + 33;
        tile_e_b_out.n = tile_b + 33 * TILEDIM_X + midx;

        /*
        tile_e_a_out.s = tile_a + midx;
        tile_e_a_out.w = tile_a + midx * TILEDIM_X;
        tile_e_a_out.e = tile_e_a_out.w + TILEDIM - 1;
        tile_e_a_out.n = tile_e_a_out.s + (TILEDIM - 1) * TILEDIM_X;

        tile_e_b_out.s = tile_b + midx;
        tile_e_b_out.w = tile_b + midx * TILEDIM_X;
        tile_e_b_out.e = tile_e_b_out.w + TILEDIM - 1;
        tile_e_b_out.n = tile_e_b_out.s + (TILEDIM - 1) * TILEDIM_X;
        */


        tile_e_a_in.s = tile_a + TILEDIM_X + midx;
        tile_e_a_in.w = tile_a + midx * TILEDIM_X + 1;
        tile_e_a_in.e = tile_a + midx * TILEDIM_X + 32;
        tile_e_a_in.n = tile_a + 32 * TILEDIM_X + midx;

        tile_e_b_in.s = tile_b + TILEDIM_X + midx;
        tile_e_b_in.w = tile_b + midx * TILEDIM_X + 1;
        tile_e_b_in.e = tile_b + midx * TILEDIM_X + 32;
        tile_e_b_in.n = tile_b + 32 * TILEDIM_X + midx;

        // no flux copyig uses different indexing, midy needed
        //copy from registers =>cannot distributer the tasks to warps
        tile_e_a_in_nf.s = tile_a + TILEDIM_X + midx;
        tile_e_a_in_nf.w = tile_a + midy * TILEDIM_X + 1;
        tile_e_a_in_nf.e = tile_a + midy * TILEDIM_X + 32;
        tile_e_a_in_nf.n = tile_a + 32 * TILEDIM_X + midx;

        tile_e_b_in_nf.s = tile_b + TILEDIM_X + midx;
        tile_e_b_in_nf.w = tile_b + midy * TILEDIM_X + 1;
        tile_e_b_in_nf.e = tile_b + midy * TILEDIM_X + 32;
        tile_e_b_in_nf.n = tile_b + 32 * TILEDIM_X + midx;

        // get addresses of the inner edges of g_u for a block (used for copy between smem and gmem)
        g_E g_e_a, g_e_b;

        g_e_a.s = &g_ua[0] + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * RES;   // south row
        g_e_a.w = g_e_a.s; // ezzel lehet registert spórolni                            // west col
        g_e_a.e = g_e_a.s + BDIM - 1;                                                   // east col
        g_e_a.n = g_e_a.s + (BDIM - 1) * RES;                                           // north row

        g_e_b.s = &g_ub[0] + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * RES;
        g_e_b.w = g_e_b.s; // ezzel lehet registert spórolni
        g_e_b.e = g_e_b.s + BDIM - 1;
        g_e_b.n = g_e_b.s + (BDIM - 1) * RES;

        // get addresses for the aligned edges from the allocated memory
        // "slicing or distributing the allcated memory for each block"
        g_E g_e_a_new, g_e_b_new;

        int tmp_c = blockIdx.x * BDIM + blockIdx.y * BDIM * GDIM;   // offset to the whole allocated memory to get the (0,0) address
        g_e_a_new.s = &(g_grid_e_a.s[tmp_c]);                       // in block coordinate system, from grid coordinate system
        g_e_a_new.w = &(g_grid_e_a.w[tmp_c]);
        g_e_a_new.e = &(g_grid_e_a.e[tmp_c]);
        g_e_a_new.n = &(g_grid_e_a.n[tmp_c]);

        g_e_b_new.s = &(g_grid_e_b.s[tmp_c]);
        g_e_b_new.w = &(g_grid_e_b.w[tmp_c]);
        g_e_b_new.e = &(g_grid_e_b.e[tmp_c]);
        g_e_b_new.n = &(g_grid_e_b.n[tmp_c]);

        // calculate block outer edge memory addresses
        g_EC g_ec_a_new, g_ec_b_new;

        g_ec_a_new.s = g_e_a_new.n - BDIM * GDIM;
        g_ec_a_new.w = g_e_a_new.e - BDIM;
        g_ec_a_new.e = g_e_a_new.w + BDIM;
        g_ec_a_new.n = g_e_a_new.s + BDIM * GDIM;
        g_ec_a_new.sw = &(g_e_a_new.n[31]) - BDIM - BDIM * GDIM;
        g_ec_a_new.se = &(g_e_a_new.n[0]) + BDIM - BDIM * GDIM;
        g_ec_a_new.nw = &(g_e_a_new.s[31]) - BDIM + BDIM * GDIM;
        g_ec_a_new.ne = &(g_e_a_new.s[0]) + BDIM + BDIM * GDIM;
        g_ec_a_new.s = g_ec_a_new.s + tidx;
        g_ec_a_new.w = g_ec_a_new.w + tidx;
        g_ec_a_new.e = g_ec_a_new.e + tidx;
        g_ec_a_new.n = g_ec_a_new.n + tidx;

        g_ec_b_new.s = g_e_b_new.n - BDIM * GDIM;
        g_ec_b_new.w = g_e_b_new.e - BDIM;
        g_ec_b_new.e = g_e_b_new.w + BDIM;
        g_ec_b_new.n = g_e_b_new.s + BDIM * GDIM;
        g_ec_b_new.sw = &(g_e_b_new.n[31]) - BDIM - BDIM * GDIM;
        g_ec_b_new.se = &(g_e_b_new.n[0]) + BDIM - BDIM * GDIM;
        g_ec_b_new.nw = &(g_e_b_new.s[31]) - BDIM + BDIM * GDIM;
        g_ec_b_new.ne = &(g_e_b_new.s[0]) + BDIM + BDIM * GDIM;
        g_ec_b_new.s = g_ec_b_new.s + tidx;
        g_ec_b_new.w = g_ec_b_new.w + tidx;
        g_ec_b_new.e = g_ec_b_new.e + tidx;
        g_ec_b_new.n = g_ec_b_new.n + tidx;

        g_e_a_new.s = g_e_a_new.s + tidx;
        g_e_a_new.w = g_e_a_new.w + tidx;
        g_e_a_new.e = g_e_a_new.e + tidx;
        g_e_a_new.n = g_e_a_new.n + tidx;

        g_e_b_new.s = g_e_b_new.s + tidx;
        g_e_b_new.w = g_e_b_new.w + tidx;
        g_e_b_new.e = g_e_b_new.e + tidx;
        g_e_b_new.n = g_e_b_new.n + tidx;

        // get if the block is at an edge or at a corner of the simuation region
        // FOR CORNERS WILL INCLUDE THE TILE32.THREAD_RANK == 0 CONDITION !!!
        b_dir b_direction;
        b_direction.s_not = !(blockIdx.y == 0);                 // block is not at south edge
        b_direction.w_not = !(blockIdx.x == 0);                 // not west
        b_direction.e_not = !(blockIdx.x == gridDim.x - 1);     // not east
        b_direction.n_not = !(blockIdx.y == gridDim.y - 1);     // not north
        b_direction.sw_not = b_direction.s_not && b_direction.w_not && tile32.thread_rank() == 0;
        b_direction.se_not = b_direction.s_not && b_direction.e_not && tile32.thread_rank() == 0;
        b_direction.nw_not = b_direction.n_not && b_direction.w_not && tile32.thread_rank() == 0;
        b_direction.ne_not = b_direction.n_not && b_direction.e_not && tile32.thread_rank() == 0;

        // get if the THREAD is NEAR an edge of the simuation region
        // used for NO FLUX
        b_dir g_direction;
        g_direction.s_not = j == 1;
        g_direction.w_not = i == 1;
        g_direction.e_not = i == RES - 2;
        g_direction.n_not = j == RES - 2;
        __syncthreads();
        g_direction.sw_not = g_direction.s_not && g_direction.w_not;
        g_direction.se_not = g_direction.s_not && g_direction.e_not;
        g_direction.nw_not = g_direction.n_not && g_direction.w_not;
        g_direction.ne_not = g_direction.n_not && g_direction.e_not;

        // used for laplace
        int inner_inner = (i < RES - 1 && j < RES - 1 && i > 0 && j > 0); 

        // ----- Load Data -----
        copy_g_to_e(g_e_a, g_e_a_new, tile32, tidx, tidx_res);
        whole_grid.sync();
        copy_g_to_smem_new(g_ua, tile_a, ThreadOffset, g_ec_a_new, midx, midy, b_direction, tile32, tidx, tidx_res);
        copy_smem_to_r(tile_a, &r_ua, midx, midy);
        copy_g_to_r(g_v, &r_v, ThreadOffset);
        // idk why but works without sync
        //whole_grid.sync();
        
#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            
            // ----- Iteration step 1 -----
            iterate_without_laplace(r_ua, r_v, &r_ub, &r_v);

            __syncthreads();
            // ----- Laplace -----
            laplace(tile_act_a, r_ua, &r_ub, midx, midy, inner_inner);
            
            // ----- "Unload" Data ----- ----- No Flux Bc -----
            copy_r_to_smem_with_no_flux(r_ub, tile_b, tile_act_b, tile_e_b_in_nf, midx, midy, g_direction);
            // __syncthreads();

            copy_smem_to_g_boundary(tile_e_b_in, g_ub, ThreadOffset, g_e_b_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            whole_grid.sync();

            copy_g_to_smem_boundary(g_ub, tile_b, tile_e_b_out, ThreadOffset, g_ec_b_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            
            
            // ----- Iteration step 2 -----
            iterate_without_laplace(r_ub, r_v, &r_ua, &r_v);
            
            __syncthreads();
            // ----- Laplace -----
            laplace(tile_act_b, r_ub, &r_ua, midx, midy, inner_inner);
            
            // ----- "Unload" Data ----- ----- No Flux Bc -----
            copy_r_to_smem_with_no_flux(r_ua, tile_a, tile_act_a, tile_e_a_in_nf, midx, midy, g_direction);
            // __syncthreads();

            copy_smem_to_g_boundary(tile_e_a_in, g_ua, ThreadOffset, g_e_a_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            whole_grid.sync();

            copy_g_to_smem_boundary(g_ua, tile_a, tile_e_a_out, ThreadOffset, g_ec_a_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            
            
        }//iteration end
        
        copy_smem_to_g(tile_a, g_ua, ThreadOffset, midx, midy);
        
    }//grid inner end
}

__global__ void iterate_kernel_smem_exp(float* g_ua, float* g_ub, float* g_v, g_E g_grid_e_a, g_E g_grid_e_b, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    int tidx = threadIdx.x;
    // int tidy = threadIdx.y;
    int tidx_res = tidx * RES;
    int midx = threadIdx.x + 1;
    int midy = threadIdx.y + 1;
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    if (i < RES && j < RES)
    {
        /*
        __shared__ float tile_a_2[TILEDIM_Y][TILEDIM_X];
        __shared__ float tile_b_2[TILEDIM_Y][TILEDIM_X];
        float* tile_a;
        float* tile_b;
        tile_a = &tile_a_2[0][0];
        tile_b = &tile_b_2[0][0];
        */
        __shared__ float tile_a[TILEDIM_Y * TILEDIM_X];
        __shared__ float tile_b[TILEDIM_Y * TILEDIM_X];
        float* tile_act_a = tile_a + midy * TILEDIM_X + midx; // use register or use increase the calculation time ???
        float* tile_act_b = tile_b + midy * TILEDIM_X + midx;
        int ThreadOffset = i + j * RES;
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_ub;

        // ====== Memory address and thread position logics calculation ======
        // to reduce calculations in the iterations

        // ----- SMEM address calculations -----
        g_E tile_e_a_in, tile_e_b_in, tile_e_a_out, tile_e_b_out, tile_e_a_in_nf, tile_e_b_in_nf;

        tile_e_a_out.s = tile_a + midx;
        tile_e_a_out.w = tile_a + midx * TILEDIM_X;
        tile_e_a_out.e = tile_a + midx * TILEDIM_X + 33;
        tile_e_a_out.n = tile_a + 33 * TILEDIM_X + midx;

        tile_e_b_out.s = tile_b + midx;
        tile_e_b_out.w = tile_b + midx * TILEDIM_X;
        tile_e_b_out.e = tile_b + midx * TILEDIM_X + 33;
        tile_e_b_out.n = tile_b + 33 * TILEDIM_X + midx;

        tile_e_a_in.s = tile_a + TILEDIM_X + midx;
        tile_e_a_in.w = tile_a + midx * TILEDIM_X + 1;
        tile_e_a_in.e = tile_a + midx * TILEDIM_X + 32;
        tile_e_a_in.n = tile_a + 32 * TILEDIM_X + midx;

        tile_e_b_in.s = tile_b + TILEDIM_X + midx;
        tile_e_b_in.w = tile_b + midx * TILEDIM_X + 1;
        tile_e_b_in.e = tile_b + midx * TILEDIM_X + 32;
        tile_e_b_in.n = tile_b + 32 * TILEDIM_X + midx;

        // no flux copyig uses different indexing
        tile_e_a_in_nf.s = tile_a + TILEDIM_X + midx;
        tile_e_a_in_nf.w = tile_a + midy * TILEDIM_X + 1;
        tile_e_a_in_nf.e = tile_a + midy * TILEDIM_X + 32;
        tile_e_a_in_nf.n = tile_a + 32 * TILEDIM_X + midx;

        tile_e_b_in_nf.s = tile_b + TILEDIM_X + midx;
        tile_e_b_in_nf.w = tile_b + midy * TILEDIM_X + 1;
        tile_e_b_in_nf.e = tile_b + midy * TILEDIM_X + 32;
        tile_e_b_in_nf.n = tile_b + 32 * TILEDIM_X + midx;

        // ----- GMEM address calculations -----
        // get addresses of the inner edges of g_u for a block (used for copy between smem and gmem)
        g_E g_e_a, g_e_b;

        g_e_a.s = &g_ua[0] + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * RES;   // south row
        g_e_a.w = g_e_a.s; // ezzel lehet registert spórolni                            // west col
        g_e_a.e = g_e_a.s + BDIM - 1;                                                   // east col
        g_e_a.n = g_e_a.s + (BDIM - 1) * RES;                                           // north row

        g_e_b.s = &g_ub[0] + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * RES;
        g_e_b.w = g_e_b.s; // ezzel lehet registert spórolni
        g_e_b.e = g_e_b.s + BDIM - 1;
        g_e_b.n = g_e_b.s + (BDIM - 1) * RES;

        // get addresses for the aligned edges from the allocated memory
        // "slicing or distributing the allcated memory for each block"
        g_E g_e_a_new, g_e_b_new;

        int tmp_c = blockIdx.x * BDIM + blockIdx.y * BDIM * GDIM;   // offset to the whole allocated memory to get the (0,0) address
        g_e_a_new.s = &(g_grid_e_a.s[tmp_c]);                       // in block coordinate system, from grid coordinate system
        g_e_a_new.w = &(g_grid_e_a.w[tmp_c]);
        g_e_a_new.e = &(g_grid_e_a.e[tmp_c]);
        g_e_a_new.n = &(g_grid_e_a.n[tmp_c]);

        g_e_b_new.s = &(g_grid_e_b.s[tmp_c]);
        g_e_b_new.w = &(g_grid_e_b.w[tmp_c]);
        g_e_b_new.e = &(g_grid_e_b.e[tmp_c]);
        g_e_b_new.n = &(g_grid_e_b.n[tmp_c]);

        // calculate block outer edge memory addresses
        g_EC g_ec_a_new, g_ec_b_new;

        g_ec_a_new.s = g_e_a_new.n - BDIM * GDIM;
        g_ec_a_new.w = g_e_a_new.e - BDIM;
        g_ec_a_new.e = g_e_a_new.w + BDIM;
        g_ec_a_new.n = g_e_a_new.s + BDIM * GDIM;
        g_ec_a_new.sw = &(g_e_a_new.n[31]) - BDIM - BDIM * GDIM;
        g_ec_a_new.se = &(g_e_a_new.n[0]) + BDIM - BDIM * GDIM;
        g_ec_a_new.nw = &(g_e_a_new.s[31]) - BDIM + BDIM * GDIM;
        g_ec_a_new.ne = &(g_e_a_new.s[0]) + BDIM + BDIM * GDIM;
        g_ec_a_new.s = g_ec_a_new.s + tidx;
        g_ec_a_new.w = g_ec_a_new.w + tidx;
        g_ec_a_new.e = g_ec_a_new.e + tidx;
        g_ec_a_new.n = g_ec_a_new.n + tidx;

        g_ec_b_new.s = g_e_b_new.n - BDIM * GDIM;
        g_ec_b_new.w = g_e_b_new.e - BDIM;
        g_ec_b_new.e = g_e_b_new.w + BDIM;
        g_ec_b_new.n = g_e_b_new.s + BDIM * GDIM;
        g_ec_b_new.sw = &(g_e_b_new.n[31]) - BDIM - BDIM * GDIM;
        g_ec_b_new.se = &(g_e_b_new.n[0]) + BDIM - BDIM * GDIM;
        g_ec_b_new.nw = &(g_e_b_new.s[31]) - BDIM + BDIM * GDIM;
        g_ec_b_new.ne = &(g_e_b_new.s[0]) + BDIM + BDIM * GDIM;
        g_ec_b_new.s = g_ec_b_new.s + tidx;
        g_ec_b_new.w = g_ec_b_new.w + tidx;
        g_ec_b_new.e = g_ec_b_new.e + tidx;
        g_ec_b_new.n = g_ec_b_new.n + tidx;

        g_e_a_new.s = g_e_a_new.s + tidx;
        g_e_a_new.w = g_e_a_new.w + tidx;
        g_e_a_new.e = g_e_a_new.e + tidx;
        g_e_a_new.n = g_e_a_new.n + tidx;

        g_e_b_new.s = g_e_b_new.s + tidx;
        g_e_b_new.w = g_e_b_new.w + tidx;
        g_e_b_new.e = g_e_b_new.e + tidx;
        g_e_b_new.n = g_e_b_new.n + tidx;

        // ----- Thread/Block position identifier calculations -----
        // get if the block is at an edge or at a corner of the simuation region
        // FOR CORNERS WILL INCLUDE THE TILE32.THREAD_RANK == 0 CONDITION !!! this masks the rank>0 threads, 1 thread is active
        b_dir b_direction;
        b_direction.s_not = !(blockIdx.y == 0);                 // block is not at south edge
        b_direction.w_not = !(blockIdx.x == 0);                 // not west
        b_direction.e_not = !(blockIdx.x == gridDim.x - 1);     // not east
        b_direction.n_not = !(blockIdx.y == gridDim.y - 1);     // not north
        b_direction.sw_not = b_direction.s_not && b_direction.w_not && tile32.thread_rank() == 0;
        b_direction.se_not = b_direction.s_not && b_direction.e_not && tile32.thread_rank() == 0;
        b_direction.nw_not = b_direction.n_not && b_direction.w_not && tile32.thread_rank() == 0;
        b_direction.ne_not = b_direction.n_not && b_direction.e_not && tile32.thread_rank() == 0;

        // get if the THREAD is NEAR an edge of the simuation region
        // used for NO FLUX
        // the values from these register threads will be copied to the edges/corners in no flux bc
        b_dir g_direction;
        g_direction.s_not = j == 1;         // thread IS at south edge
        g_direction.w_not = i == 1;         // IS at west
        g_direction.e_not = i == RES - 2;   // IS at east
        g_direction.n_not = j == RES - 2;   // IS at north
        __syncthreads();
        g_direction.sw_not = g_direction.s_not && g_direction.w_not;    // thread IS at south west corner
        g_direction.se_not = g_direction.s_not && g_direction.e_not;    // ...
        g_direction.nw_not = g_direction.n_not && g_direction.w_not;
        g_direction.ne_not = g_direction.n_not && g_direction.e_not;

        // used for laplace
        int inner_inner = (i < RES - 1 && j < RES - 1 && i > 0 && j > 0);

        // ----- Load Data -----
        copy_g_to_e(g_e_a, g_e_a_new, tile32, tidx, tidx_res);
        whole_grid.sync();
        copy_g_to_smem_new(g_ua, tile_a, ThreadOffset, g_ec_a_new, midx, midy, b_direction, tile32, tidx, tidx_res);
        copy_smem_to_r(tile_a, &r_ua, midx, midy);
        copy_g_to_r(g_v, &r_v, ThreadOffset);
        // idk why but works without sync
        //whole_grid.sync();
        int tile32_meta_group_rank = tile32.meta_group_rank(); // did not change anything !!!
        float u_th;
        float r_L;
#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {

            // ----- Iteration step 1 -----
            // iterate_without_laplace(r_ua, r_v, &r_ub, &r_v);
            u_th = (r_v + b) * ra;
            r_v = r_v + dt * (r_ua - r_v);
            r_ub = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th);

            // ----- Laplace -----
            // laplace(tile_act_a, r_ua, &r_ub, midx, midy, inner_inner);
            if (inner_inner)
            {
                r_L = 0;

                // float* tile_act = &tile[0][0] + midy * TILEDIM_X + midx; // use register or use increase the calculation time ???
                r_L += -20 * r_ua;	// fma 
                r_L += 4 * tile_act_a[+1];
                r_L += 4 * tile_act_a[-1];

                tile_act_a -= TILEDIM_X;
                r_L += 4 * tile_act_a[0];
                r_L += 1 * tile_act_a[+1];
                r_L += 1 * tile_act_a[-1];

                tile_act_a += 2 * TILEDIM_X;
                r_L += 4 * tile_act_a[0];
                r_L += 1 * tile_act_a[+1];
                r_L += 1 * tile_act_a[-1];

                tile_act_a -= TILEDIM_X;
                r_L = ccc * r_L;

                r_ub += r_L;
            }

            // ----- "Unload" Data ----- ----- No Flux Bc -----
            // copy_r_to_smem_with_no_flux(r_ub, tile_b, tile_act_b, tile_e_b_in_nf, midx, midy, g_direction);
            // __syncthreads();
            *tile_act_b = r_ub;
            // ----- No Flux BC -----

            if (g_direction.s_not) { *tile_e_b_in_nf.s = r_ub; }
            if (g_direction.w_not) { *tile_e_b_in_nf.w = r_ub; }
            if (g_direction.e_not) { *tile_e_b_in_nf.e = r_ub; }
            if (g_direction.n_not) { *tile_e_b_in_nf.n = r_ub; }

            __syncthreads();

            if (g_direction.sw_not) { tile_b[36] = r_ub; }
            if (g_direction.se_not) { tile_b[67] = r_ub; }
            if (g_direction.nw_not) { tile_b[1121] = r_ub; }
            if (g_direction.ne_not) { tile_b[1152] = r_ub; }

            /*
            if (g_direction.sw_not) { tile_b[68] = r_ub; }
            if (g_direction.se_not) { tile_b[99] = r_ub; }
            if (g_direction.nw_not) { tile_b[2145] = r_ub; }
            if (g_direction.ne_not) { tile_b[2176] = r_ub; }
            */

            // copy_smem_to_g_boundary(tile_e_b_in, g_ub, ThreadOffset, g_e_b_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            switch (tile32_meta_group_rank)
            {
            case 0: // South Edge
                if (b_direction.s_not) { *g_e_b_new.s = *tile_e_b_in.s; }
                break;
            case 1: // West Edge
                if (b_direction.w_not) { *g_e_b_new.w = *tile_e_b_in.w; }
                break;
            case 2: // East Edge
                if (b_direction.e_not) { *g_e_b_new.e = *tile_e_b_in.e; }
                break;
            case 3: // North Edge
                if (b_direction.n_not) { *g_e_b_new.n = *tile_e_b_in.n; }
                break;
            default:
                break;
            }
            whole_grid.sync();

            // copy_g_to_smem_boundary(g_ub, tile_b, tile_e_b_out, ThreadOffset, g_ec_b_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            switch (tile32_meta_group_rank)
            {
            case 0: // South Edge
                if (b_direction.s_not) { *tile_e_b_out.s = *g_ec_b_new.s; }
                break;
            case 1: // West Edge
                if (b_direction.w_not) { *tile_e_b_out.w = *g_ec_b_new.w; }
                break;
            case 2: // East Edge
                if (b_direction.e_not) { *tile_e_b_out.e = *g_ec_b_new.e; }
                break;
            case 3: // North Edge
                if (b_direction.n_not) { *tile_e_b_out.n = *g_ec_b_new.n; }
                break;
            case 4: // South-West Corner
                if (b_direction.sw_not) { tile_b[0] = *g_ec_b_new.sw; }
                break;
            case 5: // South-East Corner
                if (b_direction.se_not) { tile_b[33] = *g_ec_b_new.se; }
                break;
            case 6: // North-West Corner
                if (b_direction.nw_not) { tile_b[1155] = *g_ec_b_new.nw; }
                break;
            case 7: // North-East Corner
                if (b_direction.ne_not) { tile_b[1188] = *g_ec_b_new.ne; }
                break;
            default:
                break;
            }
            __syncthreads();

            // ----- Iteration step 2 -----
            // iterate_without_laplace(r_ub, r_v, &r_ua, &r_v);
            u_th = (r_v + b) * ra;
            r_v = r_v + dt * (r_ub - r_v);
            r_ua = r_ub + dt * repsilon * r_ub * (1 - r_ub) * (r_ub - u_th);

            // ----- Laplace -----
            // laplace(tile_act_b, r_ub, &r_ua, midx, midy, inner_inner);
            if (inner_inner)
            {
                r_L = 0;

                // float* tile_act = &tile[0][0] + midy * TILEDIM_X + midx; // use register or use increase the calculation time ???
                r_L += -20 * r_ub;	// fma 
                r_L += 4 * tile_act_b[+1];
                r_L += 4 * tile_act_b[-1];

                tile_act_b -= TILEDIM_X;
                r_L += 4 * tile_act_b[0];
                r_L += 1 * tile_act_b[+1];
                r_L += 1 * tile_act_b[-1];

                tile_act_b += 2 * TILEDIM_X;
                r_L += 4 * tile_act_b[0];
                r_L += 1 * tile_act_b[+1];
                r_L += 1 * tile_act_b[-1];

                tile_act_b -= TILEDIM_X;
                r_L = ccc * r_L;

                r_ua += r_L;
            }

            // ----- "Unload" Data ----- ----- No Flux Bc -----
            // copy_r_to_smem_with_no_flux(r_ua, tile_a, tile_act_a, tile_e_a_in_nf, midx, midy, g_direction);
            // __syncthreads();
            *tile_act_a = r_ua;
            // ----- No Flux BC -----

            if (g_direction.s_not) { *tile_e_a_in_nf.s = r_ua; }
            if (g_direction.w_not) { *tile_e_a_in_nf.w = r_ua; }
            if (g_direction.e_not) { *tile_e_a_in_nf.e = r_ua; }
            if (g_direction.n_not) { *tile_e_a_in_nf.n = r_ua; }

            __syncthreads();

            if (g_direction.sw_not) { tile_a[36] = r_ua; }
            if (g_direction.se_not) { tile_a[67] = r_ua; }
            if (g_direction.nw_not) { tile_a[1121] = r_ua; }
            if (g_direction.ne_not) { tile_a[1152] = r_ua; }

            // copy_smem_to_g_boundary(tile_e_a_in, g_ua, ThreadOffset, g_e_a_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            switch (tile32_meta_group_rank)
            {
            case 0: // South Edge
                if (b_direction.s_not) { *g_e_a_new.s = *tile_e_a_in.s; }
                break;
            case 1: // West Edge
                if (b_direction.w_not) { *g_e_a_new.w = *tile_e_a_in.w; }
                break;
            case 2: // East Edge
                if (b_direction.e_not) { *g_e_a_new.e = *tile_e_a_in.e; }
                break;
            case 3: // North Edge
                if (b_direction.n_not) { *g_e_a_new.n = *tile_e_a_in.n; }
                break;
            default:
                break;
            }
            whole_grid.sync();

            // copy_g_to_smem_boundary(g_ua, tile_a, tile_e_a_out, ThreadOffset, g_ec_a_new, midx, midy, b_direction, tile32, tidx, tidx_res);
            switch (tile32_meta_group_rank)
            {
            case 0: // South Edge
                if (b_direction.s_not) { *tile_e_a_out.s = *g_ec_a_new.s; }
                break;
            case 1: // West Edge
                if (b_direction.w_not) { *tile_e_a_out.w = *g_ec_a_new.w; }
                break;
            case 2: // East Edge
                if (b_direction.e_not) { *tile_e_a_out.e = *g_ec_a_new.e; }
                break;
            case 3: // North Edge
                if (b_direction.n_not) { *tile_e_a_out.n = *g_ec_a_new.n; }
                break;
            case 4: // South-West Corner
                if (b_direction.sw_not) { tile_a[0] = *g_ec_a_new.sw; }
                break;
            case 5: // South-East Corner
                if (b_direction.se_not) { tile_a[33] = *g_ec_a_new.se; }
                break;
            case 6: // North-West Corner
                if (b_direction.nw_not) { tile_a[1155] = *g_ec_a_new.nw; }
                break;
            case 7: // North-East Corner
                if (b_direction.ne_not) { tile_a[1188] = *g_ec_a_new.ne; }
                break;
            default:
                break;
            }
            __syncthreads();

        }//iteration end

        copy_smem_to_g(tile_a, g_ua, ThreadOffset, midx, midy);
        __syncthreads();

    }//grid inner end
}