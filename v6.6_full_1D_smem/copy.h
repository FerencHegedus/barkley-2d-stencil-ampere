// v6.6
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

#define RES 256
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

// contains the addresses of the global u OUTER edges and corners for data "above" a block
struct g_EC
{
    float* s;
    float* sw;
    float* se;
    float* w;
    float* e;
    float* n;
    float* nw;
    float* ne;
};

// contains the addresses of the global u INNER edges for data "above" a block
struct g_E
{
    float* s;
    float* w;
    float* e;
    float* n;
};

// "booleans" giving information if the BLOCK is at an edge or at a corner of the simuation region
// used for smem-gmem copying. HERE IS NOT!!!
// "booleans" giving information if the EDGE is at an edge or at a corner of the simuation region
// used for NO FLUX BC. HERE IT IS NOT NOT!!!
struct b_dir
{
    int s_not;
    int sw_not;
    int se_not;
    int w_not;
    int e_not;
    int n_not;
    int nw_not;
    int ne_not;
};

using namespace std;
using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;

void text_to_array(float* u, float* v);
void array_to_text(float* u_sol, int res);

// copy from global memory to the registers
static __device__ __forceinline__ void copy_g_to_r(float* g, float* r, int ThreadOffset)
{
    *r = g[ThreadOffset];
}

// copy from registers to global memory
static __device__ __forceinline__ void copy_r_to_g(float* g, float r, int ThreadOffset)
{
    g[ThreadOffset] = r;
}


// copy from global memory to shared memory
static __device__ __forceinline__ void copy_g_to_smem_new(float* g, float tile[TILEDIM_Y * TILEDIM_X], int thread_offset_row, g_EC g_ec, int midx, int midy, b_dir b_direction, thread_block_tile<32> tile32, int tidx, int tidx_res)
{
    // copy inner 32x32
    tile[midy * TILEDIM_X + midx] = g[thread_offset_row];
    // copy 1x32 or 32x1 edges (4) and 1x1 corners (4)
    switch (tile32.meta_group_rank())
    {
    case 0: // South Edge
        if (b_direction.s_not) { tile[0 * TILEDIM_X + midx] = *g_ec.s; }
        break;
    case 1: // North Edge
        if (b_direction.n_not) { tile[33 * TILEDIM_X + midx] = *g_ec.n; }
        break;
    case 2: // West Edge
        if (b_direction.w_not) { tile[midx * TILEDIM_X + 0] = *g_ec.w; }
        break;
    case 3: // East Edge
        if (b_direction.e_not) { tile[midx * TILEDIM_X + 33] = *g_ec.e; }
        break;
    case 4: // South-West Corner
        if (b_direction.sw_not) { tile[0 * TILEDIM_X + 0] = *g_ec.sw; }
        break;
    case 5: // South-East Corner
        if (b_direction.se_not) { tile[0 * TILEDIM_X + 33] = *g_ec.se; }
        break;
    case 6: // North-West Corner
        if (b_direction.nw_not) { tile[33 * TILEDIM_X + 0] = *g_ec.nw; }
        break;
    case 7: // North-East Corner
        if (b_direction.ne_not) { tile[33 * TILEDIM_X + 33] = *g_ec.ne; }
        break;
    default:
        break;
    }
}

// copy from shared memory to register memory
static __device__ __forceinline__ void copy_smem_to_r(float tile[TILEDIM * (TILEDIM + PADDING)], float* r, int midx, int midy)
{
    *r = tile[midy * TILEDIM_X + midx];
}

// copy from register memory to shared memory
static __device__ __forceinline__ void copy_r_to_smem(float r, float tile[TILEDIM * (TILEDIM + PADDING)], int midx, int midy)
{
    tile[midy * TILEDIM_X + midx] = r;
}

// copy from register memory to shared memory
static __device__ __forceinline__ void copy_r_to_smem_with_no_flux(float r, float* tile, float* tile_act, g_E tile_e, int midx, int midy, b_dir g_direction)
{
    // tile[midy * TILEDIM_X + midx] = r;
    *tile_act = r;
    // ----- No Flux BC -----
    
    if (g_direction.s_not) { *tile_e.s = r; }
    if (g_direction.w_not) { *tile_e.w = r; }
    if (g_direction.e_not) { *tile_e.e = r; }
    if (g_direction.n_not) { *tile_e.n = r; }
    /*
    if (g_direction.s_not) { tile[1 * TILEDIM_X + midx] = r; }
    if (g_direction.w_not) { tile[midy * TILEDIM_X + 1] = r; }
    if (g_direction.e_not) { tile[midy * TILEDIM_X + 32] = r; }
    if (g_direction.n_not) { tile[32 * TILEDIM_X + midx] = r; }
    */
    __syncthreads();
    
    if (g_direction.sw_not) { tile[36] = r; }
    if (g_direction.se_not) { tile[67] = r; }
    if (g_direction.nw_not) { tile[1121] = r; }
    if (g_direction.ne_not) { tile[1152] = r; }
    /*
    if (g_direction.sw_not) { tile[1 * TILEDIM_X + 1] = r; }
    if (g_direction.se_not) { tile[1 * TILEDIM_X + 32] = r; }
    if (g_direction.nw_not) { tile[32 * TILEDIM_X + 1] = r; }
    if (g_direction.ne_not) { tile[32 * TILEDIM_X + 32] = r; }
    */
}

static __device__ __forceinline__ void copy_smem_to_g(float tile[TILEDIM_Y * TILEDIM_X], float* g, int ThreadOffset, int midx, int midy)
{
    g[ThreadOffset] = tile[midy * TILEDIM_X + midx];
}

// copy from global memory to shared memory at block boundaries
static __device__ __forceinline__ void copy_g_to_smem_boundary(float* g, float tile[TILEDIM_Y * TILEDIM_X], g_E tile_e, int thread_offset_row, g_EC g_ec, int midx, int midy, b_dir b_direction, thread_block_tile<32> tile32, int tidx, int tidx_res)
{
    // copy 1x32 or 32x1 edges (4) and 1x1 corners (4)
    switch (tile32.meta_group_rank())
    {
    case 0: // South Edge
        if (b_direction.s_not) { *tile_e.s = *g_ec.s; }
        break;
    case 1: // West Edge
        if (b_direction.w_not) { *tile_e.w = *g_ec.w; }
        break;
    case 2: // East Edge
        if (b_direction.e_not) { *tile_e.e = *g_ec.e; }
        break;
    case 3: // North Edge
        if (b_direction.n_not) { *tile_e.n = *g_ec.n; }
        break;
    case 4: // South-West Corner
        if (b_direction.sw_not) { tile[0] = *g_ec.sw; }
        break;
    case 5: // South-East Corner
        if (b_direction.se_not) { tile[33] = *g_ec.se; }
        break;
    case 6: // North-West Corner
        if (b_direction.nw_not) { tile[1155] = *g_ec.nw; }
        break;
    case 7: // North-East Corner
        if (b_direction.ne_not) { tile[1188] = *g_ec.ne; }
        break;
    default:
        break;
    }
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
}

// copy from global memory to shared memory at block boundaries
static __device__ __forceinline__ void copy_smem_to_g_boundary(g_E tile_e, float* g, int thread_offset_row, g_E g_e, int midx, int midy, b_dir b_direction, thread_block_tile<32> tile32, int tidx, int tidx_res)
{
    // copy 1x32 and 32x1 edges (4)
    switch (tile32.meta_group_rank())
    {
    case 0: // South Edge
        if (b_direction.s_not) { *g_e.s = *tile_e.s; }
        break;
    case 1: // West Edge
        if (b_direction.w_not) { *g_e.w = *tile_e.w; }
        break;
    case 2: // East Edge
        if (b_direction.e_not) { *g_e.e = *tile_e.e; }
        break;
    case 3: // North Edge
        if (b_direction.n_not) { *g_e.n = *tile_e.n; }
        break;
    /*
    case 0: // South Edge
        if (b_direction.s_not) { *g_e.s = tile[1 * TILEDIM_X + midx]; }
        break;
    case 1: // West Edge
        if (b_direction.w_not) { *g_e.w = tile[midx * TILEDIM_X + 1]; }
        break;
    case 2: // East Edge
        if (b_direction.e_not) { *g_e.e = tile[midx * TILEDIM_X + 32]; }
        break;
    case 3: // North Edge
        if (b_direction.n_not) { *g_e.n = tile[32 * TILEDIM_X + midx]; }
        break;
    */
    default:
        break;
    }
}

static __device__ void copy_g_to_e(g_E g_e_old, g_E g_e_new, thread_block_tile<32> tile32, int tidx, int tidx_res)
{
    switch (tile32.meta_group_rank())
    {
    case 0: // south edge
        *g_e_new.s = g_e_old.s[tidx];
        break;
    case 1: // west edge        
        *g_e_new.w = g_e_old.w[tidx_res];
        break;
    case 2: // east edge
        *g_e_new.e = g_e_old.e[tidx_res];
        break;
    case 3: // north edge
        *g_e_new.n = g_e_old.n[tidx];
        break;
    default:
        break;
    }
}

void text_to_array(float* u, float* v) {
    fstream newfile;

    newfile.open("u_fc.txt", ios::in); //open a file to perform read operation using file object
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
    newfile.open("v_fc.txt", ios::in); //open a file to perform read operation using file object
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
    newfile.open("u_sol_6_6.txt", ios::out);  // open a file to perform write operation using file object
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