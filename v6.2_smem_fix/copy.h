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
    //*r_v = g_v[ThreadOffset];
}

// copy from global memory to shared memory
static __device__ __forceinline__ void copy_g_to_smem(float* g, float tile[32][32], int ThreadOffset)
{
    tile[threadIdx.y][threadIdx.x] = g[ThreadOffset];
}

// copy from shared memory to register memory
static __device__ __forceinline__ void copy_smem_to_r(float tile[32][32], float* r)
{
    *r = tile[threadIdx.y][threadIdx.x];
}

// copy from register memory to shared memory
static __device__ __forceinline__ void copy_r_to_smem(float r, float tile[32][32])
{
    tile[threadIdx.y][threadIdx.x] = r;
}


static __device__ __forceinline__ void copy_smem_to_g(float tile[32][32], float* g, int ThreadOffset, int tidx, int tidy)
{
    g[ThreadOffset] = tile[tidy][tidx];
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
    newfile.open("u_sol_6_2.txt", ios::out);  // open a file to perform write operation using file object
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