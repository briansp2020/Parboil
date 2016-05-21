/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"
#include "hip/hip_runtime.h"


__global__ void naive_kernel(grid_launch_parm lp, float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
    int i = hipThreadIdx_x;
    int j = hipBlockIdx_x+1;
    int k = hipBlockIdx_y+1;
    if(i>0)
    {
        Anext[Index3D (nx, ny, i, j, k)] =
            (A0[Index3D (nx, ny, i, j, k + 1)] +
             A0[Index3D (nx, ny, i, j, k - 1)] +
             A0[Index3D (nx, ny, i, j + 1, k)] +
             A0[Index3D (nx, ny, i, j - 1, k)] +
             A0[Index3D (nx, ny, i + 1, j, k)] +
             A0[Index3D (nx, ny, i - 1, j, k)])*c1
            - A0[Index3D (nx, ny, i, j, k)]*c0;
    }
}



