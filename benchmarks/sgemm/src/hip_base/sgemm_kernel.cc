/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Kernel of dense matrix-matrix multiplication kernel.
 */

#include <iostream>
#include "hip/hip_runtime.h"

#define CHECK_ERROR(errorMessage) {                                    \
  hipError_t err = hipGetLastError();                                    \
  if( hipSuccess != err) {                                                \
    fprintf(stderr, "HIP error: %s in file '%s' in line %i : %s.\n",    \
	errorMessage, __FILE__, __LINE__, hipGetErrorString( err) );\
    exit(EXIT_FAILURE);                                                  \
  }                                                                        \
}

// Parameters of tile sizes
#define TILE_SZ 16 

__global__ void mysgemmNT(grid_launch_parm lp, const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
  float c = 0.0f;
  int m = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int n = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

  for (int i = 0; i < k; ++i) {
    float a = A[m + i * lda]; 
    float b = B[n + i * ldb];
    c += a * b;
  }
  C[m+n*ldc] = C[m+n*ldc] * beta + alpha * c;
}

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  
  // In this code we assume the matrix sizes are multiple of tile size
  if ((m%TILE_SZ) || (n%TILE_SZ)) {
    std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_SZ
      << "; n should be multiple of " << TILE_SZ << std::endl;
  }


  dim3 grid( m/TILE_SZ, n/TILE_SZ ), threads( TILE_SZ, TILE_SZ );
  hipLaunchKernel(HIP_KERNEL_NAME(mysgemmNT), dim3(grid), dim3(threads), 0, 0,  A, lda, B, ldb, C, ldc, k, alpha, beta);
  CHECK_ERROR("mySgemm");

}

