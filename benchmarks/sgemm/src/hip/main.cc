/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>
#include <iostream>
//#include "sgemm_kernel.cu"
#include <hip_runtime.h>

extern void regtileSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc );

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

extern "C"
void computeGold(float *, const float*, const float*, unsigned int, unsigned int, unsigned int);

int
main (int argc, char *argv[]) {

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  float *dA, *dB, *dC;
  size_t A_sz, B_sz, C_sz;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  pb_InitializeTimerSet(&timers);

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);
  // copy A to device memory
  A_sz = matArow*matAcol*sizeof(float);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  B_sz = matBrow*matBcol*sizeof(float);

  // allocate space for C
  C_sz = matArow*matBcol*sizeof(float);

  // CUDA memory allocation
  std::vector<float> matC(matArow*matBcol);
  hipMalloc((void**)&dA, A_sz);
  hipMalloc((void**)&dB, B_sz);
  hipMalloc((void**)&dC, C_sz);

  // Copy A and B^T into device memory
  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  hipMemcpy(dA, &matA.front(), A_sz, hipMemcpyHostToDevice); 
  hipMemcpy(dB, &matBT.front(), B_sz, hipMemcpyHostToDevice); 

  pb_SwitchToTimer( &timers, pb_TimerID_KERNEL );

  // Use standard sgemm interface
  regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, \
      dA, matArow, dB, matBcol, 0.0f, dC, matArow);

  if (params->outFile) {
    pb_SwitchToTimer( &timers, pb_TimerID_COPY );
    hipMemcpy(&matC.front(), dC, C_sz, hipMemcpyDeviceToHost);
    /* Write C to file */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile,
	matArow, matBcol, matC); 
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_KERNEL]));
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/GPUtime/1e9 << std::endl;
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  hipFree(dA);
  hipFree(dB);
  hipFree(dC);
  return 0;
}
