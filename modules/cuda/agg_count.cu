#define THREADS _THREADS_

__global__ void agg_count(
  const int n,
  const int nz,
  const float *xy,
  int *zone_num
){
  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i>=n){
    return;
  }

  const int ii = 2*i;
  const int zi = (int) floor(xy[ii]*nz);
  const int zj = (int) floor(xy[ii+1]*nz);
  const int z = zi*nz + zj;

  atomicAdd(&zone_num[z], 1);
}

