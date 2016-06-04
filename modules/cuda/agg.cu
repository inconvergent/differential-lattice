#define THREADS _THREADS_

__global__ void agg(
  const int n,
  const int nz,
  const int zone_leap,
  const float *xy,
  int *zone_num,
  int *zone_node
){
  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i>=n){
    return;
  }

  const int ii = 2*i;
  const int zi = (int) floor(xy[ii]*nz);
  const int zj = (int) floor(xy[ii+1]*nz);
  const int z = zi*nz + zj;

  const int o = atomicAdd(&zone_num[z], 1);
  zone_node[z*zone_leap+o] = i;
}

