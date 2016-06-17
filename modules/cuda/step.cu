#define THREADS _THREADS_

__device__ float dist(const float *a, const float *b, const int ii, const int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}

__device__ int calc_zones(const int za, const int zb, const int nz, int *Z){
  int num = 0;
  for (int a=max(za-1,0);a<min(za+2,nz);a++){
    for (int b=max(zb-1,0);b<min(zb+2,nz);b++){
      Z[num] = a*nz+b;
      num += 1;
    }
  }
  return num;
}

__device__ bool is_relative(
  const int ZN,
  const int *Z,
  const int zone_leap,
  const int *zone_num,
  const int *zone_node,
  const float link_ignore_rad,
  const float *xy,
  const int ii,
  const int jj
){

  int uu;
  int z;

  if (ii == jj){
    return false;
  }

  float dd = dist(xy, xy, ii, jj);

  if (dd>link_ignore_rad){
    return false;
  }

  for (int zk=0;zk<ZN;zk++){
    z = Z[zk];
    for (int k=0;k<zone_num[z];k++){

      uu = 2*zone_node[z*zone_leap+k];

      if (jj == uu){
        continue;
      }

      if (dd>max(dist(xy, xy, ii, uu), dist(xy, xy, jj, uu))){
        return false;
      }

    }
  }
  return true;
}

__global__ void step(
  const int n,
  const int nz,
  const int zone_leap,
  const float *xy,
  float *dxy,
  int *tmp,
  int *links,
  int *link_counts,
  const int *zone_num,
  const int *zone_node,
  const float stp,
  const float reject_stp,
  const float spring_stp,
  const float cohesion_stp,
  const float spring_reject_rad,
  const float spring_attract_rad,
  const int max_capacity,
  const float outer_influence_rad,
  const float link_ignore_rad
){
  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i>=n){
    return;
  }

  const int ii = 2*i;

  int Z[9];
  const int za = (int)floor(xy[ii]*nz);
  const int zb = (int)floor(xy[ii+1]*nz);
  const int ZN = calc_zones(za, zb, nz, Z);

  float sx = 0.0f;
  float sy = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  float dd = 0.0f;

  float mx = 0.0f;
  float my = 0.0f;
  float mm = 0.0f;


  int z;
  int jj;

  int link_count = 0;
  int cand_count = 0;
  bool linked = true;

  for (int zk=0;zk<ZN;zk++){
    z = Z[zk];
    for (int k=0;k<zone_num[z];k++){

      jj = 2*zone_node[z*zone_leap+k];

      if (jj==ii){
        continue;
      }

      linked = is_relative(
        ZN,
        Z,
        zone_leap,
        zone_num,
        zone_node,
        link_ignore_rad,
        xy,
        ii,
        jj
      );

      dx = xy[ii] - xy[jj];
      dy = xy[ii+1] - xy[jj+1];
      dd = sqrt(dx*dx + dy*dy);

      if (dd<=0.0f || dd>outer_influence_rad){
        continue;
      }

      cand_count += 1;

      dx /= dd;
      dy /= dd;

      mx += xy[jj];
      my += xy[jj+1];

      if (linked){
        links[10*i+link_count] = jj/2;
        link_count += 1;
        if (dd>spring_attract_rad){
          sx += -dx*spring_stp;
          sy += -dy*spring_stp;
        }
        else if(dd<spring_reject_rad){
          sx += dx*spring_stp;
          sy += dy*spring_stp;
        }
      }
      else{ // unlinked
        sx += dx*reject_stp;
        sy += dy*reject_stp;
      }
    }
  }

  mx = mx/(float)cand_count - xy[ii];
  my = my/(float)cand_count - xy[ii+1];
  mm = sqrt(mx*mx + my*my);

  mx *= -cohesion_stp/mm;
  my *= -cohesion_stp/mm;

  dxy[ii] = (sx+mx)*stp;
  dxy[ii+1] = (sy+my)*stp;
  tmp[i] = cand_count;
  link_counts[i] = link_count;

}

