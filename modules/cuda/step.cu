#define THREADS _THREADS_
#define PROX _PROX_

__device__ float dist(const float *a, const float *b, const int ii, const int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}

__device__ int get_candidates(
  const int nz,
  const int zi,
  const int zj,
  const int *zone_num,
  const int *zone_node,
  const int zone_leap,
  const float *xy,
  const float outer_influence_rad,
  const int ii,
  int *proximity
){

  int zk;
  int jj;
  float dd;

  int count = 0;

  for (int a=max(zi-1,0);a<min(zi+2,nz);a++){
    for (int b=max(zj-1,0);b<min(zj+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){
        jj = 2*zone_node[zk*zone_leap+k];
        dd = dist(xy, xy, ii, jj);
        if (dd<outer_influence_rad && dd>0.0f){
          proximity[count] = jj/2;
          count += 1;
        }
      }
    }
  }

  return count;
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
  const int zi = (int) floor(xy[ii]*nz);
  const int zj = (int) floor(xy[ii+1]*nz);

  float sx = 0.0f;
  float sy = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  float dd = 0.0f;

  float mx = 0.0f;
  float my = 0.0f;
  float mm = 0.0f;


  int jj;
  int aa;

  int link_count = 0;

  int proximity[PROX];
  int cand_count = get_candidates(nz, zi, zj, zone_num, zone_node, zone_leap,
      xy, outer_influence_rad, ii, proximity);

  bool linked;

  for (int k=0;k<cand_count;k++){

    jj = 2*proximity[k];

    dx = xy[ii] - xy[jj];
    dy = xy[ii+1] - xy[jj+1];
    dd = sqrt(dx*dx + dy*dy);

    if (dd<=0.0f){
      continue;
    }

    linked = true;
    for (int l=0;l<cand_count;l++){
      aa = 2*proximity[l];
      if (dd>link_ignore_rad){
        linked = false;
        break;
      }
      if (dd>max(dist(xy, xy, aa, ii), dist(xy, xy, jj, aa))){
        linked = false;
        break;
      }
    }

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

