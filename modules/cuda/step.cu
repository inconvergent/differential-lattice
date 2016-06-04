#define THREADS _THREADS_
#define PROX _PROX_

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
  const float max_rad,
  const float link_ignore_rad,
  const int do_export
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
  int zk;

  int link_count = 0;
  int cand_count = 0;
  int total_count = 0;

  bool linked;

  int proximity[PROX];

  for (int a=max(zi-1,0);a<min(zi+2,nz);a++){
    for (int b=max(zj-1,0);b<min(zj+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){
        jj = 2*zone_node[zk*zone_leap+k];
        total_count += 1;
        dx = xy[ii] - xy[jj];
        dy = xy[ii+1] - xy[jj+1];
        dd = sqrt(dx*dx + dy*dy);
        if (dd<max_rad && dd>0.0f){
          proximity[cand_count] = jj/2;
          cand_count += 1;
        }
      }
    }
  }

  for (int k=0;k<cand_count;k++){

    jj = 2*proximity[k];

    dx = xy[ii] - xy[jj];
    dy = xy[ii+1] - xy[jj+1];
    dd = sqrt(dx*dx + dy*dy);

    linked = true;
    for (int l=0;l<cand_count;l++){
      aa = 2*proximity[l];
      if (dd>link_ignore_rad){
        linked = false;
        break;
      }
      if (dd>max(
          sqrt(powf(xy[ii] - xy[aa],2.0f) + powf(xy[ii+1] - xy[aa+1],2.0f)),
          sqrt(powf(xy[jj] - xy[aa],2.0f) + powf(xy[jj+1] - xy[aa+1],2.0f))
        )
      ){
        linked = false;
        break;
      }
    }

    if (dd>0.0f){

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

