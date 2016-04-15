__global__ void step(
  int n,
  int nz,
  int zone_leap,
  float *xy,
  float *dxy,
  int *tmp,
  int *zone_num,
  int *zone_node,
  float stp,
  float reject_stp,
  float spring_stp,
  float spring_reject_rad,
  float spring_attract_rad,
  int max_capacity,
  float max_rad
){
  const int i = blockIdx.x*128 + threadIdx.x;

  if (i>=n){
    return;
  }

  const int ii = 2*i;
  const int zi = (int) floor(xy[ii]*nz);
  const int zj = (int) floor(xy[ii+1]*nz);

  float sx = 0;
  float sy = 0;
  float dx = 0;
  float dy = 0;
  float dd = 0;

  int jj;
  int aa;
  int zk;

  int edge_count = 0;
  int cand_count = 0;
  int total_count = 0;

  bool linked;

  int proximity[1000];

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

      if (linked){
        edge_count += 1;
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

  dxy[ii] = sx*stp;
  dxy[ii+1] = sy*stp;
  tmp[i] = cand_count;

}
