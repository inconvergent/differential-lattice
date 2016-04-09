__global__ void step(
  int n,
  float *xy,
  int *num_edges,
  int *first,
  int *num,
  int *map,
  int *potential,
  float stp,
  float reject_stp,
  float attract_stp,
  float spring_stp,
  float node_rad
){
  const int i = blockIdx.x*512 + threadIdx.x;

  if (i>=n) {
    return;
  }

  float sx = 0;
  float sy = 0;

  float dx = 0;
  float dy = 0;
  float dd = 0;

  int j;
  int jj;
  int aa;
  int count = 0;

  bool linked;

  float vu_dst = 0;
  float ja;
  float ia;

  const int ii = 2*i;

  for (int k=0;k<num[i];k++){

    j = map[first[i]+k];
    jj = 2*j;

    dx = xy[ii] - xy[jj];
    dy = xy[ii+1] - xy[jj+1];
    dd = sqrt(dx*dx + dy*dy);


    // TODO: there is something seriously wrong here
    linked = true;
    for (int l=0;l<num[i];l++){
      aa = 2*map[first[i]+l];
      ia = sqrt(powf(xy[ii] - xy[aa],2.0) + powf(xy[ii+1] - xy[aa+1],2.0));
      ja = sqrt(powf(xy[jj] - xy[aa],2.0) + powf(xy[jj+1] - xy[aa+1],2.0));
      if (dd>max(ia,ja)){
        linked = false;
        break;
      }
    }

    if (dd>0.0){

      dx /= dd;
      dy /= dd;

      if (linked){
      //if ( dd<=3){
        // linked
        count += 1;

        if (dd>node_rad*2.0){
          // attract
          sx += -dx*spring_stp;
          sy += -dy*spring_stp;
        }
        else if(dd<node_rad*1.8){
          // reject
          sx += dx*spring_stp;
          sy += dy*spring_stp;
        }
      }
      else{
        // unlinked
        if (potential[i]>0 && potential[j]>0){
          // attract
          sx += -dx*attract_stp;
          sy += -dy*attract_stp;
        }
        else{
          // reject
          sx += dx*reject_stp;
          sy += dy*reject_stp;
        }
      }
    }

  }

  __syncthreads();

  xy[ii] = xy[ii] + sx*stp;
  xy[ii+1] = xy[ii+1] + sy*stp;
  num_edges[i] = count;

}
