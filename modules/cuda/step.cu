__global__ void step(
  int n,
  float *xy,
  int *num_edges,
  int *first,
  int *num,
  int *map,
  int *potential,
  float *intensity,
  float stp,
  float reject_stp,
  float attract_stp,
  float spring_stp,
  float spring_reject_rad,
  float spring_attract_rad,
  float node_rad,
  float alpha,
  float diminish
){
  const int i = blockIdx.x*512 + threadIdx.x;

  if (i>=n) {
    return;
  }

  float sx = 0.0f;
  float sy = 0.0f;

  float dx = 0.0f;
  float dy = 0.0f;
  float dd = 0.0f;

  int j;
  int jj;
  int aa;
  int count = 0;

  bool linked;

  float ja;
  float ia;

  float new_intensity = intensity[i];

  const int ii = 2*i;

  for (int k=0;k<num[i];k++){

    j = map[first[i]+k];
    jj = 2*j;

    dx = xy[ii] - xy[jj];
    dy = xy[ii+1] - xy[jj+1];
    dd = sqrt(dx*dx + dy*dy);

    linked = true;
    for (int l=0;l<num[i];l++){
      aa = 2*map[first[i]+l];
      ia = sqrt(powf(xy[ii] - xy[aa],2.0f) + powf(xy[ii+1] - xy[aa+1],2.0f));
      ja = sqrt(powf(xy[jj] - xy[aa],2.0f) + powf(xy[jj+1] - xy[aa+1],2.0f));
      if (dd>max(ia,ja)){
        linked = false;
        break;
      }
    }

    if (dd>0.0f){

      dx /= dd;
      dy /= dd;

      if (linked){
      /*if (dd<2*spring_attract_rad && linked){*/

        count += 1;

        new_intensity += alpha*intensity[j];

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
        if (potential[i]>0 && potential[j]>0){
          sx += -dx*attract_stp;
          sy += -dy*attract_stp;
        }
        else{
          sx += dx*reject_stp;
          sy += dy*reject_stp;
        }
      }
    }
  }

  __syncthreads();

  xy[ii] = xy[ii] + sx*stp*sqrt(intensity[i]);
  xy[ii+1] = xy[ii+1] + sy*stp*sqrt(intensity[i]);
  num_edges[i] = count;
  intensity[i] = diminish*new_intensity / (1.0 + alpha *(float)(count));

}
