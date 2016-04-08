# -*- coding: utf-8 -*-

from __future__ import print_function

import pycuda.autoinit
import pycuda.driver as drv

from numpy import pi

from numpy import concatenate
from numpy import cumsum
# from numpy import any
# from numpy import array
# from numpy import logical_and
# from numpy import logical_or
# from numpy import logical_not
from numpy import max
# from numpy import mean
from numpy import sum
from numpy import arange
from numpy import zeros
from numpy import column_stack
from numpy import sin
from numpy import cos
from numpy import ones
# from numpy import reshape

from numpy.random import random
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree as kdt

from numpy import float32
from numpy import int32


TWOPI = pi*2
PI = pi
HPI = pi*0.5

npfloat = float32
npint = int32




class DifferentialLattice(object):

  def __init__(
      self,
      size,
      stp,
      spring_stp,
      reject_stp,
      attract_stp,
      max_capacity,
      cand_count_limit,
      node_rad,
      disconnect_rad,
      inner_influence_rad,
      outer_influence_rad,
      nmax = 1000000
    ):

    self.itt = 0

    self.threads = 256

    self.nmax = nmax
    self.size = size

    self.one = 1.0/size

    self.stp = stp
    self.spring_stp = spring_stp
    self.attract_stp = attract_stp
    self.reject_stp = reject_stp
    self.max_capacity = max_capacity
    self.cand_count_limit = cand_count_limit
    self.node_rad = node_rad
    self.disconnect_rad = disconnect_rad
    self.inner_influence_rad = inner_influence_rad
    self.outer_influence_rad = outer_influence_rad

    self.__init()
    self.__cuda_init()

  def __cuda_init(self):

    from pycuda.compiler import SourceModule

    mod = SourceModule('''
      __global__ void step(
        int n,
        float *xy,
        int *first,
        int *num,
        int *link,
        int *map,
        int *potential,
        float stp,
        float reject_stp,
        float attract_stp,
        float spring_stp,
        float node_rad
      ){
        const int i = blockIdx.x*256 + threadIdx.x;

        if (i>=n) {
          return;
        }

        float sx = 0;
        float sy = 0;

        float dx = 0;
        float dy = 0;
        float dd = 0;

        int f;
        int j;

        const int ii = 2*i;
        int jj;

        for (int k=0;k<num[i];k++){

          f = link[first[i]+k];
          j = map[first[i]+k];
          jj = 2*j;

          dx = xy[ii] - xy[jj];
          dy = xy[ii+1] - xy[jj+1];
          dd = sqrt(dx*dx + dy*dy);

          if (dd>0.0){

            dx /= dd;
            dy /= dd;

            if (f>0){
              // linked

              if (dd>node_rad*1.8){
                // attract
                sx += -dx*spring_stp;
                sy += -dy*spring_stp;
              }
              else if(dd<node_rad){
                // reject
                sx += dx*reject_stp;
                sy += dy*reject_stp;
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
            sx += dx;
            sy += dy;
          }

        }

        __syncthreads();

        xy[ii] = xy[ii] + sx*stp;
        xy[ii+1] = xy[ii+1] + sy*stp;

      }
    ''')

    self.cuda_step = mod.get_function('step')

  def __init(self):

    self.num = 0
    nmax = self.nmax

    self.xy = zeros((nmax, 2), npfloat)
    self.dxy = zeros((nmax, 2), npfloat)
    self.num_edges = zeros((nmax, 1), 'int')

    self.link_num = zeros((nmax, 1), npint)
    self.link_map = zeros((nmax, 1), npint)
    self.link_first = zeros((nmax, 1), npint)

    self.intensity = zeros((nmax, 1), npfloat)

    self.potential = zeros((nmax, 1), npint)
    self.cand_count = zeros((nmax, 1), 'int')

  def spawn(self, n, xy, dst, rad=0.4):

    num = self.num
    theta = random(n)*TWOPI
    new_xy = xy + column_stack([cos(theta), sin(theta)])*rad
    new_num = len(new_xy)
    if new_num>0:
      self.xy[num:num+new_num,:] = new_xy

    self.num += new_num
    return new_num

  def cand_spawn(self, ratio):

    num = self.num
    mask = self.cand_count[:num,0] < self.cand_count_limit

    inds = arange(num)[mask]
    selected = inds[random(len(inds))<ratio]

    new_num = len(selected)
    if new_num>0:
      new_xy = self.xy[selected,:]
      theta = random(new_num)*TWOPI
      offset = column_stack([cos(theta), sin(theta)])*self.node_rad*0.5
      self.xy[num:num+new_num,:] = new_xy+offset
      self.num += new_num
      return new_num

    return 0

  def __is_relative_neighbor(self, i, cands):

    if len(cands)<2:
      return ones(1, 'bool')

    xy = self.xy

    inds = cands + [i]
    dists = cdist(xy[inds,:], xy[inds,:])

    uv = dists[:-1,:-1]
    us = dists[:-1,-1]
    mas = max(uv, axis=0)

    return us<mas

  def structure(self):

    num = self.num
    num_edges = self.num_edges
    cand_count = self.cand_count
    rels = []

    candidate_sets = kdt(self.xy[:self.num,:]).query_ball_point(
      self.xy[:num,:],
      self.disconnect_rad
    )

    for i, cands in enumerate(candidate_sets):

      cand_count[i,0] = len(cands)
      cands = [c for c in cands if c != i]
      rel = self.__is_relative_neighbor(i, cands)
      rels.append(rel)
      num_edges[i,0] = sum(rel)

    self.link_num[:num,0] = self.cand_count[:num,0]
    self.link_map = concatenate(candidate_sets).astype(npint)
    self.link_link = concatenate(rels).astype(npint)
    self.link_first[1:num,0] = cumsum(self.link_num[:num-1])

    self.potential[:num,0] = self.num_edges[:num,0] < self.max_capacity

  def forces(self):

    self.itt += 1

    num = self.num
    xy = self.xy

    blocks = (num)//self.threads + 1
    self.cuda_step(
      npint(num),
      drv.InOut(xy[:num,:]),
      drv.In(self.link_first[:num,0]),
      drv.In(self.link_num[:num,0]),
      drv.In(self.link_link),
      drv.In(self.link_map),
      drv.In(self.potential[:num,0]),
      npfloat(self.stp),
      npfloat(self.reject_stp),
      npfloat(self.attract_stp),
      npfloat(self.spring_stp),
      npfloat(self.node_rad),
      block=(self.threads,1,1),
      grid=(blocks,1)
    )

