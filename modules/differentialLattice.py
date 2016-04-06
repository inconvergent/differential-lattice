# -*- coding: utf-8 -*-

from __future__ import print_function

import pycuda.autoinit
import pycuda.driver as drv

from numpy import pi
from numpy import any
from numpy import array
from numpy import logical_and
from numpy import logical_or
from numpy import logical_not
from numpy import max
from numpy import mean
from numpy import sum
from numpy import arange
from numpy import zeros
from numpy import column_stack
from numpy import sin
from numpy import cos
from numpy import ones
from numpy import reshape

from numpy.random import random
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree as kdt

from numpy import float32
from numpy import int32


npfloat = float32
npint = int32




TWOPI = pi*2
PI = pi
HPI = pi*0.5


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
      capacity_cool_down,
      node_rad,
      disconnect_rad,
      inner_influence_rad,
      outer_influence_rad,
      nmax = 1000000
    ):

    self.itt = 0

    self.threads = 256

    from timers import named_sub_timers

    self.t = named_sub_timers('dl')

    self.nmax = nmax
    self.size = size

    self.one = 1.0/size

    self.stp = stp
    self.spring_stp = spring_stp
    self.attract_stp = attract_stp
    self.reject_stp = reject_stp
    self.max_capacity = max_capacity
    self.cand_count_limit = cand_count_limit
    self.capacity_cool_down = capacity_cool_down
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
        float *dest,
        int *first,
        int *num,
        int *link,
        int *map,
        int *potential,
        float *xy,
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
              if (potential[i]>0){
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

        dest[ii] = xy[ii] + sx*stp;
        dest[ii+1] = xy[ii+1] + sy*stp;

      }
    ''')

    self.cuda_step = mod.get_function('step')

  def __init(self):

    self.num = 0
    nmax = self.nmax

    self.xy = zeros((nmax, 2), npfloat)
    self.dxy = zeros((nmax, 2), npfloat)
    self.cool_down = zeros((nmax, 1), 'int')
    self.edges = zeros((nmax, self.max_capacity), 'int')
    self.capacities = zeros((nmax, 1), 'int') + self.max_capacity
    self.num_edges = zeros((nmax, 1), 'int')

    self.link_num = zeros((nmax, 1), npint)
    self.link_map = zeros((nmax, 1), npint)
    self.link_first = zeros((nmax, 1), npint)

    self.intensity = zeros((nmax, 1), npfloat)

    self.potential = zeros((nmax, 1), 'bool')
    self.cand_count = zeros((nmax, 1), 'int')

  def spawn(self, n, xy, dst, rad=0.4):

    # from dddUtils.random import darts
    num = self.num
    # new_xy = darts(n, 0.5, 0.5, rad, dst)
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

  def __is_connected(self, a, b):

    if self.num_edges[a] < 1 or self.num_edges[b] < 1:
      return False

    return any(self.edges[a,:self.num_edges[a]] == b)

  def __connect(self, a, b):

    self.edges[a,self.num_edges[a]] = b
    self.edges[b,self.num_edges[b]] = a
    self.num_edges[a] += 1
    self.num_edges[b] += 1

  def __disconnect(self, a, b):

    na = self.num_edges[a]
    nb = self.num_edges[b]

    for i,e in enumerate(self.edges[a,:na]):
      if e == b:
       self.edges[a,i] = self.edges[a,na-1]

    for i,e in enumerate(self.edges[b,:nb]):
      if e == a:
       self.edges[b,i] = self.edges[b,nb-1]

    self.num_edges[a] -= 1
    self.num_edges[b] -= 1

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

    from numpy import concatenate
    from numpy import cumsum

    num = self.num

    self.tree = kdt(self.xy[:self.num,:])
    self.num_edges[:num,0] = 0

    candidate_sets = self.tree.query_ball_point(
      self.xy[:num,:],
      self.disconnect_rad
    )

    self.cand_count[:num,0] = [len(c) for c in candidate_sets]
    rels = []

    for i, cands in enumerate(candidate_sets):

      cands = [c for c in cands if c != i]
      rel = self.__is_relative_neighbor(i, cands)
      rels.append(rel)

      for j,c in enumerate(cands):
        if self.num_edges[i]>=self.max_capacity:
          break
        if self.num_edges[c]>=self.max_capacity:
          continue
        if self.__is_connected(i, c):
          continue
        if rel[j]:
          self.__connect(i, c)

    self.link_num[:num,0] = self.cand_count[:num,0]
    self.link_map = concatenate(candidate_sets).astype(npint)
    self.link_link = concatenate(rels).astype(npint)
    self.link_first[1:num,0] = cumsum(self.link_num[:num-1])

    self.potential[:num,0] = self.num_edges[:num,0] < self.capacities[:num,0]

  def forces(self):

    self.itt += 1

    num = self.num
    xy = self.xy

    blocks = (num)//self.threads + 1
    self.cuda_step(
      npint(num),
      drv.Out(xy[:num,:]),
      drv.In(self.link_first[:num,0]),
      drv.In(self.link_num[:num,0]),
      drv.In(self.link_link),
      drv.In(self.link_map),
      drv.In(self.potential[:num,0].astype(npint)),
      drv.In(xy[:num,:]),
      npfloat(self.stp),
      npfloat(self.reject_stp),
      npfloat(self.attract_stp),
      npfloat(self.spring_stp),
      npfloat(self.node_rad),
      block=(self.threads,1,1),
      grid=(blocks,1)
    )

