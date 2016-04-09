# -*- coding: utf-8 -*-

from __future__ import print_function

import pycuda.autoinit
import pycuda.driver as drv

from numpy import pi

from numpy import concatenate
from numpy import cumsum
from numpy import zeros
from numpy import column_stack
from numpy import sin
from numpy import cos

from numpy.random import random
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
      node_rad,
      spring_reject_rad,
      spring_attract_rad,
      inner_influence_rad,
      outer_influence_rad,
      nmax = 100000
    ):

    self.itt = 0

    self.threads = 512

    self.nmax = nmax
    self.size = size

    self.one = 1.0/size

    self.stp = stp
    self.spring_stp = spring_stp
    self.attract_stp = attract_stp
    self.reject_stp = reject_stp
    self.spring_attract_rad = spring_attract_rad
    self.spring_reject_rad = spring_reject_rad
    self.max_capacity = max_capacity
    self.node_rad = node_rad
    self.inner_influence_rad = inner_influence_rad
    self.outer_influence_rad = outer_influence_rad

    self.__init()
    self.__cuda_init()

  def __init(self):

    self.num = 0
    nmax = self.nmax

    self.xy = zeros((nmax, 2), npfloat)
    self.dxy = zeros((nmax, 2), npfloat)
    self.num_edges = zeros((nmax, 1), npint)

    self.cand_num = zeros((nmax, 1), npint)
    self.tmp = zeros((nmax, 1), npint)
    self.link_map = zeros((nmax, 1), npint)
    self.link_first = zeros((nmax, 1), npint)

    self.potential = zeros((nmax, 1), npint)

  def __cuda_init(self):

    from helpers import load_kernel

    self.cuda_step = load_kernel('modules/cuda/step.cu', 'step')

  def spawn(self, n, xy, dst, rad=0.4):

    num = self.num

    # from dddUtils.random import darts
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
    inds = self.potential[:num,0].nonzero()[0]
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

  def forces(self, t=None):

    self.itt += 1

    num = self.num
    xy = self.xy

    candidate_sets = kdt(xy[:num,:]).query_ball_point(
      xy[:num,:],
      self.outer_influence_rad
    )

    if t:
      t.t('kdt')

    self.cand_num[:num,0] = [len(c) for c in candidate_sets]
    self.link_map = concatenate(candidate_sets).astype(npint)
    self.link_first[1:num,0] = cumsum(self.cand_num[:num-1])

    if t:
      t.t('stage')

    blocks = (num)//self.threads + 1
    self.cuda_step(
      npint(num),
      drv.InOut(xy[:num,:]),
      drv.Out(self.num_edges[:num,:]),
      drv.In(self.link_first[:num,0]),
      drv.In(self.cand_num[:num,0]),
      drv.In(self.link_map),
      drv.In(self.potential[:num,0]),
      npfloat(self.stp),
      npfloat(self.reject_stp),
      npfloat(self.attract_stp),
      npfloat(self.spring_stp),
      npfloat(self.spring_reject_rad),
      npfloat(self.spring_attract_rad),
      npfloat(self.node_rad),
      block=(self.threads,1,1),
      grid=(blocks,1)
    )

    if t:
      t.t('cuda')

    self.potential[:num,0] = self.cand_num[:num,0] < self.max_capacity

    # print('mean max edges', mean(self.num_edges[:num,0]), max(self.num_edges[:num,0]))
    # print('mean max candidates', mean(self.cand_num[:num,0]), max(self.cand_num[:num,0]))
    # print()


