# -*- coding: utf-8 -*-

from __future__ import print_function

import pycuda.autoinit
import pycuda.driver as drv

from numpy import pi

from numpy import zeros
from numpy import column_stack
from numpy import sin
from numpy import cos

from numpy.random import random

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

    self.zone_leap = 1000

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

    nz = int(1.0/self.outer_influence_rad)
    self.nz = nz
    self.nz2 = nz**2
    nmax = self.nmax

    self.xy = zeros((nmax, 2), npfloat)
    self.potential = zeros((nmax, 1), npint)
    self.zone_num = zeros((self.nz2, 1), npint)
    self.zone_node = zeros((self.nz2*self.zone_leap, 1), npint)

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

    blocks = (num)//self.threads + 1
    self.cuda_step(
      npint(num),
      npint(self.nz),
      npint(self.zone_leap),
      drv.InOut(xy[:num,:]),
      drv.Out(self.potential[:num,:]),
      drv.In(self.zone_num),
      drv.In(self.zone_node),
      npfloat(self.stp),
      npfloat(self.reject_stp),
      npfloat(self.attract_stp),
      npfloat(self.spring_stp),
      npfloat(self.spring_reject_rad),
      npfloat(self.spring_attract_rad),
      npfloat(self.node_rad),
      npint(self.max_capacity),
      npfloat(self.outer_influence_rad),
      block=(self.threads,1,1),
      grid=(blocks,1)
    )

    if t:
      t.t('cuda')

    print(max(self.potential[:num]))

