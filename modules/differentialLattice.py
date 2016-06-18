# -*- coding: utf-8 -*-

from __future__ import print_function


from numpy import pi
from numpy import zeros
from numpy import column_stack
from numpy import sin
from numpy import cos
from numpy.random import random

from numpy import float32 as npfloat
from numpy import int32 as npint
# from numpy import bool as npbool
from numpy import logical_and
from numpy import logical_not


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
      cohesion_stp,
      max_capacity,
      node_rad,
      spring_reject_rad,
      spring_attract_rad,
      outer_influence_rad,
      link_ignore_rad,
      threads = 256,
      zone_leap = 200,
      nmax = 100000
      ):

    self.itt = 0

    self.threads = threads

    self.nmax = nmax
    self.size = size

    self.one = 1.0/size

    assert spring_attract_rad<=outer_influence_rad
    assert spring_reject_rad<=outer_influence_rad
    assert link_ignore_rad<=outer_influence_rad

    self.stp = stp
    self.spring_stp = spring_stp
    self.reject_stp = reject_stp
    self.cohesion_stp = cohesion_stp
    self.spring_attract_rad = spring_attract_rad
    self.spring_reject_rad = spring_reject_rad
    self.max_capacity = max_capacity
    self.zone_leap = zone_leap
    self.node_rad = node_rad
    self.outer_influence_rad = outer_influence_rad
    self.link_ignore_rad = link_ignore_rad

    self.__init()
    self.__cuda_init()

  def __init(self):

    self.num = 0

    nz = int(0.5/self.outer_influence_rad)
    self.nz = nz
    self.nz2 = nz**2
    nmax = self.nmax

    self.xy = zeros((nmax, 2), npfloat)
    self.dxy = zeros((nmax, 2), npfloat)
    self.potential = zeros((nmax, 1), 'bool')
    self.tmp = zeros((nmax, 1), npint)
    self.link_counts = zeros((nmax, 1), npint)
    self.links = zeros((10*nmax, 1), npint)
    self.zone_num = zeros(self.nz2, npint)
    self.zone_node = zeros(self.nz2*self.zone_leap, npint)
    self.age = zeros((self.nmax,1), npint)

  def __cuda_init(self):

    import pycuda.autoinit
    from helpers import load_kernel

    self.cuda_agg = load_kernel(
        'modules/cuda/agg.cu',
        'agg',
        subs={'_THREADS_': self.threads}
        )
    self.cuda_step = load_kernel(
        'modules/cuda/step.cu',
        'step',
        subs={
          '_THREADS_': self.threads,
          '_PROX_': self.zone_leap
          }
        )

  def spawn(self, ratio, age=None):

    num = self.num
    self.potential[:num,0] = self.tmp[:num,0]<self.max_capacity

    inds = self.potential[:num,0].nonzero()[0]

    if age is not None:
      mask = self.age[inds,0]>self.itt-age
      inds = inds[mask]

    selected = inds[random(len(inds))<ratio]

    new_num = len(selected)
    if new_num>0:
      new_xy = self.xy[selected,:]
      theta = random(new_num)*TWOPI
      offset = column_stack([cos(theta), sin(theta)])*self.node_rad*0.5
      self.xy[num:num+new_num,:] = new_xy+offset
      self.num += new_num
      self.age[num:num+new_num] = self.itt

    if age is not None:
      self.decay(age)

    return 0

  def decay(self, age, ratio=0.01):

    num = self.num
    die = logical_and(
        self.age[:num,0]<self.itt-age,
        self.link_counts[:num,0]<1
        )
    alive = logical_not(logical_and(die, random(len(die))<ratio))

    inds = alive.nonzero()[0]

    new_num = len(inds)

    self.xy[:new_num,:] = self.xy[inds,:]
    self.num = new_num

  def link_export(self):

    from numpy import row_stack

    num = self.num
    links = self.links[:num*10, 0]

    edges = set()
    for i, c in enumerate(self.link_counts[:num,0]):
      for k in xrange(c):

        j = links[10*i+k]
        if i<j:
          lnk = (i, j)
        else:
          lnk = (j, i)

        if lnk not in edges:
          edges.add(lnk)

    return self.xy[:num,:], row_stack(list(edges))


  def step(self):

    import pycuda.driver as drv

    self.itt += 1

    num = self.num
    xy = self.xy
    dxy = self.dxy
    blocks = num//self.threads + 1

    self.zone_num[:] = 0

    self.cuda_agg(
        npint(num),
        npint(self.nz),
        npint(self.zone_leap),
        drv.In(xy[:num,:]),
        drv.InOut(self.zone_num),
        drv.InOut(self.zone_node),
        block=(self.threads,1,1),
        grid=(blocks,1)
        )

    self.cuda_step(
        npint(num),
        npint(self.nz),
        npint(self.zone_leap),
        drv.In(xy[:num,:]),
        drv.Out(dxy[:num,:]),
        drv.Out(self.tmp[:num,:]),
        drv.Out(self.links[:num*10,:]),
        drv.Out(self.link_counts[:num,:]),
        drv.In(self.zone_num),
        drv.In(self.zone_node),
        npfloat(self.stp),
        npfloat(self.reject_stp),
        npfloat(self.spring_stp),
        npfloat(self.cohesion_stp),
        npfloat(self.spring_reject_rad),
        npfloat(self.spring_attract_rad),
        npint(self.max_capacity),
        npfloat(self.outer_influence_rad),
        npfloat(self.link_ignore_rad),
        block=(self.threads,1,1),
        grid=(blocks,1)
        )

    xy[:num,:] += dxy[:num,:]

