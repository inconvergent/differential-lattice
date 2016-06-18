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

from timers import named_sub_timers


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
      nmax = 100000
      ):

    self.itt = 0

    self.threads = threads

    self.nmax = nmax
    self.size = size

    self.one = 1.0/size

    self.timer = named_sub_timers('lattice')

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
    self.node_rad = node_rad
    self.outer_influence_rad = outer_influence_rad
    self.link_ignore_rad = link_ignore_rad

    self.num = 0

    nz = int(0.5/self.outer_influence_rad)
    self.nz = nz
    self.nz2 = nz**2
    nmax = self.nmax

    self.xy = zeros((nmax, 2), npfloat)
    self.dxy = zeros((nmax, 2), npfloat)
    self.potential = zeros((nmax, 1), 'bool')
    self.cand_count = zeros((nmax, 1), npint)
    self.link_counts = zeros((nmax, 1), npint)
    self.links = zeros((10*nmax, 1), npint)
    self.age = zeros((self.nmax,1), npint)

    self.zone = zeros(nmax, npint)

    zone_map_size = self.nz2*64
    self.zone_node = zeros(zone_map_size, npint)

    self.zone_num = zeros(self.nz2, npint)

    self.__cuda_init()

  def __cuda_init(self):

    import pycuda.autoinit
    from helpers import load_kernel

    self.cuda_agg_count = load_kernel(
        'modules/cuda/agg_count.cu',
        'agg_count',
        subs = {'_THREADS_': self.threads}
        )

    self.cuda_agg = load_kernel(
        'modules/cuda/agg.cu',
        'agg',
        subs = {'_THREADS_': self.threads}
        )

    self.cuda_step = load_kernel(
        'modules/cuda/step.cu',
        'step',
        subs={
          '_THREADS_': self.threads
          }
        )

  def spawn(self, ratio, age=None):

    num = self.num

    self.potential[:num,0] = self.cand_count[:num,0]<self.max_capacity

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

  def __make_zonemap(self):

    from pycuda.driver import In
    from pycuda.driver import Out
    from pycuda.driver import InOut

    num = self.num
    xy = self.xy[:num, :]

    zone_num = self.zone_num
    zone = self.zone

    zone_num[:] = 0

    self.cuda_agg_count(
        npint(num),
        npint(self.nz),
        In(xy),
        InOut(zone_num),
        block=(self.threads,1,1),
        grid=(num//self.threads + 1,1)
        )

    zone_leap = zone_num[:].max()
    zone_map_size = self.nz2*zone_leap

    if zone_map_size>len(self.zone_node):
      print('resize, new zone leap: ', zone_map_size*2./self.nz2)
      self.zone_node = zeros(zone_map_size*2, npint)

    self.zone_node[:] = 0
    zone_num[:] = 0

    self.cuda_agg(
        npint(num),
        npint(self.nz),
        npint(zone_leap),
        In(xy),
        InOut(zone_num),
        InOut(self.zone_node),
        Out(zone[:num]),
        block=(self.threads,1,1),
        grid=(num//self.threads + 1,1)
        )

    return zone_leap, self.zone_node, zone_num

  def step(self):

    import pycuda.driver as drv

    self.itt += 1

    num = self.num
    xy = self.xy[:num,:]
    dxy = self.dxy[:num,:]

    self.timer.start()

    zone_leap, zone_node, zone_num = self.__make_zonemap()

    self.timer.t('zone')

    self.cuda_step(
        npint(num),
        npint(self.nz),
        npint(zone_leap),
        drv.In(xy),
        drv.Out(dxy),
        drv.Out(self.cand_count[:num,:]),
        drv.Out(self.links[:num*10,:]),
        drv.Out(self.link_counts[:num,:]),
        drv.In(zone_num),
        drv.In(zone_node),
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
        grid=(num//self.threads+1,1)
        )

    self.timer.t('step')

    xy += dxy

    self.timer.t('add')

    print(self.nz)

    self.timer.p()

