# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import any
from numpy import array
from numpy import logical_and
from numpy import logical_or
from numpy import tile
from numpy import max
from numpy import mean
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
      max_capacity,
      min_capacity,
      cand_count_limit,
      capacity_cool_down,
      node_rad,
      disconnect_rad,
      inner_influence_rad,
      outer_influence_rad,
      nmax = 1000000
    ):

    self.itt = 0


    self.nmax = nmax
    self.size = size
    self.one = 1.0/size

    self.stp = stp
    self.spring_stp = spring_stp
    self.reject_stp = reject_stp
    self.max_capacity = max_capacity
    self.min_capacity = min_capacity
    self.cand_count_limit = cand_count_limit
    self.capacity_cool_down = capacity_cool_down
    self.node_rad = node_rad
    self.disconnect_rad = disconnect_rad
    self.inner_influence_rad = inner_influence_rad
    self.outer_influence_rad = outer_influence_rad

    self.__init()

  def __init(self):

    self.num = 0
    nmax = self.nmax

    self.xy = zeros((nmax, 2), 'float')
    self.dxy = zeros((nmax, 2), 'float')
    self.cool_down = zeros((nmax, 1), 'int')
    self.edges = zeros((nmax, self.max_capacity), 'int')
    self.capacities = zeros((nmax, 1), 'int') + self.max_capacity
    self.num_edges = zeros((nmax, 1), 'int')

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
      offset = column_stack([cos(theta), sin(theta)])*self.node_rad*0.05
      self.xy[num:num+new_num,:] = new_xy+offset
      self.num += new_num
      return new_num

    return 0

  def is_connected(self, a, b):

    if self.num_edges[a] < 1 or self.num_edges[b] < 1:
      return False

    return any(self.edges[a,:self.num_edges[a]] == b)

  def connect(self, a, b):

    self.edges[a,self.num_edges[a]] = b
    self.edges[b,self.num_edges[b]] = a
    self.num_edges[a] += 1
    self.num_edges[b] += 1

  def disconnect(self, a, b):

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

    num = self.num

    self.tree = kdt(self.xy[:self.num,:])
    self.num_edges[:num,0] = 0

    candidate_sets = self.tree.query_ball_point(
      self.xy[:num,:],
      self.disconnect_rad
    )

    self.cand_count[:num,0] = [len(c) for c in candidate_sets]

    for i, cands in enumerate(candidate_sets):

      cands = [c for c in cands if c != i]
      rel = self.__is_relative_neighbor(i, cands)

      for j,c in enumerate(cands):
        if self.num_edges[i]>=self.max_capacity:
          break
        if self.num_edges[c]>=self.max_capacity:
          continue
        if self.is_connected(i, c):
          continue
        if rel[j]:
          self.connect(i, c)

    self.potential[:num,0] = self.num_edges[:num,0] < self.capacities[:num,0]

  def forces(self):

    num = self.num
    xy = self.xy
    edges = self.edges
    num_edges = self.num_edges
    potential = self.potential[:num,:]
    dxy = self.dxy

    dxy[:num,:] = 0

    candidate_sets = self.tree.query_ball_point(
      self.xy[:num,:],
      self.outer_influence_rad
    )

    for i in xrange(self.num):
      ne = num_edges[i]
      e = edges[i,:ne]

      # connected
      if ne>0:
        dx = xy[e,:]-xy[i,:]
        dd = norm(dx, axis=1)
        reject = dd<self.node_rad*1.8
        mid = logical_or(
          dd<self.node_rad*1.8,
          dd>self.node_rad*2.2
        )
        if any(mid):
          dx /= reshape(dd,(-1,1))
          dx[reject] *= -1
          dxy[i,:] += reshape(mean(dx[mid,:], axis=0), 2)*self.spring_stp

      # unconnected
      out = set([i]+list(e))
      cands = [c for c in candidate_sets[i] if c not in out]
      nc = len(cands)
      if nc>1:

        inv = ones(nc, 'float')
        if potential[i]:
          inv[potential[cands,0]] = -1

        dx = xy[i,:]-xy[cands,:]
        dd = norm(dx, axis=1)
        force = (self.outer_influence_rad-dd)/self.outer_influence_rad
        dx *= reshape(inv*force/dd,(-1,1))
        dxy[i,:] += reshape(mean(dx, axis=0), 2)*self.reject_stp

    xy[:num,:] += dxy[:num,:]*self.stp


