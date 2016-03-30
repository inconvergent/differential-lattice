# -*- coding: utf-8 -*-

from __future__ import print_function

from render.render import Animate
from fn import Fn

from numpy import pi
from numpy import any
from numpy import array
from numpy import logical_and
from numpy import logical_or
from numpy import logical_not
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

BACK = [1,1,1,1]
FRONT = [0,0,0,0.5]
CYAN = [0,0.6,0.6,0.5]
LIGHT = [0,0,0,0.05]


class DifferentialLattice(object):

  def __init__(
      self,
      size,
      prefix = './res/',
      nmax = 1000000,
      back = BACK,
      front = FRONT
    ):

    self.itt = 0

    self.repeats = 10

    self.size = size
    self.one = 1.0/size

    self.stp = 4e-4
    self.spring_stp = 1
    self.reject_stp = 1

    self.nmax = nmax
    self.max_capacity = 5
    self.min_capacity = 3

    self.alive_age = 15

    self.node_rad = 7*self.one
    self.disconnect_rad = 2*self.node_rad
    self.inner_influence_rad = 2*self.node_rad
    self.outer_influence_rad = 20*self.node_rad

    self.fn = Fn(prefix=prefix, postfix='.png')
    self.render = Animate(size, back, front, self.step)
    self.render.set_line_width(self.one)

    self.__init()
    self.spawn(100, xy=array([[0.4,0.4]]),dst=self.node_rad*0.8, rad=0.1)
    self.spawn(100, xy=array([[0.6,0.6]]),dst=self.node_rad*0.8, rad=0.1)

    self.render.start()

  def __init(self):

    self.num = 0
    nmax = self.nmax

    self.xy = zeros((nmax, 2), 'float')
    self.dxy = zeros((nmax, 2), 'float')
    self.age = zeros((nmax, 1), 'int')
    self.active = ones((nmax, 1), 'bool')
    self.capacities = zeros((nmax, 1), 'int') + self.max_capacity

    self.edges = {}
    self.num_edges = zeros((nmax, 1), 'int')

    self.potential = zeros((nmax, 1), 'bool')
    self.cand_count = zeros((nmax, 1), 'int')

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

  def potential_spawn(self, ratio):

    num = self.num
    # potential = self.potential[:num,0]
    potential = self.cand_count[:num,0] < 5

    inds = arange(num)[potential]
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

    edges = self.edges

    if a not in edges:
      return False
    if b not in edges:
      return False

    return b in edges[a]

  def connect(self, a, b):

    edges = self.edges

    if a in edges:
      edges[a].add(b)
    else:
      edges[a] = set([b])

    if b in edges:
      edges[b].add(a)
    else:
      edges[b] = set([a])

    self.num_edges[a] += 1
    self.num_edges[b] += 1

  def disconnect(self, a, b):

    self.edges[a].remove(b)
    self.edges[b].remove(a)
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

  def __reset_edges(self):

    self.edges = {}
    self.num_edges[:self.num,0] = 0

    # for v1, ee in self.edges.iteritems():
      # self.num_edges[v1,0] = len(ee)

  def structure(self):

    num = self.num

    self.tree = kdt(self.xy[:self.num,:])
    self.__reset_edges()

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
    self.age[:num,0] += 1
    self.active[:num,0] = self.age[:num,0] < self.alive_age

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

      # connected

      e = set([])

      if i in edges:
        e = edges[i]
        dx = xy[list(e),:]-xy[i,:]
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
      out = e.union([i])
      cands = [c for c in candidate_sets[i] if c not in out]
      if len(cands)>1:

        if potential[i]:
          mask = potential[cands,0]
          active_cands = array(cands)[mask]
          dx = xy[i,:]-xy[active_cands,:]
          dd = norm(dx, axis=1)
          force = (self.outer_influence_rad-dd)/self.outer_influence_rad
          dx *= reshape(-force/dd,(-1,1))
          dxy[i,:] += reshape(mean(dx, axis=0), 2)*self.reject_stp

    xy[:num,:] += dxy[:num,:]*self.stp

  def step(self, render):

    self.itt += 1
    print('itt', self.itt, 'num', self.num)

    self.structure()

    self.show()
    self.render.write_to_png(self.fn.name())

    self.potential_spawn(ratio=0.01)

    for i in xrange(self.repeats):
      self.forces()

    return True

  def show(self):

    node_rad = self.node_rad
    xy = self.xy
    num = self.num
    potential = self.potential[:num,0]
    arc = self.render.ctx.arc
    stroke = self.render.ctx.stroke
    fill = self.render.ctx.fill
    move_to = self.render.ctx.move_to
    line_to = self.render.ctx.line_to

    self.render.clear_canvas()

    # cap = self.capacities[:num,0] < self.max_capacity
    cand_flag = self.cand_count[:num,0] < 5

    self.render.ctx.set_source_rgba(*FRONT)
    for v1,ee in self.edges.iteritems():

      # nc = self.num_edges[i]
      # if nc>0:
        # t = tile(i, nc)
        # origin = xy[t,:]
        # stop = xy[self.edges[i,:nc],:]
        # self.render.sandstroke(column_stack([origin, stop]), grains=5)

      if cand_flag[v1]:
        self.render.ctx.set_source_rgba(*CYAN)
      else:
        self.render.ctx.set_source_rgba(*FRONT)
      arc(xy[v1,0], xy[v1,1], 0.5*node_rad, 0, TWOPI)
      fill()

      self.render.ctx.set_source_rgba(*FRONT)
      for v2 in ee:
        move_to(xy[v1,0], xy[v1,1])
        line_to(xy[v2,0], xy[v2,1])
        stroke()

