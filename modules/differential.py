# -*- coding: utf-8 -*-

from __future__ import print_function

from render.render import Animate
from fn import Fn

from scipy.spatial import cKDTree as kdt

from numpy import pi
from numpy import any
from numpy import logical_and
from numpy import array
from numpy import concatenate
from numpy import arange
from numpy import zeros
from numpy import ones
from numpy import reshape
from numpy.random import random

from dddUtils.random import darts


TWOPI = pi*2
HPI = pi*0.5

BACK = [1,1,1,1]
FRONT = [0,0,0,0.5]
CYAN = [0,1,1,0.5]
LIGHT = [0,0,0,0.05]


class Differential(object):

  def __init__(
      self,
      size,
      prefix = './res/',
      nmax = 1000000,
      back = BACK,
      front = FRONT
    ):

    self.itt = 0

    self.size = size
    self.one = 1.0/size

    self.stp = 1e-4
    self.reject_ratio = 1e-3
    self.attract_ratio = 1e-3

    self.nmax = nmax
    self.max_capacity = 6
    self.min_capacity = 3

    self.capacity_cool_down = 100

    self.node_rad = 3*self.one
    self.disconnect_rad = 2*self.node_rad
    self.inner_influence_rad = 2*self.node_rad
    self.outer_influence_rad = 4*self.node_rad

    self.fn = Fn(prefix=prefix, postfix='.png')
    self.render = Animate(size, back, front, self.step)
    self.render.set_line_width(self.one)

    self.__init()
    self.random_init(1000)

    self.render.start()

  def __init(self):

    self.num = 0
    nmax = self.nmax

    self.xy = zeros((nmax, 2), 'float')
    self.dxy = zeros((nmax, 2), 'float')
    self.cool_down = zeros((nmax, 1), 'int')
    self.edges = zeros((nmax, self.max_capacity), 'int')
    self.capacities = zeros((nmax, 1), 'int')
    self.num_edges = zeros((nmax, 1), 'int')

  def random_init(self, n):

    num = self.num

    new_xy = darts(n, 0.5, 0.5, 0.1, self.node_rad*1.5)
    new_num = len(new_xy)
    self.xy[num:new_num,:] = new_xy
    self.capacities[num:new_num, 0] = self.max_capacity
    # self.cool_down[num:new_num] = 0
    # self.num_edges[num:new_num] = 0

    self.num += new_num
    return new_num

  def make_tree(self):

    self.tree = kdt(self.xy[:self.num,:])

  def is_connected(self, a, b):

    if self.num_edges[a] < 1 or self.num_edges[b] < 1:
      return False

    return any(self.edges[a,:self.num_edges[a]] == b)

  def connect(self, a, b):

    self.edges[a,self.num_edges[a]] = b
    self.edges[b,self.num_edges[b]] = a
    self.num_edges[a] += 1
    self.num_edges[b] += 1

  def structure(self):

    num = self.num

    potentials = self.num_edges[:num,0] < self.capacities[:num,0]
    potentials_inds = arange(num)[potentials]
    candidate_sets = self.tree.query_ball_point(self.xy[potentials_inds,:], self.disconnect_rad)

    for i, cands in zip(potentials_inds, candidate_sets):

      for c in cands:
        if i == c:
          continue
        if self.num_edges[i]>=self.max_capacity:
          break
        if self.num_edges[c]>=self.max_capacity:
          continue
        if potentials[c] and not self.is_connected(i, c):
          print('do connect', i, c)
          self.connect(i, c)

    # reduced = self.capacities[:num,0] < self.max_capacity
    self.cool_down[potentials_inds,0] += 1
    cool = logical_and(
      self.cool_down[potentials_inds,0] > self.capacity_cool_down,
      self.capacities[potentials_inds,0] > self.min_capacity
    )

    self.capacities[potentials_inds[cool]] -= 1
    self.cool_down[potentials_inds[cool],0] = 0


  # def forces(self):

    # num = self.num
    # self.dxy[:num, :]
    # tree = self.tree

  def step(self, render):

    self.itt += 1
    print(self.itt)
    self.make_tree()
    self.structure()
    # self.forces()
    self.show()
    # self.render.write_to_png(self.fn.name())

    return True

  def show(self):


    node_rad = self.node_rad
    xy = self.xy
    arc = self.render.ctx.arc
    stroke = self.render.ctx.stroke
    fill = self.render.ctx.fill
    move_to = self.render.ctx.move_to
    line_to = self.render.ctx.line_to
    self.render.clear_canvas()

    # cap_flag = self.capacities[:self.num,0] < self.max_capacity
    potentials_flag = self.num_edges[:self.num,0] < self.capacities[:self.num,0]

    for i in xrange(self.num):
      if potentials_flag[i]:
        self.render.ctx.set_source_rgba(*CYAN)
        arc(xy[i,0], xy[i,1], node_rad, 0, TWOPI)
        fill()
        self.render.ctx.set_source_rgba(*FRONT)

      arc(xy[i,0], xy[i,1], node_rad, 0, TWOPI)
      stroke()

    self.render.ctx.set_source_rgba(*FRONT)
    for i in xrange(self.num):

      nc = self.num_edges[i]
      for j in xrange(nc):
        c = self.edges[i,j]
        move_to(xy[i,0], xy[i,1])
        line_to(xy[c,0], xy[c,1])

      stroke()

