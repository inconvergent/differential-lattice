#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function

from numpy import array
from numpy import pi


TWOPI = pi*2
PI = pi
HPI = pi*0.5

BACK = [1,1,1,1]
FRONT = [0,0,0,0.5]
CYAN = [0,0.6,0.6,0.5]
LIGHT = [0,0,0,0.05]


def step(dl):

  dl.structure()
  dl.cand_spawn(ratio=0.1)

  for i in xrange(10):
    dl.forces()

  return True



def main():

  from modules.differentialLattice import DifferentialLattice
  from render.render import Animate
  from fn import Fn


  size = 500
  one = 1.0/size

  stp = 4e-4
  spring_stp = 1.0
  reject_stp = 1.0

  max_capacity = 5
  min_capacity = 3

  cand_count_limit = 4
  capacity_cool_down = 15

  node_rad = 7*one
  disconnect_rad = 2*node_rad
  inner_influence_rad = 2*node_rad
  outer_influence_rad = 20*node_rad


  fn = Fn(prefix='./res/', postfix='.png')

  DL = DifferentialLattice(
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
    outer_influence_rad
  )

  DL.spawn(100, xy=array([[0.75,0.5]]),dst=node_rad*0.8, rad=0.1)
  DL.spawn(100, xy=array([[0.5,0.5]]),dst=node_rad*0.8, rad=0.3)

  def wrap(render):

    res = step(DL)

    render.set_line_width(DL.one)
    xy = DL.xy
    num = DL.num
    arc = render.ctx.arc
    stroke = render.ctx.stroke
    fill = render.ctx.fill
    move_to = render.ctx.move_to
    line_to = render.ctx.line_to

    render.clear_canvas()

    cand_flag = DL.cand_count[:num,0] < DL.cand_count_limit

    render.ctx.set_source_rgba(*LIGHT)
    for i in xrange(num):

      # nc = self.num_edges[i]
      # if nc>0:
        # t = tile(i, nc)
        # origin = xy[t,:]
        # stop = xy[self.edges[i,:nc],:]
        # self.render.sandstroke(column_stack([origin, stop]), grains=5)

      if cand_flag[i]:
        render.ctx.set_source_rgba(*CYAN)
      else:
        render.ctx.set_source_rgba(*FRONT)
      arc(xy[i,0], xy[i,1], 0.5*node_rad, 0, TWOPI)
      fill()

      render.ctx.set_source_rgba(*FRONT)
      nc = DL.num_edges[i]
      for j in xrange(nc):
        c = DL.edges[i,j]

        move_to(xy[i,0], xy[i,1])
        line_to(xy[c,0], xy[c,1])
        stroke()

    return res


  render = Animate(size, BACK, FRONT, wrap)
  render.start()


if __name__ == '__main__':

  main()

