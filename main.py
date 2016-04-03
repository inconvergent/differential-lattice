#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function




def get_step():

  from modules.timers import named_sub_timers

  t = named_sub_timers('timer')

  def step(dl):

    t.start()

    dl.structure()
    dl.cand_spawn(ratio=0.1)
    t.t('structure')

    for i in xrange(10):
      dl.forces()
    t.t('forces')
    t.p()

    return True

  return step

def get_wrap(dl, colors):

  from numpy import pi
  twopi = pi*2

  xy = dl.xy
  cand_count = dl.cand_count
  edges = dl.edges

  step = get_step()

  def wrap(render):

    res = step(dl)

    num = dl.num
    render.set_line_width(dl.one)
    arc = render.ctx.arc
    stroke = render.ctx.stroke
    fill = render.ctx.fill
    move_to = render.ctx.move_to
    line_to = render.ctx.line_to

    render.clear_canvas()

    cand_flag = cand_count[:num,0] < dl.cand_count_limit

    render.ctx.set_source_rgba(*colors['light'])
    for i in xrange(num):

      # nc = self.num_edges[i]
      # if nc>0:
        # t = tile(i, nc)
        # origin = xy[t,:]
        # stop = xy[edges[i,:nc],:]
        # self.render.sandstroke(column_stack([origin, stop]), grains=5)

      if cand_flag[i]:
        render.ctx.set_source_rgba(*colors['cyan'])
      else:
        render.ctx.set_source_rgba(*colors['light'])
      arc(xy[i,0], xy[i,1], dl.one*2, 0, twopi)
      fill()

      render.ctx.set_source_rgba(*colors['front'])
      nc = dl.num_edges[i]
      for j in xrange(nc):
        c = edges[i,j]

        move_to(xy[i,0], xy[i,1])
        line_to(xy[c,0], xy[c,1])
        stroke()

    return res

  return wrap



def main():

  from numpy import array
  from modules.differentialLattice import DifferentialLattice
  from render.render import Animate
  from fn import Fn

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.5],
    'cyan': [0,0.6,0.6,0.3],
    'light': [0,0,0,0.3],
  }

  size = 1200
  one = 1.0/size

  # stp = 5e-6
  stp = 1e-4
  spring_stp = 1
  reject_stp = 1
  attract_stp = reject_stp

  max_capacity = 6

  cand_count_limit = 5
  capacity_cool_down = 15

  node_rad = 7.0*one
  disconnect_rad = 2.0*node_rad
  inner_influence_rad = 2.0*node_rad
  outer_influence_rad = 8.0*node_rad


  fn = Fn(prefix='./res/', postfix='.png')

  DL = DifferentialLattice(
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
    outer_influence_rad
  )

  DL.spawn(100, xy=array([[0.5,0.5]]),dst=node_rad*0.8, rad=0.1)
  DL.spawn(100, xy=array([[0.5,0.5]]),dst=node_rad*0.8, rad=0.3)

  render = Animate(size, colors['back'], colors['front'], get_wrap(DL, colors))
  render.start()


if __name__ == '__main__':

  main()

