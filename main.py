#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function




def get_step(t=None):

  def step(dl):

    dl.forces(t)
    dl.cand_spawn(ratio=0.1)

    return True

  return step

def get_wrap(dl, colors):

  from numpy import pi
  from fn import Fn
  from modules.timers import named_sub_timers

  twopi = pi*2

  t = named_sub_timers('dl')

  xy = dl.xy

  fn = Fn(prefix='./res/', postfix='.png')

  step = get_step(t)

  def wrap(render):

    res = step(dl)

    if dl.itt % 100 != 0:
      return res

    print('itt', dl.itt, 'num', dl.num)
    t.p()
    num = dl.num
    render.set_line_width(dl.one)
    arc = render.ctx.arc
    fill = render.ctx.fill
    # stroke = render.ctx.stroke

    render.clear_canvas()

    # cand_flag = dl.potential[:num,0]

    render.ctx.set_source_rgba(*colors['light'])
    for i in xrange(num):

      # if dl.potential[i]:
        # render.ctx.set_source_rgba(*colors['cyan'])
        # arc(xy[i,0], xy[i,1], dl.node_rad*0.7, 0, twopi)
      # else:
      render.ctx.set_source_rgba(*colors['light'])
      arc(xy[i,0], xy[i,1], dl.node_rad, 0, twopi)

      fill()

    render.write_to_png(fn.name())

    return res

  return wrap



def main():

  from numpy import array
  from modules.differentialLattice import DifferentialLattice
  from render.render import Animate

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.7],
    'cyan': [0,0.6,0.6,0.6],
    'light': [0,0,0,0.6],
  }

  threads = 512

  size = 2000
  one = 1.0/size

  # stp = 5e-6
  stp = 5e-5
  spring_stp = 2
  reject_stp = 0.1

  max_capacity = 30

  node_rad = 1*one
  spring_reject_rad = node_rad*1.9
  spring_attract_rad = node_rad*2.0
  inner_influence_rad = 2.0*node_rad
  outer_influence_rad = 12.0*node_rad

  DL = DifferentialLattice(
    size,
    stp,
    spring_stp,
    reject_stp,
    max_capacity,
    node_rad,
    spring_reject_rad,
    spring_attract_rad,
    inner_influence_rad,
    outer_influence_rad,
    threads=threads,
    nmax=300000
  )

  DL.spawn(200, xy=array([[0.5,0.5]]),dst=node_rad*0.8, rad=0.01)

  render = Animate(size, colors['back'], colors['front'], get_wrap(DL, colors))
  render.start()


if __name__ == '__main__':

  main()

