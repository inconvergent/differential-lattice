#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function



def get_step(t=None):

  def step(dl):

    dl.step(t)

    return True

  return step

def get_wrap(dl, colors, export_steps=10):

  from numpy import pi
  from fn import Fn
  from modules.timers import named_sub_timers
  from dddUtils.ioOBJ import export_2d as export


  twopi = pi*2

  t = named_sub_timers('dl')

  fn = Fn(prefix='./res/')

  step = get_step(t)

  def wrap(render):

    res = step(dl)

    if dl.itt % export_steps != 0:
      return res

    print('itt', dl.itt, 'num', dl.num)
    t.p()
    num = dl.num


    arc = render.ctx.arc
    line_to = render.ctx.line_to
    move_to = render.ctx.move_to
    fill = render.ctx.fill
    stroke = render.ctx.stroke

    render.clear_canvas()
    render.set_line_width(dl.one)

    vertices, edges = dl.link_export()

    ## edges
    # render.ctx.set_source_rgba(*colors['cyan'])

    # render.ctx.set_source_rgba(*colors['light'])
    render.ctx.set_source_rgba(*colors['front'])
    # for a,b in edges:
      # move_to(*vertices[a,:])
      # line_to(*vertices[b,:])
      # stroke()

    # render.ctx.set_source_rgba(*colors['front'])
    ## dots
    # render.ctx.set_source_rgba(*colors['cyan'])
    for i in xrange(num):
      arc(vertices[i,0], vertices[i,1], 0.5*dl.node_rad, 0, twopi)
      fill()

    name = fn.name()
    render.write_to_png(name+'.png')
    # export('lattice', name+'.2obj', vertices, edges=edges)

    return res

  return wrap



def main():

  from numpy import array
  from modules.differentialLattice import DifferentialLattice
  from render.render import Animate
  from modules.helpers import spawn_circle

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.7],
    'cyan': [0,0.6,0.6,0.6],
    'light': [0,0,0,0.2],
  }

  threads = 512
  zone_leap = 512

  export_steps = 50

  size = 512*2
  one = 1.0/size

  stp = one*0.03
  spring_stp = 5
  reject_stp = 0.1
  cohesion_stp = 1.0

  max_capacity = 30

  node_rad = 1.5*one
  spring_reject_rad = node_rad*1.9
  spring_attract_rad = node_rad*2.0
  outer_influence_rad = 10.0*node_rad

  # link_ignore_rad = spring_attract_rad*2.0
  link_ignore_rad = outer_influence_rad # use a number larger than one to disable this effect entirely

  DL = DifferentialLattice(
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
    link_ignore_rad = link_ignore_rad,
    threads = threads,
    zone_leap = zone_leap,
    nmax = 50000000
  )

  spawn_circle(DL, 200, xy=array([[0.5,0.5]]), dst=node_rad*0.8, rad=0.01)

  render = Animate(size, colors['back'], colors['front'], get_wrap(DL, colors, export_steps))
  render.start()


if __name__ == '__main__':

  main()

