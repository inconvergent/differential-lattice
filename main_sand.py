#!/usr/bin/python3
# -*- coding: utf-8 -*-

BACK = [1, 1, 1, 1]
FRONT = [0, 0, 0, 0.001]

def main():
  from fn import Fn
  from modules.differentialLattice import DifferentialLattice
  from modules.helpers import get_colors
  from modules.helpers import spawn_circle
  from numpy import array
  from numpy import cumsum
  from numpy import sqrt
  from numpy import linspace
  from numpy import sort
  from numpy import ones
  from numpy.random import random
  from numpy.random import seed
  from sand import Sand
  from numpy import pi
  from numpy import sin
  from numpy import cos
  TWOPI = pi*2.0


  fn = Fn(prefix='./res/')

  size = 512
  one = 1.0/size

  threads = 512
  zone_leap = 512
  grains =  20


  init_num = 20

  stp = one*0.03
  spring_stp = 5
  reject_stp = 0.1
  cohesion_stp = 1.0

  max_capacity = 30


  node_rad = 4*one
  spring_reject_rad = node_rad*1.9
  spring_attract_rad = node_rad*2.0
  outer_influence_rad = 10.0*node_rad
  link_ignore_rad = 0.5*outer_influence_rad

  colors = get_colors('../colors/black_t.gif')
  # colors = get_colors('../colors/ir.jpg')
  nc = len(colors)

  sand = Sand(size)
  sand.set_bg(BACK)
  sand.set_rgba(FRONT)

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
      link_ignore_rad,
      threads=threads,
      zone_leap=zone_leap,
      nmax=50000000
      )

  spawn_circle(DL, init_num, xy=array([[0.5, 0.5]]), dst=node_rad*0.8, rad=0.01)

  itt = 0
  while True:

    itt += 1
    DL.step()
    DL.spawn(ratio=0.1, age=1000)

    if not itt%20:
      print(('itt', DL.itt, 'num', DL.num))

    vertices, edges = DL.link_export()

    # sand.set_rgba(FRONT)
    # sand.paint_strokes(
    #     vertices[edges[:,0],:].astype('double'),
    #     vertices[edges[:,1],:].astype('double'),
    #     grains
    #     )

    # sand.paint_circles(
    #     vertices.astype('double'),
    #     random(len(vertices))*one*4.0,
    #     grains
    #     )

    # for k,(a,b) in enumerate(edges):
    #   w = a*nc+b
    #   rgba = colors[w%nc]+[0.001]
    #   sand.set_rgba(rgba)
    #   sand.paint_strokes(
    #       vertices[a:a+1,:].astype('double'),
    #       vertices[b:b+1,:].astype('double'),
    #       grains
    #       )


    n = 20
    for k, (x, y) in enumerate(vertices):
      rgba = colors[k%nc]+[0.0005]
      sand.set_rgba(rgba)
      o = ones((n, 2), 'float')
      o[:,0] *= x
      o[:,1] *= y
      r = (1.0-2.0*random(n))*4*one
      sand.paint_filled_circles(
          o,
          r,
          grains
          )

    if not itt%5:

      # vertices, edges = DL.link_export()
      # n = 1000
      # sand.set_bg(BACK)
      # seed(1)
      # for k, (x, y) in enumerate(vertices):
      #   rgba = colors[k%nc]+[0.005]
      #   sand.set_rgba(rgba)
      #   o = ones((n, 2), 'float')
      #   o[:,0] *= x
      #   o[:,1] *= y
      #   r = random()*one*3+cumsum(random(n)*random()*10)*one*0.002
      #   # r = sqrt(linspace(2.0, 10.0, n))*one
      #   # r = ones(n, 'float')*one*
      #   sand.paint_circles(
      #       o,
      #       r,
      #       grains
      #       )

      name = fn.name() + '.png'
      print(name)
      sand.write_to_png(name, 2)


if __name__ == '__main__':

  main()

