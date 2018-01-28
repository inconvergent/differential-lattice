#!/usr/bin/python3
# -*- coding: utf-8 -*-


def get_coord(vertices, i, scale=1.0):
  a = float(vertices[i, 0]), float(vertices[i, 1])
  return tuple(map(lambda x: x*scale, a))


def write_svg(fn, vertices, edges):
  import svgwrite
  dwg = svgwrite.Drawing(fn+'.svg', profile='tiny')

  for a, b in edges:
    p1 = get_coord(vertices, a, scale=1000.0)
    p2 = get_coord(vertices, b, scale=1000.0)
    dwg.add(dwg.line(p1, p2, stroke=svgwrite.rgb(10, 10, 16, '%')))

  dwg.save()


def get_wrap(dl, colors, render_steps=10, export_steps=10):
  from fn import Fn
  # from dddUtils.ioOBJ import export_2d as export

  fn = Fn(prefix='./res/')

  def wrap(render):

    dl.step()

    dl.spawn(ratio=0.1, age=1000)

    if not dl.itt % export_steps:

      print(('itt', dl.itt, 'num', dl.num))
      num = dl.num

      render.clear_canvas()

      vertices, edges = dl.link_export()

      render.ctx.set_source_rgba(*colors['purple'])
      for a, b in edges:
        render.line(
            vertices[a, 0],
            vertices[a, 1],
            vertices[b, 0],
            vertices[b, 1]
            )

      render.ctx.set_source_rgba(*colors['front'])
      for i in range(num):
        render.circle(
            vertices[i, 0],
            vertices[i, 1],
            dl.node_rad*0.6,
            fill=True
            )

    if not dl.itt % export_steps:

      name = fn.name()
      render.write_to_png(name+'.png')
      # export('lattice', name+'.2obj', vertices, edges=edges)
      write_svg(name, vertices, edges)

    return True

  return wrap



def main():
  from numpy import array
  from modules.differentialLattice import DifferentialLattice
  from iutils.render import Animate
  from modules.helpers import spawn_circle

  colors = {
      'back': [1, 1, 1, 1],
      'front': [0, 0, 0, 0.7],
      'cyan': [0, 0.6, 0.6, 0.7],
      'purple': [0.6, 0.0, 0.6, 0.7],
      'light': [0, 0, 0, 0.2],
      }

  threads = 512
  zone_leap = 512

  size = 2048
  one = 1.0/size

  export_steps = 100
  render_steps = 100

  init_num = 20

  line_width = one*2.5

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
  wrap = get_wrap(DL, colors, render_steps, export_steps)
  render = Animate(size, colors['back'], colors['front'], wrap)

  render.set_line_width(line_width)
  render.start()


if __name__ == '__main__':

  main()

