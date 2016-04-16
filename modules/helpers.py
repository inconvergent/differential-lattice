#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function

def load_kernel(fn, name, subs={}):

  from pycuda.compiler import SourceModule

  with open(fn, 'r') as f:
    kernel = f.read()

  for k,v in subs.iteritems():
    kernel = kernel.replace(k, str(v))

  mod = SourceModule(kernel)
  return mod.get_function(name)

def spawn_darts(dl, n, xy, dst, rad=0.4):

  from dddUtils.random import darts
  num = dl.num

  new_xy = darts(n, 0.5, 0.5, rad, dst)
  new_num = len(new_xy)
  if new_num>0:
    dl.xy[num:num+new_num,:] = new_xy

  dl.num += new_num
  return new_num

def spawn_circle(dl, n, xy, dst, rad=0.4):

  from numpy.random import random
  from numpy import pi
  from numpy import column_stack
  from numpy import sin
  from numpy import cos

  num = dl.num
  theta = random(n)*2*pi
  new_xy = xy + column_stack([cos(theta), sin(theta)])*rad
  new_num = len(new_xy)
  if new_num>0:
    dl.xy[num:num+new_num,:] = new_xy

  dl.num += new_num
  return new_num

