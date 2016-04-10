#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from operator import itemgetter
from numpy import array
from numpy import concatenate
from numpy import logical_not
from numpy import zeros
from numpy import reshape
from numpy import int
from numpy.linalg import norm


__ALL__ = ['Zonemap']


class Zonemap(object):

  def __init__(self, size, rad, xy):

    self.one = 1.0/size
    self.zonewidth = 2.*(rad/self.one)
    self.num_zones = int(size/self.zonewidth)
    self.zn = [set() for i in xrange((self.num_zones+2)**2)]
    self.nz = zeros(len(xy), 'int')
    self.xy = xy
    self.num = 0

    print('zonewidth', self.zonewidth)
    print('num_zones', self.num_zones)

  def __nearby_zones(self, x):

    num_zones = self.num_zones
    i = 1+int(x[0]*num_zones)
    j = 1+int(x[1]*num_zones)
    ij = array([i-1,i,i+1,i-1,i,i+1,i-1,i,i+1])*num_zones+\
         array([j+1,j+1,j+1,j,j,j,j-1,j-1,j-1])
    return ij

  def near_zone_inds(self, x):

    zs = self.__nearby_zones(x)
    its = itemgetter(*zs)(self.zn)
    return concatenate([list(s) for s in its]).astype('int')

  def get_z(self, x):

    num_zones = self.num_zones
    i = 1+(x[:,0]*num_zones).astype('int')
    j = 1+(x[:,1]*num_zones).astype('int')
    z = i*num_zones+j
    return z

  def sphere_vertices(self, x, rad):

    inds = self.near_zone_inds(x)
    dd = norm(x - self.xy[inds,:], axis=1)
    return inds[dd<rad]

  def __remove(self, i):

    s = self.nz[i]
    self.zn[s].remove(i)
    self.num -= 1

  def update(self):

    zz = self.get_z(self.xy[:self.num,:])
    change_mask = zz == self.nz
    changed_inds = logical_not(change_mask).nonzero()[0]
    for i in changed_inds:
      self.__remove(i)
      self.add(i)

    return len(changed_inds)

  def add(self, i):

    z = self.get_z(reshape(self.xy[i,:], (-1,2)))
    self.zn[z].add(i)
    self.nz[i] = z
    self.num += 1

def main():

  from numpy.random import random

  size = 800
  rad = 0.01
  n = 100

  xy = random(size=(n,2))
  Z = Zonemap(size, rad, xy)

  for i in xrange(n):
    Z.add(i)

  zz = Z.get_z(random(size=(n, 2)))
  print(zz)

  for x in xy:
    inds = Z.near_zone_inds(x)
    print(inds)

  for x in random(size=(n, 2)):
    inds = Z.near_zone_inds(x)
    # print(x, inds)

  changed = Z.update()
  print('changed', changed)
  xy[:10,:] = random(size=(10,2))
  changed = Z.update()
  print('changed', changed)
  changed = Z.update()
  print('changed', changed)

if __name__ == '__main__':

  main()

