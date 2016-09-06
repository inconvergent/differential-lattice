# -*- coding: utf-8 -*-

from time import time
from collections import defaultdict

class named_sub_timers(object):

  def __init__(self, name=None):

    self.name = name
    self.times = defaultdict(float)
    self.now = time()
    self.total = 0.

  def start(self):

    self.now = time()

  def t(self,n):

    t = time()
    diff = t-self.now

    self.times[n] += diff
    self.total += diff

    self.now = t

  def p(self):

    total = self.total

    print(('{:s}'.format('' if not self.name else self.name)))

    for n,t in list(self.times.items()):

      print(('{:s}\t{:0.6f}\t{:0.6f}'.format(n,t,t/total)))

    print(('total\t{:0.6f}\n'.format(total)))

