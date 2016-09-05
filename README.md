# Differential Lattice

A generative algorithm.


![ani](/img/ani.gif?raw=true "ani")

![ani](/img/ani2.gif?raw=true "ani")

![ani](/img/ani4.gif?raw=true "ani")

![ani](/img/ani5.gif?raw=true "ani")

Sometimes bugs are the best results:

![ani](/img/ani3.gif?raw=true "ani")


## Prerequisites

This code relies on Python 3.

In order for this code to run you must first download and install:

*    `iutils`: https://github.com/inconvergent/iutils
*    `fn`: https://github.com/inconvergent/fn-python3 (used to generate file
     names, can be removed in main file)

## Other Dependencies

The code also depends on:

*    `numpy`
*    `python-cairo` (do not install with pip, this generally does not work)
*    `pycuda`


note to self: if cuda is not working try `sudo ldconfig`. and check
$LD_LIBRARY_PATH

