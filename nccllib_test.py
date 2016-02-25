#!/usr/bin/env python
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from ncclcomm import NCCLComm

# This is the list of devices to be used by ordinals
ndev = 2
devlist = range(ndev)

# Setup the pycuda side
drv.init()
ctxs = [drv.Device(i).retain_primary_context() for i in devlist]

# Setup the communicator object
nc = NCCLComm(devlist)

# Now create gpuarrays for sending/recv buffers
srcs, dsts, size = [], [], 10

# Create some test arrays
for ctx in ctxs:
    ctx.push()
    srcs.append(gpuarray.arange(100, 200, size, dtype='<f4'))
    dsts.append(gpuarray.zeros((size,), dtype='<f4'))
    ctx.pop()

# Perform the reduction
nc.all_reduce(size, srcs, dsts)
nc.sync()

# Look at the results
for c, i, o in zip(ctxs, srcs, dsts):
    c.push()
    print i.get()
    print o.get()
    c.pop()
