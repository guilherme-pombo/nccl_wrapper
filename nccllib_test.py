#!/usr/bin/env python
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from ncclcomm import NCCLComm

drv.init()
ndev = 2
devlist = range(ndev)
devs = [drv.Device(i) for i in devlist]
ctxs = [dev.retain_primary_context() for dev in devs]
inputs  = []
outputs = []
sz = 10
nc = NCCLComm(devlist)

for ctx in ctxs:
    ctx.push()
    inputs.append(gpuarray.arange(100, 200, sz, dtype='<f4'))
    outputs.append(gpuarray.zeros((sz,), dtype='<f4'))
    ctx.pop()

nc.all_reduce(sz, inputs, outputs)
nc.sync()

for c, i, o in zip(ctxs, inputs, outputs):
    c.push()
    print i.get()
    print o.get()
    c.pop()
