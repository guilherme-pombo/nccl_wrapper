#!/usr/bin/env python
import os
from ctypes import *
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np

drv.init()

libpath = os.path.dirname(os.path.realpath(__file__))
nccllib = cdll.LoadLibrary(os.path.join(libpath, 'nccllib.so'))

ndev = 2
devlist = range(ndev)
devs = [drv.Device(i) for i in devlist]
ctxs = [dev.retain_primary_context() for dev in devs]
inputs  = []
outputs = []
sz = 10
for ctx in ctxs:
    ctx.push()
    inputs.append(gpuarray.arange(100, 200, sz, dtype='<f4'))
    outputs.append(gpuarray.zeros((sz,), dtype=np.float32))
    ctx.pop()

srcs = [cast(int(x.gpudata), c_void_p) for x in inputs]
dsts = [cast(int(x.gpudata), c_void_p) for x in outputs]

nccllib.create.restype = c_void_p
nccllib.create.argtypes = [c_int, POINTER(c_int)]
nccllib.kill.argtypes = [c_void_p]
nccllib.sync.argtypes = [c_void_p]
nccllib.all_reduce.argtypes = [c_void_p,
                               c_int,
                               POINTER(c_void_p),
                               POINTER(c_void_p)]

def int_p(_ary):
    return cast(byref(_ary), POINTER(c_int))

def void_p(_ary):
    return cast(byref(_ary), POINTER(c_void_p))

_devs = (c_int * sz)(*devlist)
_srcs = (c_void_p * sz)(*srcs)
_dsts = (c_void_p * sz)(*dsts)

ncc = nccllib.create(c_int(ndev), int_p(_devs))
nccllib.all_reduce(ncc, c_int(sz), void_p(_srcs), void_p(_dsts))

nccllib.sync(ncc)
for c, i, o in zip(ctxs, inputs, outputs):
    c.push()
    print i.get()
    print o.get()
    c.pop()

nccllib.kill(ncc)
