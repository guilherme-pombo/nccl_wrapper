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

def get_pointer(inlist, typ):
    _ary = (typ * len(inlist))(*inlist)
    return cast(byref(_ary), POINTER(typ))

_devs = (c_int * sz)(*devlist)
_srcs = (c_void_p * sz)(*srcs)
_dsts = (c_void_p * sz)(*dsts)

ncc = nccllib.create(c_int(ndev), cast(byref(_devs), POINTER(c_int)))
nccllib.all_reduce(ncc, c_int(sz), cast(byref(_srcs), POINTER(c_void_p)), cast(byref(_dsts), POINTER(c_void_p)))


nccllib.sync(ncc)
for c, i, o in zip(ctxs, inputs, outputs):
    c.push()
    print i.get()
    print o.get()
    c.pop()

nccllib.kill(ncc)
