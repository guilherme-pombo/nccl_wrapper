#!/usr/bin/env python
import threading

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from ncclcomm import NCCLComm

def init_comm_entry(nc_handle, dev_index, ctx):
	ctx.push()
	nc_handle.init_comm(dev_index)
	ctx.pop()

# This is the list of devices to be used by ordinals
ndev = 2
devlist = range(ndev)

# Setup the pycuda side
drv.init()

ctxs = []
strms = []
for i in devlist:
	ctx = drv.Device(i).make_context()
	ctxs.append(ctx)
	strms.append(drv.Stream())
	ctx.pop()

# Setup the communicator object
nc = NCCLComm(devlist, strms)
threads = []
for ctx, dev_index in zip(ctxs, range(ndev)):
    t = threading.Thread(target=init_comm_entry, args=(nc, dev_index, ctx))
    threads.append(t)
    t.start()

for t in threads:
	t.join()

# Now create gpuarrays for sending/recv buffers
srcs, dsts, size = [], [], 10

# Create some test arrays
for ctx in ctxs:
    ctx.push()
    srcs.append(gpuarray.arange(100, 200, size, dtype='<f4'))
    dsts.append(gpuarray.zeros((size,), dtype='<f4'))
    ctx.pop()

# Perform the reduction
for ctx, src, dst, dev_index in zip(ctxs, srcs, dsts, range(ndev)):
	ctx.push()
	nc.all_reduce(size, src, dst, dev_index)
	ctx.pop()

for stream in strms:
	stream.synchronize()

# Look at the results
for ctx, i, o in zip(ctxs, srcs, dsts):
    ctx.push()
    print(i.get())
    print(o.get())
    ctx.pop()

# Destroy comms
for ctx, dev_index in zip(ctxs, range(ndev)):
	ctx.push()
	nc.destroy_comm(dev_index)
	ctx.pop()
