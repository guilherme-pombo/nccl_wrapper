# nccl_wrapper
ctypes wrapper for nccl with pycuda

In order for the cuda driver level allocations to work with the runtime api calls in nccl, we need to be able to access primary contexts.

Use the pycuda fork in apark263/pycuda.
