# NCCL wrapper for pycuda
We are attempting to create a shared library interface to NCCL primitives that would be usable by pycuda through ctypes.

In order for the cuda driver level allocations to work with the runtime api calls in nccl, we need to be able to access primary contexts.

Use [this pycuda fork](https://github.com/apark263/pycuda).

