from ctypes import *
from os import path as osp

def int_p(_ary):
    return cast(byref(_ary), POINTER(c_int))

def void_p(_ary):
    return cast(byref(_ary), POINTER(c_void_p))

class NCCLComm(object):

    def __init__(self, devlist, streamlist):
        try:
            libpath = osp.dirname(osp.realpath(__file__))
            self.nccllib = cdll.LoadLibrary(osp.join(libpath, 'nccllib.so'))
            self.nccllib.create.restype = c_void_p
            self.nccllib.create.argtypes = [c_int, POINTER(c_int), POINTER(c_void_p)]
            self.nccllib.init_comm.argtypes = [c_void_p, c_int]
            self.nccllib.destroy_comm.argtypes = [c_void_p, c_int]
            self.nccllib.kill.argtypes = [c_void_p]
            self.nccllib.all_reduce.argtypes = [c_void_p, c_int,
                                                c_void_p, c_void_p, c_int]
        except:
            print('Unable to loader shared lib')

        self.ndevs = len(devlist)
        self._devs = (c_int * self.ndevs)(*devlist)
        self._streams = streamlist

        streams_arg = (c_void_p * len(self._streams))(*[cast(s.handle, c_void_p) for s in self._streams])
        self.comm = self.nccllib.create(c_int(self.ndevs), int_p(self._devs), void_p(streams_arg))

    def init_comm(self, dev_index):
        self.nccllib.init_comm(self.comm, c_int(dev_index))

    def destroy_comm(self, dev_index):
        self.nccllib.destroy_comm(self.comm, c_int(dev_index))

    def all_reduce(self, size, source, dest, dev_index):
        '''
        For not provided destination array list, we do reduce in place
        '''
        _src = cast(int(source.gpudata), c_void_p)
        if not dest:
            _dst = _src
        else:
            _dst = cast(int(dest.gpudata), c_void_p)

        self.nccllib.all_reduce(self.comm, c_int(size), _src, _dst, c_int(dev_index))

    def __del__(self):
        self.nccllib.kill(self.comm)
