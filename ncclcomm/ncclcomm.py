from ctypes import *
from os import path as osp

def int_p(_ary):
    return cast(byref(_ary), POINTER(c_int))

def void_p(_ary):
    return cast(byref(_ary), POINTER(c_void_p))

class NCCLComm(object):

    def __init__(self, devlist):
        try:
            libpath = osp.dirname(osp.realpath(__file__))
            self.nccllib = cdll.LoadLibrary(osp.join(libpath, 'nccllib.so'))
            self.nccllib.create.restype = c_void_p
            self.nccllib.create.argtypes = [c_int, POINTER(c_int)]
            self.nccllib.kill.argtypes = [c_void_p]
            self.nccllib.sync.argtypes = [c_void_p]
            self.nccllib.all_reduce.argtypes = [c_void_p, c_int,
                                                POINTER(c_void_p), POINTER(c_void_p)]
        except:
            print('Unable to loader shared lib')

        self.ndevs = len(devlist)
        self._devs = (c_int * self.ndevs)(*devlist)

        self.comm = self.nccllib.create(c_int(self.ndevs), int_p(self._devs))

    def all_reduce(self, size, s_arylist, d_arylist=None):
        '''
        For not provided destination array list, we do reduce in place
        '''
        _srcs = (c_void_p * self.ndevs)(*[cast(int(x.gpudata), c_void_p) for x in s_arylist])
        if not d_arylist:
            _dsts = _srcs
        else:
            _dsts = (c_void_p * self.ndevs)(*[cast(int(x.gpudata), c_void_p) for x in d_arylist])

        self.nccllib.all_reduce(self.comm, c_int(size), void_p(_srcs), void_p(_dsts))

    def sync(self):
        self.nccllib.sync(self.comm)

    def __del__(self):
        self.nccllib.kill(self.comm)
