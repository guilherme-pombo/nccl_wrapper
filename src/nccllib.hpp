#include <nccl.h>
#include <stdlib.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

class NcclComm {
public:
    NcclComm(int nGPU, int* GPUs) {
        _ndevs = nGPU;
        _devcs = (int *) malloc(sizeof(int) * _ndevs);
        _strms = (cudaStream_t *) malloc(sizeof(cudaStream_t) * _ndevs);
        _comms = (ncclComm_t *) malloc(sizeof(ncclComm_t) * _ndevs);

        for (int i = 0; i < _ndevs; ++i) {
            _devcs[i] = GPUs[i];
            // Correct device must be set prior to each collective call.
            CUDACHECK(cudaSetDevice(_devcs[i]));
            CUDACHECK(cudaStreamCreate(_strms+i));
        }
        NCCLCHECK(ncclCommInitAll(_comms, _ndevs, _devcs));
    }

    virtual ~NcclComm() {
        for (int i = 0; i < _ndevs; ++i) {
            CUDACHECK(cudaSetDevice(_devcs[i]));
            CUDACHECK(cudaStreamDestroy(_strms[i]));
        }
        for(int i = 0; i < _ndevs; ++i)
            ncclCommDestroy(_comms[i]);
        free(_comms);
        free(_devcs);
        free(_strms);
        return;

    }

    void all_reduce(int size, void** sendBuffs, void** recvBuffs) {
        for (int i = 0; i < _ndevs; ++i) {
            CUDACHECK(cudaSetDevice(_devcs[i]));
            NCCLCHECK(ncclAllReduce((const void*) sendBuffs[i],
                                    (void*) recvBuffs[i],
                                    size, ncclFloat, ncclSum, _comms[i], _strms[i]));
        }
    }

    void reduce_scatter(int size, void** sendBuffs, void** recvBuffs) {
        for (int i = 0; i < _ndevs; ++i) {
            CUDACHECK(cudaSetDevice(_devcs[i]));
            NCCLCHECK(ncclReduceScatter((const void*) sendBuffs[i],
                                        (void*) recvBuffs[i],
                                        size, ncclFloat, ncclSum, _comms[i], _strms[i]));
        }
    }

    void sync() {
        for (int i = 0; i < _ndevs; ++i) {
            CUDACHECK(cudaSetDevice(_devcs[i]));
            CUDACHECK(cudaStreamSynchronize(_strms[i]));
        }
    }


protected:
    ncclComm_t*     _comms;
    cudaStream_t*   _strms;
    int             _ndevs;
    int*            _devcs;
};
