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
    NcclComm(int nGPU, int* GPUs, cudaStream_t* streams) {
        _ndevs = nGPU;
        _devcs = (int *) malloc(sizeof(int) * _ndevs);
        _strms = (cudaStream_t *) malloc(sizeof(cudaStream_t) * _ndevs);
        _comms = (ncclComm_t *) malloc(sizeof(ncclComm_t) * _ndevs);

        for (int i = 0; i < _ndevs; ++i) {
            _devcs[i] = GPUs[i];
            _strms[i] = streams[i];
        }
        NCCLCHECK(ncclGetUniqueId(&_comm_id));
    }

    virtual ~NcclComm() {
        free(_comms);
        free(_devcs);
        free(_strms);
        return;
    }

    void init_comm(int devIndex) {
        NCCLCHECK(ncclCommInitRank(&_comms[devIndex], _ndevs, _comm_id, devIndex));
    }

    void destroy_comm(int devIndex) {
        ncclCommDestroy(_comms[devIndex]);
    }

    void all_reduce(int size, void* sendBuff, void* recvBuff, int devIndex) {
        NCCLCHECK(ncclAllReduce((const void*) sendBuff,
                                (void*) recvBuff,
                                size, ncclFloat, ncclSum, _comms[devIndex], _strms[devIndex]));
    }

    void reduce_scatter(int size, void* sendBuff, void* recvBuff, int devIndex) {
        NCCLCHECK(ncclReduceScatter((const void*) sendBuff,
                                    (void*) recvBuff,
                                    size, ncclFloat, ncclSum, _comms[devIndex], _strms[devIndex]));
    }

protected:
    ncclComm_t*     _comms;
    cudaStream_t*   _strms;
    ncclUniqueId    _comm_id;
    int             _ndevs;
    int*            _devcs;
};
