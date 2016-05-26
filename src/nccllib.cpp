#include "nccllib.hpp"

extern "C" {

    extern void* create(int nGPU, int* GPUs, cudaStream_t* streams) {
        try {
            NcclComm* nc = new NcclComm(nGPU, GPUs, streams);
            return reinterpret_cast<void*>(nc);
        } catch(...) {
            return 0;
        }
    }

    extern int init_comm(NcclComm *nc, int dev_index) {
        try {
            nc->init_comm(dev_index);
            return 0;
        } catch(...) {
            return -1;
        }
    }

    extern int destroy_comm(NcclComm *nc, int dev_index) {
        try {
            nc->destroy_comm(dev_index);
            return 0;
        } catch(...) {
            return -1;
        }
    }

    extern int all_reduce(NcclComm *nc, int size, void* sendBuff, void* recvBuff, int dev_index) {
        try {
            nc->all_reduce(size, sendBuff, recvBuff, dev_index);
            return 0;
        } catch(...) {
            return -1;
        }
    }

    extern int reduce_scatter(NcclComm *nc, int size, void* sendBuff, void* recvBuff, int dev_index) {
        try {
            nc->reduce_scatter(size, sendBuff, recvBuff, dev_index);
            return 0;
        } catch(...) {
            return -1;
        }
    }

    extern int kill(NcclComm* nc) {
        try {
            delete nc;
            return 0;
        } catch(...) {
            return -1;
        }
    }
}
