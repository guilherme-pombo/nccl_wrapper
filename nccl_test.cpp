#include "nccl_test.hpp"

extern "C" {

    extern void* create(int nGPU, int* GPUs) {
        try {
            NcclComm* nc = new NcclComm(nGPU, GPUs);
            return reinterpret_cast<void*>(nc);
        } catch(...) {
            return 0;
        }
    }

    extern int all_reduce(NcclComm *nc, int size, void** sendBuffs, void** recvBuffs) {
        try {
            nc->all_reduce(size, sendBuffs, recvBuffs);
            return 0;
        } catch(...) {
            return -1;
        }
    }

    extern int reduce_scatter(NcclComm *nc, int size, void** sendBuffs, void** recvBuffs) {
        try {
            nc->reduce_scatter(size, sendBuffs, recvBuffs);
            return 0;
        } catch(...) {
            return -1;
        }
    }

    extern int sync(NcclComm* nc) {
        try {
            nc->sync();
            return 0;
        } catch(...) {
            return -1;
        }
    }

    extern int kill(NcclComm* nc) {
        try {
            nc->sync();
            delete nc;
            return 0;
        } catch(...) {
            return -1;
        }
    }
}
