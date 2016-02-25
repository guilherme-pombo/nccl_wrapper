SRCS := $(wildcard src/*.hpp)
CFLAGS := -Wall -Werror -O3 -std=c++11
CC := g++
export CUDA_ROOT:=$(patsubst %/bin/nvcc,%, $(realpath $(shell which nvcc)))

INC := -I$(CUDA_ROOT)/include $(INC)
ifeq ($(shell uname -s),Darwin)
	LDIR := -L$(CUDA_ROOT)/lib $(LDIR)
else
	LDIR := -L$(CUDA_ROOT)/lib64 $(LDIR)
endif
LIBS := -lcuda -lcudart -lnccl $(LIBS)

all: ncclcomm/nccllib.so

ncclcomm/nccllib.so: src/nccllib.cpp $(SRCS)
	$(CC) -shared -o $@ -fPIC $(CFLAGS) $< $(INC) $(LDIR) $(LIBS)
