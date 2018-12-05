all: ./conv

# CUDADIR :=  /share/apps/cuda/9.0.176
CUDADIR := /usr/local/cuda-9.2/bin/
# CUDNNDIR := /share/apps/cudnn/9.0v7.0.5
CUDNNDIR := /usr/include

CPPFLAGS := -O2 --std=c++11 -I$(CUDADIR)/include -I$(CUDNNDIR)/include
LDFLAGS := -L$(CUDADIR)/lib -L$(CUDNNDIR)/lib64
LDLIBS := -lcublas -lcudnn

NVCC := nvcc
CC := $(NVCC)

%.o: %.cu
		$(NVCC) $(CPPFLAGS) -c $< -o $@ -lineinfo

.PHONY: clean
clean:
		rm  ./conv
