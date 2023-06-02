EXECUTABLE := gemv

CU_FILES   := test_gemv.cu fast_gemv.cu fast_gemv_quantized.cu utility.cu

CU_DEPS    := fast_gemv.cuh fast_gemv_quantized.cuh utility.cuh

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -std=c++17
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 -ccbin /usr/bin/gcc -std=c++17 \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/test_gemv.o $(OBJDIR)/fast_gemv.o $(OBJDIR)/fast_gemv_quantized.o $(OBJDIR)/utility.o

.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu $(CU_DEPS)
	$(NVCC) $< $(NVCCFLAGS) -c -o $@