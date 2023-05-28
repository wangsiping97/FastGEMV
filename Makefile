EXECUTABLE := gemv

CU_FILES   := test_gemv.cu fast_gemv.cu simple_tensor.cu

CU_DEPS    := fast_gemv.cuh

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 -ccbin /usr/bin/gcc \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/test_gemv.o $(OBJDIR)/fast_gemv.o $(OBJDIR)/simple_tensor.o

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