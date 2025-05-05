BUILDDIR     ?= bin
HOST_TARGET  = $(BUILDDIR)/kmeans_host
DPU_TARGET   = $(BUILDDIR)/kmeans_dpu

HOST_SRCS    = host_kmeans.c
DPU_SRCS     = dpu_kmeans.c

NR_DPUS     ?= DPU_ALLOCATE_ALL
NR_TASKLETS ?= 12

# for single DPU, single tasklet
HOST_CFLAGS  = -std=c11 -Wall -Wextra -O2 \
               -DNR_DPUS=$(NR_DPUS) -DNR_TASKLETS=$(NR_TASKLETS) \
               -I. $(shell dpu-pkg-config --cflags dpu)

DPU_CFLAGS   = -Wall -Wextra -O2 -DNR_TASKLETS=$(NR_TASKLETS) 

.PHONY: all clean

all: $(HOST_TARGET) $(DPU_TARGET) 

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/kmeans_host.o: $(HOST_SRCS) | $(BUILDDIR)
	$(CC) $(HOST_CFLAGS) -c -o $@ $(HOST_SRCS)

$(HOST_TARGET): $(BUILDDIR)/kmeans_host.o
	$(CC) $(HOST_CFLAGS) $< -o $@ $(shell dpu-pkg-config --libs dpu) -lm

$(DPU_TARGET): $(DPU_SRCS) | $(BUILDDIR)
	dpu-upmem-dpurte-clang $(DPU_CFLAGS) -o $@ $(DPU_SRCS)

clean:
	rm -rf $(BUILDDIR)