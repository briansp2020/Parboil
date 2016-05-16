# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# c.default is the base along with CUDA configuration in this setting
include $(PARBOIL_ROOT)/common/platform/c.default.mk

# Paths
HIPHOME=/opt/rocm/hip

# Programs
HIPCC=$(HIPHOME)/bin/hipcc
HIPLINK=$(HIPHOME)/bin/hipcc
#CC=/opt/rocm/hcc/bin/clang

# Flags
PLATFORM_HIPCFLAGS=-O3
PLATFORM_HIPLDFLAGS=-lm -lpthread


