# (c) 2007 The Board of Trustees of the University of Illinois.

# HIP-related definitions common to all benchmarks

########################################
# Variables
########################################

# Paths
HIPHOME=/opt/rocm/hip

# Programs
HIPCC=$(HIPHOME)/bin/hipcc
HIPLINK=$(HIPHOME)/bin/hipcc
#CC=/opt/rocm/hcc/bin/clang
CC = $(HIPCC)
CXX = $(HIPCC)
LINKER = $(HIPLINK)

# Flags
PLATFORM_HIPCFLAGS=-O3
PLATFORM_HIPLDFLAGS=-lm -lpthread


