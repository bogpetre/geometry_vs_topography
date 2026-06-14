# Makefile for compiling rdm_similarity.cpp

# Output binary name
TARGET32 = bin/rdm_similarity32
TARGET64 = bin/rdm_similarity64

# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp

# Include paths
EIGEN_INC = $(CONDA_PREFIX)/include/eigen3
JSON_INC  = $(CONDA_PREFIX)/include
LIB_PATH = ${CONDA_PREFIX}/lib

# LAPACK / BLAS (assumes conda-forge lapack/OpenBLAS or MKL)
LDFLAGS = -L${LIB_PATH} -lblas -llapack

# Source file
SRC32 = src/rdm_similarity/rdm_similarity32.cpp
SRC64 = src/rdm_similarity/rdm_similarity64.cpp

# Build all

all: build_dir $(TARGET32) $(TARGET64)

build_dir:
	mkdir -p bin

$(TARGET32): $(SRC32) src/rdm_similarity/rdm_similarity.inl
	$(CXX) $(CXXFLAGS) -I$(EIGEN_INC) -I$(JSON_INC) $(SRC32) -o $(TARGET32) ${LDFLAGS}

$(TARGET64): $(SRC64) src/rdm_similarity/rdm_similarity.inl
	$(CXX) $(CXXFLAGS) -I$(EIGEN_INC) -I$(JSON_INC) $(SRC64) -o $(TARGET64) ${LDFLAGS}

# Clean target
clean:
	rm -f $(TARGET32) $(TARGET64)
