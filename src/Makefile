# Makefile for compiling rdm_similarity.cpp

# Output binary name
TARGET = rdm_similarity

# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp

# Include paths
EIGEN_INC = $(CONDA_PREFIX)/include/eigen3
JSON_INC  = $(CONDA_PREFIX)/include
LIB_PATH = ${CONDA_PREFIX}/lib

# LAPACK / BLAS (assumes conda-forge lapack/OpenBLAS or MKL)
LDFLAGS = -L${LIB_PATH} -lblas

# Source file
SRC = rdm_similarity.cpp

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -I$(EIGEN_INC) -I$(JSON_INC) $(SRC) -o $(TARGET) ${LDFLAGS}

# Clean target
clean:
	rm -f $(TARGET)

