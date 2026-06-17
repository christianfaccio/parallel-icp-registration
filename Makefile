# Makefile for the serial ICP point-cloud registration baseline
#
# Targets:
#   make            build the serial solver (bin/icp_serial)
#   make run        build then run with default arguments
#   make debug      build with -O0 -g for gdb
#   make asan       build with AddressSanitizer + UBSan
#   make clean      remove build artifacts
#
# Layout:
#   include/   public headers
#   src/       translation units
#   build/     object files (generated)
#   bin/       executables (generated)
#
# The serial build is the correctness baseline; the OpenMP / MPI / CUDA
# backends will be added as additional targets below (see "Parallel backends").

# ---- Toolchain & flags -------------------------------------------------------

CC       := cc
CSTD     := -std=c11
WARN     := -Wall -Wextra
OPT      := -O3 -march=native
VEC	 := -ftree-vectorize -funroll-loops -mtune=native
VEC_INFO:= -fopt-info-vec-optimized -fopt-info-vec-missed -fopt-info-vec-all
CPPFLAGS := -Iinclude
CFLAGS   := $(CSTD) $(WARN) $(OPT) 
LDLIBS   := -lm

# ---- Sources & objects -------------------------------------------------------

SRC_DIR   := src/serial
BUILD_DIR := build/serial
BIN_DIR   := bin/serial

SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))

TARGET := $(BIN_DIR)/icp_serial

SRC_DIR_VEC	:= src/vectorized
BUILD_DIR_VEC	:= build/vectorized
BIN_DIR_VEC	:= bin/vectorized

SOURCES_VEC	:= $(wildcard $(SRC_DIR_VEC)/*.c)
OBJECTS_VEC	:= $(patsubst $(SRC_DIR_VEC)/%.c,$(BUILD_DIR_VEC)/%.o,$(SOURCES_VEC))

TARGET_VEC	:= $(BIN_DIR_VEC)/icp_vectorized

# ---- Default target ----------------------------------------------------------

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OBJECTS) $(LDLIBS) -o $@

# Compile each .c -> .o, regenerating when its headers change (see deps below).
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

.PHONY: vec
vec: $(TARGET_VEC)

$(TARGET_VEC): $(OBJECTS_VEC) | $(BIN_DIR_VEC)
	$(CC) $(CFLAGS) $(VEC) $(VEC_INFO) $(OBJECTS_VEC) $(LDLIBS) -o $@

$(BUILD_DIR_VEC)/%.o: $(SRC_DIR_VEC)/%.c | $(BUILD_DIR_VEC)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(VEC) $(VEC_INFO) -MMD -MP -c $< -o $@

$(BUILD_DIR_VEC) $(BIN_DIR_VEC):
	mkdir -p $@

# ---- Convenience targets -----------------------------------------------------

.PHONY: run
run: $(TARGET)
	./$(TARGET)

.PHONY: run_vec
run_vec: $(TARGET_VEC)
	./$(TARGET_VEC)

# Unoptimized build with symbols, for gdb.
.PHONY: debug
debug: CFLAGS := $(CSTD) $(WARN) -O0 -g
debug: clean $(TARGET)

# AddressSanitizer build: catches leaks, use-after-free, out-of-bounds.
.PHONY: asan
asan: CFLAGS := $(CSTD) $(WARN) -O1 -g -fsanitize=address,undefined
asan: LDLIBS := -lm -fsanitize=address,undefined
asan: clean $(TARGET)

# ---- Parallel backends (to be added) -----------------------------------------
# When the OpenMP backend lands, add e.g.:
#   bin/icp_omp: CFLAGS += -fopenmp
#   bin/icp_omp: LDLIBS += -fopenmp
# guarded behind its own object dir so serial and OpenMP objects don't collide.

# ---- Housekeeping ------------------------------------------------------------

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(BUILD_DIR_VEC) $(BIN_DIR_VEC)

# Auto-generated header dependencies (-MMD). Silently ignored on first build.
-include $(OBJECTS:.o=.d)
