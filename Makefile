# Makefile for the ICP point-cloud registration project
#
# Serial:
#   make            build optimized serial (bin/serial/icp_serial)
#   make serial     same as `make`
#   make baseline   unoptimized serial (-O0) cache-sweep baseline
#   make debug      -O0 -g for gdb
#   make asan       AddressSanitizer + UBSan
#
# Vectorized (versioned kd-tree backends):
#   make vec        build ALL versions (v0, v1, v2)
#   make vec_v0     build only bin/vectorized/icp_vectorized_v0  (leaf-SIMD baseline)
#   make vec_v1     build only ..._v1                            (flattened + iterative)
#   make vec_v2     build only ..._v2                            (parallel queries)
#   make run_vec_v1 build then run a specific version
#
#   Each src/vectorized/<ver>/ is a self-contained source set (its own icp.c,
#   kdtreeV*.c, main.c, ...), built into its own binary.
#
# OpenMP (src/openmp/ backend: flattened kd-tree + threaded query loop):
#   make omp        build bin/openmp/icp_openmp (vectorized leaf + -fopenmp)
#   make run_omp    build then run it (set OMP_NUM_THREADS to pick thread count)
#
#   make clean      remove all build artifacts

# ---- Toolchain & flags -------------------------------------------------------

CC       := cc
CSTD     := -std=c11
WARN     := -Wall -Wextra
OPT      := -O3 -march=native
VEC      := -ftree-vectorize -funroll-loops -mtune=native
VEC_INFO := -fopt-info-vec-optimized -fopt-info-vec-missed -fopt-info-vec-all
CPPFLAGS := -Iinclude
CFLAGS   := $(CSTD) $(WARN) $(OPT)
LDLIBS   := -lm

# ---- Serial: sources & objects ----------------------------------------------

SRC_DIR   := src/serial
BUILD_DIR := build/serial
BIN_DIR   := bin/serial

SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))
TARGET  := $(BIN_DIR)/icp_serial

# ---- Default / serial --------------------------------------------------------

.PHONY: all serial
all:    $(TARGET)
serial: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OBJECTS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# ---- Vectorized: versioned kd-tree backends (folder-per-version) ------------
# Each src/vectorized/<ver>/ is a self-contained source set. A version is built
# by compiling everything under its folder into bin/vectorized/icp_vectorized_<ver>,
# with objects kept apart in build/vectorized/<ver>/.

SRC_DIR_VEC   := src/vectorized
BUILD_DIR_VEC := build/vectorized
BIN_DIR_VEC   := bin/vectorized

VERSIONS := v0 v1 v2

VEC_CFLAGS := $(CFLAGS) $(VEC) $(VEC_INFO)

# Build every version.
.PHONY: vec
vec: $(foreach V,$(VERSIONS),$(BIN_DIR_VEC)/icp_vectorized_$(V))

$(BIN_DIR_VEC):
	mkdir -p $@

# Per-version: discover that folder's sources, compile into its own object dir,
# then link them into the version binary.
define VEC_RULE
SRCS_$(1) := $$(wildcard $$(SRC_DIR_VEC)/$(1)/*.c)
OBJS_$(1) := $$(patsubst $$(SRC_DIR_VEC)/$(1)/%.c,$$(BUILD_DIR_VEC)/$(1)/%.o,$$(SRCS_$(1)))

$$(BUILD_DIR_VEC)/$(1):
	mkdir -p $$@

$$(BUILD_DIR_VEC)/$(1)/%.o: $$(SRC_DIR_VEC)/$(1)/%.c | $$(BUILD_DIR_VEC)/$(1)
	$$(CC) $$(CPPFLAGS) $$(VEC_CFLAGS) -MMD -MP -c $$< -o $$@

$$(BIN_DIR_VEC)/icp_vectorized_$(1): $$(OBJS_$(1)) | $$(BIN_DIR_VEC)
	$$(CC) $$(VEC_CFLAGS) $$(OBJS_$(1)) $$(LDLIBS) -o $$@

.PHONY: vec_$(1)
vec_$(1): $$(BIN_DIR_VEC)/icp_vectorized_$(1)

.PHONY: run_vec_$(1)
run_vec_$(1): $$(BIN_DIR_VEC)/icp_vectorized_$(1)
	./$$<
endef
$(foreach V,$(VERSIONS),$(eval $(call VEC_RULE,$(V))))

# ---- OpenMP: dedicated src/openmp/ backend ----------------------------------
# Its own sources (flattened kd-tree + the threaded query loop in icp.c),
# compiled with VEC for the SIMD leaf and -fopenmp for the threads. Single
# binary; pick the thread count at run time with OMP_NUM_THREADS.

OMP           := -fopenmp
SRC_DIR_OMP   := src/openmp
BUILD_DIR_OMP := build/openmp
BIN_DIR_OMP   := bin/openmp

OMP_CFLAGS  := $(CFLAGS) $(VEC) $(OMP)
SOURCES_OMP := $(wildcard $(SRC_DIR_OMP)/*.c)
OBJECTS_OMP := $(patsubst $(SRC_DIR_OMP)/%.c,$(BUILD_DIR_OMP)/%.o,$(SOURCES_OMP))
TARGET_OMP  := $(BIN_DIR_OMP)/icp_openmp

.PHONY: omp
omp: $(TARGET_OMP)

$(TARGET_OMP): $(OBJECTS_OMP) | $(BIN_DIR_OMP)
	$(CC) $(OMP_CFLAGS) $(OBJECTS_OMP) $(LDLIBS) -o $@

$(BUILD_DIR_OMP)/%.o: $(SRC_DIR_OMP)/%.c | $(BUILD_DIR_OMP)
	$(CC) $(CPPFLAGS) $(OMP_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR_OMP) $(BIN_DIR_OMP):
	mkdir -p $@

.PHONY: run_omp
run_omp: $(TARGET_OMP)
	./$(TARGET_OMP)

# ---- Convenience -------------------------------------------------------------

.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Unoptimized build with symbols, for gdb.
.PHONY: debug
debug: CFLAGS := $(CSTD) $(WARN) -O0 -g
debug: clean $(TARGET)

# ---- Unoptimized timing baseline --------------------------------------------
BUILD_DIR_BASE := build/baseline
BIN_DIR_BASE   := bin/baseline
OBJECTS_BASE   := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR_BASE)/%.o,$(SOURCES))
TARGET_BASE    := $(BIN_DIR_BASE)/icp_baseline
BASE_CFLAGS    := $(CSTD) $(WARN) -O0

.PHONY: baseline
baseline: $(TARGET_BASE)

$(TARGET_BASE): $(OBJECTS_BASE) | $(BIN_DIR_BASE)
	$(CC) $(BASE_CFLAGS) $(OBJECTS_BASE) $(LDLIBS) -o $@

$(BUILD_DIR_BASE)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR_BASE)
	$(CC) $(CPPFLAGS) $(BASE_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR_BASE) $(BIN_DIR_BASE):
	mkdir -p $@

# AddressSanitizer build.
.PHONY: asan
asan: CFLAGS := $(CSTD) $(WARN) -O1 -g -fsanitize=address,undefined
asan: LDLIBS := -lm -fsanitize=address,undefined
asan: clean $(TARGET)

# ---- Housekeeping ------------------------------------------------------------

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(BUILD_DIR_VEC) $(BIN_DIR_VEC) \
	$(BUILD_DIR_OMP) $(BIN_DIR_OMP) $(BUILD_DIR_BASE) $(BIN_DIR_BASE)

# Auto-generated header dependencies.
-include $(OBJECTS:.o=.d)
-include $(OBJECTS_BASE:.o=.d)
-include $(wildcard $(BUILD_DIR_VEC)/*/*.d)
-include $(wildcard $(BUILD_DIR_OMP)/*.d)
