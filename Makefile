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
#   Only src/vectorized/kdtreeV_vN.c differs per version; the other vectorized
#   sources are shared and compiled once.
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

# ---- Vectorized: versioned kd-tree backends ---------------------------------

SRC_DIR_VEC   := src/vectorized
BUILD_DIR_VEC := build/vectorized
BIN_DIR_VEC   := bin/vectorized

VERSIONS := v0 v1 v2 

# Per-version source is kdtreeV_vN.c; everything else is shared.
VERSIONED_VEC_SRC := $(wildcard $(SRC_DIR_VEC)/kdtreeV_v*.c)
COMMON_VEC_SRC    := $(filter-out $(VERSIONED_VEC_SRC),$(wildcard $(SRC_DIR_VEC)/*.c))
COMMON_VEC_OBJ    := $(patsubst $(SRC_DIR_VEC)/%.c,$(BUILD_DIR_VEC)/%.o,$(COMMON_VEC_SRC))

VEC_CFLAGS := $(CFLAGS) $(VEC) $(VEC_INFO)

# Compile any vectorized .c (shared or versioned) with the VEC flags.
$(BUILD_DIR_VEC)/%.o: $(SRC_DIR_VEC)/%.c | $(BUILD_DIR_VEC)
	$(CC) $(CPPFLAGS) $(VEC_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR_VEC) $(BIN_DIR_VEC):
	mkdir -p $@

# Build every version.
.PHONY: vec
vec: $(foreach V,$(VERSIONS),$(BIN_DIR_VEC)/icp_vectorized_$(V))

# Generate per-version link + phony rules: link common objs + the one kd-tree obj.
define VEC_RULE
$(BIN_DIR_VEC)/icp_vectorized_$(1): $$(COMMON_VEC_OBJ) $$(BUILD_DIR_VEC)/kdtreeV_$(1).o | $$(BIN_DIR_VEC)
	$$(CC) $$(VEC_CFLAGS) $$^ $$(LDLIBS) -o $$@

.PHONY: vec_$(1)
vec_$(1): $(BIN_DIR_VEC)/icp_vectorized_$(1)

.PHONY: run_vec_$(1)
run_vec_$(1): $(BIN_DIR_VEC)/icp_vectorized_$(1)
	./$$<
endef
$(foreach V,$(VERSIONS),$(eval $(call VEC_RULE,$(V))))

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
	$(BUILD_DIR_BASE) $(BIN_DIR_BASE)

# Auto-generated header dependencies.
-include $(OBJECTS:.o=.d)
-include $(OBJECTS_BASE:.o=.d)
-include $(wildcard $(BUILD_DIR_VEC)/*.d)
