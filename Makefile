BUILD_DIR := $(CURDIR)/build
SRC_DIR   := $(CURDIR)/llvm

CC  := /usr/bin/clang-21
CXX := /usr/bin/clang++-21

SYMBOLIZER := LLVM_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer-21

CLANG   := $(BUILD_DIR)/bin/clang
CIR_OPT := $(BUILD_DIR)/bin/cir-opt

CMAKE_COMMON := \
	-G Ninja \
	-DCMAKE_C_COMPILER=$(CC) \
	-DCMAKE_CXX_COMPILER=$(CXX) \
	-DCMAKE_C_COMPILER_LAUNCHER=ccache \
	-DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
	-DLLVM_ENABLE_PROJECTS="clang;mlir" \
	-DLLVM_TARGETS_TO_BUILD=host \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DLLVM_USE_LINKER=/usr/bin/ld.lld-21 \
	-DCLANG_ENABLE_CIR=ON

CMAKE_FLAGS       := $(CMAKE_COMMON) -DCMAKE_BUILD_TYPE=Release
CMAKE_FLAGS_DEBUG := $(CMAKE_COMMON) -DCMAKE_BUILD_TYPE=RelWithDebInfo

.PHONY: all configure configure-debug build clang cir-opt cir-opt-debug \
        filecheck littools test-throughmlir run run-debug c2cir c2mlir clean help

# Default: just cir-opt for fast iteration on lowering code
all: cir-opt

# ── Configure ────────────────────────────────────────────────────────────────

configure: $(BUILD_DIR)/build.ninja

$(BUILD_DIR)/build.ninja:
	cmake -S $(SRC_DIR) -B $(BUILD_DIR) $(CMAKE_FLAGS)

# Reconfigure in-place for RelWithDebInfo (debug symbols, ~Release speed)
configure-debug:
	cmake -S $(SRC_DIR) -B $(BUILD_DIR) $(CMAKE_FLAGS_DEBUG)

# ── Build targets ─────────────────────────────────────────────────────────────

# Fast incremental build after editing LowerCIRToMLIR.cpp
cir-opt: $(BUILD_DIR)/build.ninja
	ninja -C $(BUILD_DIR) cir-opt

# Reconfigure for debug symbols then rebuild cir-opt
cir-opt-debug: configure-debug
	ninja -C $(BUILD_DIR) cir-opt

# Build clang (required for C → CIR; takes ~30–60 min first time)
clang: $(BUILD_DIR)/build.ninja
	ninja -C $(BUILD_DIR) clang

mlir-opt: $(BUILD_DIR)/build.ninja
	ninja -C $(BUILD_DIR) mlir-opt

# Full build of everything
build: $(BUILD_DIR)/build.ninja
	ninja -C $(BUILD_DIR)

# ── Run / test ────────────────────────────────────────────────────────────────

# Compile a C file to CIR
# Usage: make c2cir FILE=foo.c
c2cir: clang
	$(CLANG) -emit-cir -S $(FILE) -o -

# Compile a C file all the way to MLIR (via CIR)
# Usage: make c2mlir FILE=foo.c
c2mlir: clang cir-opt
	$(CLANG) -emit-cir -S $(FILE) -o - | $(CIR_OPT) - --cir-to-mlir

# Lower an existing .cir file to MLIR
# Usage: make run FILE=clang/test/CIR/Lowering/ThroughMLIR/ptrstride.cir
run: cir-opt
	$(CIR_OPT) $(FILE) --cir-to-mlir

# Same as run but with symbolized crash output
run-debug: cir-opt
	$(SYMBOLIZER) $(CIR_OPT) $(FILE) --cir-to-mlir

# Build FileCheck and other lit tools (needed for lit tests)
filecheck: $(BUILD_DIR)/build.ninja
	ninja -C $(BUILD_DIR) FileCheck

littools: $(BUILD_DIR)/build.ninja
	ninja -C $(BUILD_DIR) FileCheck count not llvm-symbolizer mlir-translate llvm-config

# Run the ThroughMLIR lit test suite
test-throughmlir: cir-opt littools
	PATH=$(BUILD_DIR)/bin:$$PATH $(BUILD_DIR)/bin/llvm-lit -v \
		--path $(BUILD_DIR)/bin \
		clang/test/CIR/Lowering/ThroughMLIR/

# ── Misc ──────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Configure:"
	@echo "  configure             cmake configure (Release)"
	@echo "  configure-debug       Reconfigure build dir for RelWithDebInfo"
	@echo ""
	@echo "Build:"
	@echo "  all / cir-opt         Build cir-opt (fast incremental)"
	@echo "  cir-opt-debug         RelWithDebInfo reconfigure + rebuild cir-opt"
	@echo "  clang                 Build clang (needed for C → CIR)"
	@echo "  build                 Full ninja build"
	@echo ""
	@echo "Run / test:"
	@echo "  run FILE=<f.cir>      Lower .cir file via --cir-to-mlir"
	@echo "  run-debug FILE=<f>    Same with symbolized crash output"
	@echo "  c2cir FILE=<f.c>      Compile C to CIR (stdout)"
	@echo "  c2mlir FILE=<f.c>     Compile C all the way to MLIR (stdout)"
	@echo "  test-throughmlir      Run ThroughMLIR lit test suite"
	@echo ""
	@echo "  clean                 Delete build directory"
