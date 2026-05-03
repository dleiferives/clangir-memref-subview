# ThroughMLIR output quality — known problems

The goal of `--cir-to-mlir` is to produce well-formed, standard MLIR that any downstream MLIR tool can consume.
Currently it does not meet that bar. Known issues:

---

## 1. `scf.while` with multi-block "after" region

**Symptom:**
```
error: 'scf.while' op expects region #1 to have 0 or 1 blocks
```
Reproduced by running `mlir-opt --allow-unregistered-dialect --finalize-memref-to-llvm ... correlation.cir.mlir`.

**Cause:**
`LowerCIRLoopToSCF.cpp` maps `cir.while`/`cir.for` bodies directly to `scf.while` regions. When the loop body contains control flow (e.g. `cir.if`, `cir.break`, `cir.continue`), the lowered body has multiple basic blocks. `scf.while` requires both regions to be single-block.

**Impact:** The `.cir.mlir` output fails SCF verification and cannot be processed by `mlir-opt` or any tool that verifies ops on parse.

**Fix direction:** Either flatten multi-block loop bodies to `cf` dialect before emitting `scf.while`, or lower complex CIR loops to `cf` (unstructured CFG) directly instead of `scf`.

---

## 2. `DenseIntElementsAttr` element-width mismatch in `CIRGlobalOpLowering`

**Symptom:** Assertion crash inside `DenseIntOrFPElementsAttr::getRawIntOrFloat`:
```
Assertion `::isValidIntOrFloat(type.getElementType(), dataEltSize, isInt, isSigned)` failed.
```
Triggered by `-emit-mlir=core` (the intermediate dump of the first lowering stage).

**Cause:** `CIRGlobalOpLowering::matchAndRewrite` calls `DenseIntElementsAttr::get<int>(ShapedType, int)` for zero-initialised globals. When the element type is `i8` (e.g. a string literal global), passing a raw `int` (32-bit) mismatches the 8-bit element width. The fix is `APInt(elementWidth, 0)` — already documented in `dense_int_elements_width` memory but not yet applied to this call site.

**Impact:** Crashes the compiler when using `-emit-mlir=core`. The full `-c` path (`lowerFromCIRToMLIRToLLVMIR`) avoids this by taking a different route through the lowering, masking the bug.

---

## 3. CIR dialect attributes left on the module

**Symptom:**
```
error: #"cir"<"lang<c>"> : 'none' attribute created with unregistered dialect
```
Triggered by `mlir-opt` and `mlir-translate` when processing the `.cir.mlir` output.

**Cause:** The `module` op retains CIR-specific attributes (`cir.lang`, `cir.triple`, `cir.type_size_info`, etc.) after `--cir-to-mlir`. Any tool without the CIR dialect registered rejects the file on parse.

**Impact:** The output is not portable — only tools that link CIR (i.e. `cir-opt`) can parse it. Standard `mlir-opt`/`mlir-translate` are blocked.

**Fix direction:** Strip or migrate CIR module attributes to LLVM/standard equivalents during lowering (e.g. `cir.triple` → `llvm.target_triple` attribute, rest dropped or moved to comments).

---

## Current workaround

Use `clang -fno-clangir-direct-lowering -c` — the full ThroughMLIR pipeline inside clang compiles to a valid ELF object and avoids all three issues above, but produces no inspectable intermediate MLIR. The `.cir.mlir` file produced by `cir-opt --cir-to-mlir` is not yet usable as a standalone artifact.
