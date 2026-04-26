# CIR → MLIR Lowering: Emit `memref.subview` Instead of `memref.reinterpret_cast`

## Why This Matters

This work is the prerequisite for an alias metadata propagation pass
(described in `progress-report.pdf` by Shravan Sheth, April 2026).

The pass works in two steps:
1. **`MarkAliasGroups`** — walks all `memref.subview` ops in a function, finds
   pairs that share the same base and satisfy a partition-by-endpoint invariant,
   and tags the resulting `memref.load`/`memref.store` ops with
   `alias_meta.group_id` / `alias_meta.role` discardable attributes.
2. **`LowerWithAliasMeta`** — converts only the tagged `memref.load`/`memref.store`
   to `llvm.load`/`llvm.store` with `!alias.scope` / `!noalias` metadata attached,
   before the standard `finalize-memref-to-llvm` sees them.

**The key requirement**: the pass can only detect disjoint memory regions when
`memref.subview` ops are present in the IR.  Currently CIR's lowering emits
`memref.reinterpret_cast` everywhere instead, which is flat offset arithmetic
with no structural disjointness information.  The goal is to change the
lowering so that pointer arithmetic originating from array accesses becomes
`memref.subview` ops.

---

## Background: What the Alias Pass Detects

`MarkAliasGroups` recognises three forms of partition-by-endpoint on dimension 0:

- **Form A (SSA identity)**: `hi.offset[0] == lo.size[0]` as the **same SSA
  value** (or same compile-time constant), and `lo.offset[0] == 0`.
  This is the dynamic split: `lo = A[0..n-1]`, `hi = A[n..]`.

- **Form B (arith.addi chain)**: `hi.offset[0] == arith.addi(lo.offset[0], lo.size[0])`
  in either operand order.  This is the tile-boundary pattern:
  `lo = A[tile*N .. (tile+1)*N-1]`, `hi = A[(tile+1)*N ..]`.

- **Form C (all constants)**: all offset/size values are compile-time constants
  satisfying the equation numerically.

For Forms A and B to fire, **the size operand of `lo`'s subview and the offset
operand of `hi`'s subview must be the same SSA value** in the IR.  This means
the subview that represents `lo` must explicitly carry the size `%n`, and the
subview that represents `hi` must use that exact same `%n` as its offset.
`reinterpret_cast` only carries an offset and no size, so the pass cannot
detect the relationship.

---

## Source File

All changes are in:

```
clang/lib/CIR/Lowering/ThroughMLIR/LowerCIRToMLIR.cpp
```

Total: 2156 lines.  Relevant classes and functions:

| Lines | Name | Purpose |
|-------|------|---------|
| 332–368 | `findBaseAndIndices` | Unwraps reinterpret_cast chain to extract base memref + index list for load/store |
| 371–418 | `eraseIfSafe` | Erases intermediate cast ops after load/store has absorbed their offsets |
| 420–449 | `prepareReinterpretMetadata` | Fills sizes/strides SmallVectors from a MemRefType for use in reinterpret_cast |
| 451–474 | `CIRLoadOpLowering` | Lowers `cir.load` → `memref.load` using findBaseAndIndices |
| 476–498 | `CIRStoreOpLowering` | Lowers `cir.store` → `memref.store` using findBaseAndIndices |
| 1499–1611 | `CIRCastOpLowering` | Lowers all `cir.cast` kinds; `array_to_ptrdecay` case at line 1516 |
| 1613–1667 | `CIRGetElementOpLowering` | Lowers `cir.get_element` → reinterpret_cast (to be changed to subview) |
| 1669–1836 | `CIRPtrStrideOpLowering` | Lowers `cir.ptr_stride`; three sub-paths described below |
| 1943–2030 | `prepareTypeConverter` | Defines CIR→MLIR type mappings |
| 1896–1921 | `populateCIRToMLIRConversionPatterns` | Registers all lowering patterns |

---

## Type Converter (critical context)

```cpp
// !cir.ptr<T> where T is NOT an array type → memref<?xT>
converter.addConversion([&](cir::PointerType type) -> mlir::Type {
  auto ty = convertTypeForMemory(converter, type.getPointee());
  if (isa<cir::ArrayType>(type.getPointee()))
    return ty;   // array pointer returns the array's memref directly
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, ty, ...);
});

// !cir.array<T x N> → memref<NxT>  (multi-dim: memref<NxMxT>)
converter.addConversion([&](cir::ArrayType type) -> mlir::Type {
  SmallVector<int64_t> shape;
  // walks nested arrays to build full shape
  return mlir::MemRefType::get(shape, elementType);
});
```

So:
- `!cir.ptr<!s32i>` → `memref<?xi32>`
- `!cir.ptr<!cir.array<!s32i x 100>>` → `memref<100xi32>`
- `!cir.ptr<!cir.array<!cir.array<!s32i x 512> x N>>` → `memref<?x512xi32>`

---

## Current Lowering Paths (what exists today)

### Path 1: `cir.cast array_to_ptrdecay` (line 1516)

```cpp
case CIR::array_to_ptrdecay: {
  auto newDstType = llvm::cast<mlir::MemRefType>(convertTy(dstType));
  llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
  prepareReinterpretMetadata(newDstType, rewriter, sizes, strides, op);
  rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
      op, newDstType, src, rewriter.getIndexAttr(0), sizes, strides);
  return mlir::success();
}
```

Input:  `!cir.ptr<!cir.array<!s32i x 100>>` (adapted: `memref<100xi32>`)
Output: `memref.reinterpret_cast %src, 0, [?], [1] → memref<?xi32>`

This decays the static-array memref to a dynamic pointer with offset 0.

### Path 2: `cir.ptr_stride` simple path — `rewritePtrStrideToReinterpret` (line 1716)

Called when the ptr_stride result is only used by load/store/cast-array-to-ptr
**and** the pointee is NOT an array type.

```cpp
rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
    op, memrefType, base, stride, mlir::ValueRange{}, mlir::ValueRange{}, ...);
```

Input:  `cir.ptr_stride %p, %n : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>`
Adapted base: `memref<?xi32>`, stride: `%n` (after index_cast)
Output: `memref.reinterpret_cast %p, %n, [], [] → memref<?xi32>`

Note: empty ValueRange{} for sizes/strides means the result inherits from the
result type's layout.  `findBaseAndIndices` in `CIRLoadOpLowering` peels this
off to get base=`%p`, index=`%n`, then emits `memref.load %p[%n]`, erasing the
reinterpret_cast.

### Path 3: `cir.ptr_stride` array-decay consumer — `rewriteArrayDecay` (line 1745)

Called when the base of ptr_stride comes from a `cir.cast array_to_ptrdecay`
(detected by checking if adaptor.getBase() is a `memref.reinterpret_cast` with
the array base as its source).  It folds the two casts together:

```cpp
auto baseDefiningOp = adaptor.getBase().getDefiningOp<mlir::memref::ReinterpretCastOp>();
rewritePtrStrideToReinterpret(op, baseDefiningOp->getOperand(0), adaptor, rewriter);
rewriter.eraseOp(baseDefiningOp);
```

So `cast(array_to_ptrdecay) + ptr_stride %n` collapses to a single
`reinterpret_cast %arraybase, %n, [], []`.  This is what produces the clean
`memref.load %base[%n]` in tests like `ptrstride.cir`.

### Path 4: `cir.ptr_stride` complex fallback (line 1770)

Called when the pointee IS an array type (e.g., pointer to `!cir.array<!s32i x 8>`).
This is pointer-to-array stride, meaning each step advances by the full array size.
Currently emits a raw pointer arithmetic chain:

```
ptr.type_offset <element_type> : index
arith.index_cast stride : T to index
arith.muli stride_idx, elem_size : index
arith.muli result, inner_array_size : index  (if mulSize > 1)
memref.memory_space_cast base → generic_space
ptr.get_metadata base_generic
ptr.to_ptr base_generic
ptr.ptr_add ptr, offset_bytes
ptr.from_ptr new_ptr, metadata
memref.memory_space_cast result → original_space
```

This is the current fallback for patterns like:
```
%1 = cir.ptr_stride %p, %stride : (!cir.ptr<!cir.array<!s32i x 8>>, !s32i)
```

### Path 5: `cir.get_element` (line 1636)

```cpp
// Only rewrite if all users are load/stores/other get_elements.
if (!isLoadStoreOrGetProducer(op))
  return mlir::failure();

// ... index_cast %i to index ...

auto dstType = cast<mlir::MemRefType>(getTypeConverter()->convertType(op.getType()));
llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
prepareReinterpretMetadata(dstType, rewriter, sizes, strides, op);
rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
    op, dstType, adaptor.getBase(), index, sizes, strides);
```

Input:  `cir.get_element %base[%i] : (!cir.ptr<!cir.array<!s32i x 8>>, !s32i) -> !cir.ptr<!s32i>`
Adapted base: `memref<8xi32>`
Output: `memref.reinterpret_cast %base, %i, [?], [1] → memref<?xi32>`

The `isLoadStoreOrGetProducer` guard means this pattern returns `mlir::failure()`
if the result is used anywhere other than load/store/get_element — there is no
fallback, so such cases currently fail conversion entirely.

---

## The `findBaseAndIndices` / `eraseIfSafe` Mechanism

`CIRLoadOpLowering` and `CIRStoreOpLowering` both call:

```cpp
bool eraseIntermediateOp = findBaseAndIndices(adaptor.getAddr(), base, indices, eraseList, rewriter);
```

`findBaseAndIndices` (line 334):
- Walks a chain of `memref.reinterpret_cast` ops
- At each step: pushes `addrOp->getOperand(1)` (the dynamic offset value) into `indices`, moves to `addrOp->getOperand(0)` (the source), adds op to `eraseList`
- Also handles a `memref.cast` on top of an `alloca`/`get_global` (pushes constant 0)
- If no reinterpret_cast found at all: fills `indices` with zeros matching the base memref's rank
- Returns `true` if any intermediate ops were found

`eraseIfSafe` (line 374):
- Counts uses of the original addr value (`oldAddr`)
- Counts how many load/store users of `base` use the same forwarded index value
  (checks `strideVal == reinterpretOp.getOffsets()[0]` — both must be the same SSA Value)
- If counts match, erases all ops in `eraseList`

**Important**: `reinterpretOp.getOffsets()[0]` returns the dynamic offset as an
`mlir::Value` (the operand).  The comparison is SSA-value identity.  So
`findBaseAndIndices` pushes `addrOp->getOperand(1)` as the index, and
`eraseIfSafe` checks that the generated load uses that same value.

---

## Required Changes

### Change 1: `CIRCastOpLowering` — `array_to_ptrdecay` (line 1516)

**Current**: emits `reinterpret_cast %src, 0, [?], [1] → memref<?xT>`

**Target**: emit `memref.subview %src[0][%total_size][1] → memref<?xT>`

The size `%total_size` is the number of elements in dimension 0 of the source:
- For `memref<NxT>` (static): `arith.constant N : index`
- For `memref<?xT>` (dynamic): `memref.dim %src, 0`

```cpp
case CIR::array_to_ptrdecay: {
  auto newDstType = llvm::cast<mlir::MemRefType>(convertTy(dstType));
  auto srcMemref = llvm::cast<mlir::MemRefType>(src.getType());
  auto loc = op.getLoc();

  // Compute size of dimension 0
  mlir::Value size;
  int64_t staticSize = srcMemref.getShape()[0];
  if (mlir::ShapedType::isDynamic(staticSize)) {
    auto dimIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    size = mlir::memref::DimOp::create(rewriter, loc, src, dimIdx);
  } else {
    size = mlir::arith::ConstantIndexOp::create(rewriter, loc, staticSize);
  }

  mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
  mlir::Value one  = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

  // subview result type: memref<?xT, strided<[1], offset: 0>>
  // then reinterpret_cast to newDstType for type-converter compatibility
  llvm::SmallVector<mlir::OpFoldResult> svOffsets = {mlir::OpFoldResult(zero)};
  llvm::SmallVector<mlir::OpFoldResult> svSizes   = {mlir::OpFoldResult(size)};
  llvm::SmallVector<mlir::OpFoldResult> svStrides  = {mlir::OpFoldResult(one)};

  auto svType = mlir::memref::SubViewOp::inferResultType(
      srcMemref, svOffsets, svSizes, svStrides);
  auto sv = mlir::memref::SubViewOp::create(rewriter, loc,
      llvm::cast<mlir::MemRefType>(svType), src, svOffsets, svSizes, svStrides);

  // Wrap in reinterpret_cast to produce the exact newDstType expected downstream
  llvm::SmallVector<mlir::OpFoldResult> rcSizes, rcStrides;
  prepareReinterpretMetadata(newDstType, rewriter, rcSizes, rcStrides, op);
  rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
      op, newDstType, sv.getResult(), zero, rcSizes, rcStrides);
  return mlir::success();
}
```

**Why reinterpret_cast wrapper**: The type converter says `!cir.ptr<T>` →
`memref<?xT>` (identity layout).  `memref.subview` produces a result with a
strided layout attribute encoding the offset, which is a different MLIR type.
The `reinterpret_cast` with offset=0 converts back to the expected type.
`findBaseAndIndices` will peel the reinterpret_cast (extracting index=0),
leaving the subview alive in the final IR as the load's direct operand
(`memref.load %sv[0]`).

---

### Change 2: `CIRPtrStrideOpLowering::rewritePtrStrideToReinterpret` (line 1716)

**Current**:
```cpp
rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
    op, memrefType, base, stride, mlir::ValueRange{}, mlir::ValueRange{}, ...);
```

**Target**: emit `memref.subview %base[%stride][%remaining][1]` then wrap with
`reinterpret_cast` for type compatibility.

```cpp
mlir::LogicalResult rewritePtrStrideToReinterpret(
    cir::PtrStrideOp op, mlir::Value base, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto ptrType   = op.getType();
  auto memrefType = llvm::cast<mlir::MemRefType>(convertTy(ptrType));
  auto stride    = adaptor.getStride();
  auto indexType = rewriter.getIndexType();
  auto loc       = op.getLoc();

  if (stride.getType() != indexType)
    stride = mlir::arith::IndexCastOp::create(rewriter, loc, indexType, stride);

  // Compute remaining size = dim(base, 0) - stride
  auto baseMemref = llvm::cast<mlir::MemRefType>(base.getType());
  mlir::Value totalSize;
  int64_t staticSize = baseMemref.getShape()[0];
  if (mlir::ShapedType::isDynamic(staticSize)) {
    auto dimIdx  = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    totalSize    = mlir::memref::DimOp::create(rewriter, loc, base, dimIdx);
  } else {
    totalSize    = mlir::arith::ConstantIndexOp::create(rewriter, loc, staticSize);
  }
  mlir::Value remaining = mlir::arith::SubIOp::create(rewriter, loc, totalSize, stride);

  mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

  llvm::SmallVector<mlir::OpFoldResult> svOffsets = {mlir::OpFoldResult(stride)};
  llvm::SmallVector<mlir::OpFoldResult> svSizes   = {mlir::OpFoldResult(remaining)};
  llvm::SmallVector<mlir::OpFoldResult> svStrides  = {mlir::OpFoldResult(one)};

  auto svType = mlir::memref::SubViewOp::inferResultType(
      baseMemref, svOffsets, svSizes, svStrides);
  auto sv = mlir::memref::SubViewOp::create(rewriter, loc,
      llvm::cast<mlir::MemRefType>(svType), base, svOffsets, svSizes, svStrides);

  // Wrap for type compatibility
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
  llvm::SmallVector<mlir::OpFoldResult> rcSizes, rcStrides;
  prepareReinterpretMetadata(memrefType, rewriter, rcSizes, rcStrides, op);
  rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
      op, memrefType, sv.getResult(), zero, rcSizes, rcStrides);

  return mlir::success();
}
```

**Why `remaining = dim - stride`**: For the alias pass to detect Form A, the
`lo` subview must have `size = %n` and the `hi` subview must have `offset = %n`
where `%n` is the **same SSA value**.  If `lo` is a ptrdecay (offset=0,
size=%n) and `hi` is a ptr_stride (offset=%n), and `%n` is the same value
used both as `lo`'s size and `hi`'s offset, Form A fires.

---

### Change 3: `rewriteArrayDecay` (line 1745)

Currently this folds `cast(array_to_ptrdecay)` + `ptr_stride` into a single
`reinterpret_cast`.  After Change 1 and 2, the `cast(array_to_ptrdecay)` will
produce a `reinterpret_cast(subview(...))` chain.  The `rewriteArrayDecay`
function checks `adaptor.getBase().getDefiningOp<mlir::memref::ReinterpretCastOp>()`
to find the array base — it needs to also look through the subview wrapper.

**Option A** (simplest): check if the defining op is a `reinterpret_cast` whose
source is a `memref.subview`:
```cpp
auto baseReinterpret = adaptor.getBase().getDefiningOp<mlir::memref::ReinterpretCastOp>();
if (!baseReinterpret) return mlir::failure();

// The source of the reinterpret_cast may be a subview (after Change 1)
mlir::Value actualBase = baseReinterpret->getOperand(0);
if (auto sv = actualBase.getDefiningOp<mlir::memref::SubViewOp>())
  actualBase = sv.getSource();
```

**Option B** (cleaner long-term): remove `rewriteArrayDecay` entirely and let
the separate cast + ptr_stride patterns each emit their own subviews.  The
alias pass then sees two independent subviews of the same original array base,
which is exactly what it needs.

Option B is preferred because it makes the subview structure more explicit.
The `rewriteArrayDecay` optimisation was a codegen trick to collapse two
temporary reinterpret_casts; with subviews as a semantic representation it
is no longer beneficial.

---

### Change 4: `CIRPtrStrideOpLowering` complex fallback (line 1770)

**Current**: when the pointee is an array type (`!cir.ptr<!cir.array<T x N>>`),
emits `ptr.type_offset` + `ptr.to_ptr` + `ptr.ptr_add` + `ptr.from_ptr` byte
arithmetic.

**Target**: emit a rank-preserving `memref.subview` on dimension 0.

The adapted base for `!cir.ptr<!cir.array<T x N>>` is `memref<NxT>`.  A
ptr_stride by `%s` advances `%s` rows.  The correct subview:

```mlir
%remaining_rows = arith.subi %total_rows, %stride
%sv = memref.subview %base[%stride, 0][%remaining_rows, N][1, 1]
    : memref<?xNxT> to memref<?xNxT, strided<[N, 1], offset: ?>>
```

For a 1D array pointee `memref<NxT>` (rank 1), this degenerates to the same
pattern as Change 2.

The byte arithmetic fallback is only needed when the memref cannot be used
directly (e.g., opaque pointers, address-space casts).  For the common case of
a CIR array pointer, the subview is correct and far cleaner.

---

### Change 5: `CIRGetElementOpLowering` (line 1636)

**Current**: restricted to load/store users only; emits `reinterpret_cast`.

**Target**: remove the `isLoadStoreOrGetProducer` guard; emit `memref.subview`.

```cpp
mlir::LogicalResult
matchAndRewrite(cir::GetElementOp op, OpAdaptor adaptor,
                mlir::ConversionPatternRewriter &rewriter) const override {
  // NOTE: removed isLoadStoreOrGetProducer guard

  auto index    = adaptor.getIndex();
  auto indexType = rewriter.getIndexType();
  auto loc      = op.getLoc();
  if (index.getType() != indexType)
    index = mlir::arith::IndexCastOp::create(rewriter, loc, indexType, index);

  auto dstType  = cast<mlir::MemRefType>(getTypeConverter()->convertType(op.getType()));
  auto elemType = dstType.getElementType();
  auto baseMemref = llvm::cast<mlir::MemRefType>(adaptor.getBase().getType());

  mlir::Value one  = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

  llvm::SmallVector<mlir::OpFoldResult> svOffsets = {mlir::OpFoldResult(index)};
  llvm::SmallVector<mlir::OpFoldResult> svSizes   = {mlir::OpFoldResult(one)};
  llvm::SmallVector<mlir::OpFoldResult> svStrides  = {mlir::OpFoldResult(one)};

  auto svType = mlir::memref::SubViewOp::inferResultType(
      baseMemref, svOffsets, svSizes, svStrides);
  auto sv = mlir::memref::SubViewOp::create(rewriter, loc,
      llvm::cast<mlir::MemRefType>(svType), adaptor.getBase(),
      svOffsets, svSizes, svStrides);

  // Wrap in reinterpret_cast to produce dstType (memref<?xT>)
  llvm::SmallVector<mlir::OpFoldResult> rcSizes, rcStrides;
  prepareReinterpretMetadata(dstType, rewriter, rcSizes, rcStrides, op);
  rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
      op, dstType, sv.getResult(), zero, rcSizes, rcStrides);

  return mlir::success();
}
```

The subview here has size=1 (single-element view), which is correct: `get_element`
accesses exactly one element.

---

## How `findBaseAndIndices` and `eraseIfSafe` Interact With the New Pattern

After the changes, the chain from a ptr_stride or get_element looks like:

```
%sv = memref.subview %base[%offset][%size][1]  → memref<...xT, strided<[1], offset: ?>>
%rc = memref.reinterpret_cast %sv, %zero, [?], [1]  → memref<?xT>
cir.load uses %rc
```

`CIRLoadOpLowering` calls `findBaseAndIndices(%rc, ...)`:
1. Sees `reinterpret_cast`: pushes `operand(1)` = `%zero` as index, moves to `%sv`
2. `%sv` is not a `reinterpret_cast`: loop exits
3. Returns `base = %sv`, `indices = [%zero]`, `eraseList = [%rc]`

Then emits: `memref.load %sv[%zero]`

`eraseIfSafe` checks: does the load's index (`%zero`) equal `%rc.getOffsets()[0]`
(also `%zero`, the same SSA value)? Yes → erases `%rc`.

**Result**: the `reinterpret_cast` is erased. The `memref.subview` remains in
the IR as the direct operand of the load.  This is what the `MarkAliasGroups`
pass needs to see.

**No changes needed** to `findBaseAndIndices` or `eraseIfSafe` themselves.

---

## Case-by-Case MLIR Output After Changes

### Case 1: `dynamic_split` — `lo = A[0..n-1]`, `hi = A[n..]`

C source:
```c
float *lo = A;
float *hi = A + n;
// loop: hi[i] += lo[0]
```

CIR (approximately):
```
%base = cir.get_global @A
%lo_ptr = cir.cast array_to_ptrdecay %base
%hi_ptr = cir.ptr_stride %lo_ptr, %n
```

After changes, MLIR output:
```mlir
%base = memref.get_global @A : memref<2048xf32>
%n_idx = arith.index_cast %n : i64 to index
%total = arith.constant 2048 : index

// lo subview: offset=0, size=%n_idx
%lo_sv = memref.subview %base[0][%n_idx][1] : memref<2048xf32> to memref<?xf32, strided<[1], offset: 0>>
%lo_rc = memref.reinterpret_cast %lo_sv, %c0, [?], [1] → memref<?xf32>

// hi subview: offset=%n_idx  ← SAME SSA value as lo's size → Form A fires
%hi_remaining = arith.subi %total, %n_idx
%hi_sv = memref.subview %base[%n_idx][%hi_remaining][1] : memref<2048xf32> to memref<?xf32, strided<[1], offset: ?>>
%hi_rc = memref.reinterpret_cast %hi_sv, %c0, [?], [1] → memref<?xf32>

// load lo[0] → memref.load %lo_sv[0]  (rc erased)
// load hi[i] → memref.load %hi_sv[%i]  (rc erased)
```

`MarkAliasGroups` sees: two subviews of `%base`, `hi_sv.offset[0]` IS `lo_sv.size[0]`
(both `%n_idx`), `lo_sv.offset[0]` == 0. **Form A fires.**

### Case 2: `adjacent_tiles` — tile boundary

C source:
```c
float *src = A + tile * N;       // offset = tile*N, size = N
float *dst = A + (tile+1) * N;   // offset = tile*N + N
```

CIR:
```
%src_ptr = cir.ptr_stride %A_decayed, %tile_times_N
%dst_ptr = cir.ptr_stride %A_decayed, %tile_times_N_plus_N
```

After changes:
```mlir
%src_sv = memref.subview %base[%tile_N][%N][1]
%dst_offset = arith.addi %tile_N, %N
%dst_sv = memref.subview %base[%dst_offset][%remaining][1]
```

`MarkAliasGroups` sees: `dst_sv.offset[0] = arith.addi(src_sv.offset[0], src_sv.size[0])`.
**Form B fires.**

### Case 3: `matrix_row_split` — 2D

C source:
```c
float (*lo)[512] = M;        // rows 0..n_lo-1
float (*hi)[512] = M + n_lo; // rows n_lo..
```

Adapted `%M` base is `memref<?x512xf32>`.  CIR ptr_stride on row-pointer:

```mlir
%lo_sv = memref.subview %M[0][%n_lo][1]  : memref<?x512xf32> to memref<?x512xf32, ...>
%hi_sv = memref.subview %M[%n_lo][%rem][1]: memref<?x512xf32> to memref<?x512xf32, ...>
```

Form A on dim 0: `hi.offset[0]` IS `lo.size[0]` (both `%n_lo`).

Note: the inner dimension (512) is inherited automatically from the source memref.
The `MarkAliasGroups` pass checks only dimension 0 (documented limitation).

---

## Test Files to Update

These existing tests check for `reinterpret_cast` in the output and will need
updating:

| Test file | What changes |
|-----------|-------------|
| `clang/test/CIR/Lowering/ThroughMLIR/ptrstride.cir` | 1D array access: `memref.load %base[%idx]` becomes `memref.subview + memref.load %sv[0]` |
| `clang/test/CIR/Lowering/ThroughMLIR/ptrstride-ptr.cir` | Raw pointer stride and complex array pointer |
| `clang/test/CIR/Lowering/ThroughMLIR/array.cir` | Array alloca (may not change) |
| `clang/test/CIR/Lowering/ThroughMLIR/memref.cir` | Simple alloca + load/store |

New tests to add (to validate the alias-pass inputs):
- `clang/test/CIR/Lowering/ThroughMLIR/subview-dynamic-split.cir` — Form A check
- `clang/test/CIR/Lowering/ThroughMLIR/subview-adjacent-tiles.cir` — Form B check
- `clang/test/CIR/Lowering/ThroughMLIR/subview-matrix-rows.cir` — 2D Form A check

---

## Build and Test Commands

```bash
# From the clangir build directory (assumed: build/)
ninja cir-opt

# Run a specific lowering test
cir-opt clang/test/CIR/Lowering/ThroughMLIR/ptrstride.cir \
    --cir-to-mlir | FileCheck clang/test/CIR/Lowering/ThroughMLIR/ptrstride.cir \
    -check-prefix=MLIR

# Run full lowering test suite
llvm-lit clang/test/CIR/Lowering/ThroughMLIR/
```

---

## Key API Notes

- **`mlir::memref::SubViewOp::inferResultType`**: given source MemRefType and
  OpFoldResult offsets/sizes/strides, returns the correct result MemRefType
  with the appropriate strided layout attribute.  Use this rather than
  constructing the result type manually.

- **`mlir::StridedLayoutAttr::get(ctx, offset, strides)`**: if you need to
  construct the strided layout manually; `offset = mlir::ShapedType::kDynamic`
  means dynamic.

- **`memref.dim %v, %c0`**: returns the size of dimension 0 at runtime.
  For a static-shape memref, MLIR will constant-fold this.

- **`OpFoldResult`**: wraps either an `mlir::Value` (dynamic) or an
  `mlir::IntegerAttr` (static constant).  Use `rewriter.getIndexAttr(N)` for
  static, `mlir::OpFoldResult(value)` for dynamic.

- **`OpBuilder::create` vs deprecated `OpBuilder::create<Op>`**: In LLVM 22,
  use `OpTy::create(rewriter, loc, ...)` not `rewriter.create<OpTy>(loc, ...)`.
  Both `LLVM::LoadOp::create` and `LLVM::StoreOp::create` have been updated;
  follow the same pattern for `SubViewOp::create`, `DimOp::create`, etc.

- **`prepareReinterpretMetadata`** (line 420): fills sizes/strides from a
  `MemRefType`'s shape and layout.  For `memref<?xT>` (identity layout):
  `sizes = [kDynamic]`, `strides = [1]`, `layoutOffset = 0`.  Call this for
  the wrapper `reinterpret_cast` after each subview.

---

## Summary of Changes Table

| Location | Current | Target |
|----------|---------|--------|
| `CIRCastOpLowering` `array_to_ptrdecay` (~line 1516) | `reinterpret_cast %src, 0, [?], [1]` | `subview %src[0][dim0_size][1]` + `reinterpret_cast` wrapper |
| `CIRPtrStrideOpLowering::rewritePtrStrideToReinterpret` (~line 1716) | `reinterpret_cast %base, %stride, [], []` | `subview %base[%stride][dim0-stride][1]` + `reinterpret_cast` wrapper |
| `CIRPtrStrideOpLowering::rewriteArrayDecay` (~line 1745) | folds two casts into one `reinterpret_cast` | remove or update to look through subview |
| `CIRPtrStrideOpLowering` complex fallback (~line 1770) | `ptr.type_offset` + byte arithmetic | rank-preserving `subview` on dim 0 |
| `CIRGetElementOpLowering` (~line 1636) | `reinterpret_cast %base, %i, [?], [1]`; fails if non-load/store users | `subview %base[%i][1][1]` + `reinterpret_cast` wrapper; no user restriction |
| `findBaseAndIndices` (line 334) | handles `reinterpret_cast` chain | **no change needed** — still peels the wrapper `reinterpret_cast` |
| `eraseIfSafe` (line 374) | erases `reinterpret_cast` ops | **no change needed** — erases wrapper; subview stays alive |
