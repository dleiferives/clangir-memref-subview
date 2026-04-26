# CIR Global Lowering: The Opaque Pointer Crash

## The Crash

Running `cir-opt correlation.cir --cir-to-mlir` aborts with:

```
<unknown>:0: error: invalid memref element type
Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed.
  #10 mlir::MemRefType::get(...)          BuiltinTypes.cpp:579
  #9  CIRGlobalOpLowering::matchAndRewrite  LowerCIRToMLIR.cpp:1387
```

## The Offending Global

```
cir.global "private" external @stderr : !cir.ptr<!rec__IO_FILE>
```

`!rec__IO_FILE` is a record (struct) type. The type converter has no rule for
it, so it returns null. The `PointerType` converter catches null pointee
conversions and falls back to `!llvm.ptr`:

```cpp
// LowerCIRToMLIR.cpp:2251-2263
converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    auto ty = convertTypeForMemory(converter, type.getPointee());
    if (!ty)
        return mlir::LLVM::LLVMPointerType::get(type.getContext()); // <-- fallback
    ...
    return mlir::MemRefType::get({ShapedType::kDynamic}, ty, ...);
});
```

This fallback was added for variadic call lowering (passing `FILE*` to
`fprintf`). It works there. It does not work in `CIRGlobalOpLowering`, which
gets `!llvm.ptr` back as `convertedType` and then tries:

```cpp
// LowerCIRToMLIR.cpp:1387
memrefType = mlir::MemRefType::get({1}, convertedType, ...);
//                                      ^^^^^^^^^^^^^ = !llvm.ptr
```

`!llvm.ptr` is not a valid memref element type (`BaseMemRefType::isValidElementType`
returns false for it), so the assertion inside `MemRefType::get` fires.

## Why `!llvm.ptr` Is Not a Valid Memref Element Type

`BaseMemRefType::isValidElementType` (BuiltinTypes.h:413) accepts:

- int, index, or float scalars
- `ComplexType`, `MemRefType`, `VectorType`, `UnrankedMemRefType`
- anything implementing `MemRefElementTypeInterface`

`LLVM::LLVMPointerType` satisfies none of these.

## The Cascading Problem in CIRGetGlobalOpLowering

`cir.get_global @stderr` has type `!cir.ptr<!cir.ptr<!rec__IO_FILE>>` — the
address of the global, which is a pointer-to-pointer. Converting the outer
pointer type hits the same fallback: its pointee (`!cir.ptr<!rec__IO_FILE>`)
converts to `!llvm.ptr`, which is not a valid memref element, so the outer
pointer also degrades to `!llvm.ptr`.

`CIRGetGlobalOpLowering` then does:

```cpp
// LowerCIRToMLIR.cpp:1489
mlir::MemRefType resultTypeMemref = mlir::cast<mlir::MemRefType>(resultType);
```

`resultType` is `!llvm.ptr`. `mlir::cast` to `MemRefType` aborts.

## What the Globals Actually Are

There are 8 `cir.global` ops in `correlation.cir`:

| Global | CIR Type | Converts? |
|--------|----------|-----------|
| `@stderr` | `!cir.ptr<!rec__IO_FILE>` | No — struct pointer, no rule |
| `@".str"` … `@".str.6"` | `!cir.array<!s8i x N>` | Yes — `memref<N x i8>` |

`@stderr` is the only broken one. It is an external symbol with no initializer.
It is used exclusively in the `POLYBENCH_DUMP_ARRAYS` output block — irrelevant
to the computational kernel and any affine analysis.

## Why `!llvm.ptr` Appears at All

The `PointerType` converter's `!llvm.ptr` fallback was introduced to support
variadic call lowering (`fprintf`, `printf`, etc.) where operands typed as
`!cir.ptr<struct>` need to become something LLVM-ABI-compatible. It is a
CIR-to-LLVM concern sitting inside a CIR-to-MLIR pass. The fallback leaks
`!llvm.ptr` into the type system whenever any pointer-to-struct is touched,
including globals.

## Relevant Code

| Location | Lines | Role |
|----------|-------|------|
| `LowerCIRToMLIR.cpp` | 1366–1465 | `CIRGlobalOpLowering::matchAndRewrite` |
| `LowerCIRToMLIR.cpp` | 1467–1511 | `CIRGetGlobalOpLowering::matchAndRewrite` |
| `LowerCIRToMLIR.cpp` | 2251–2263 | `PointerType` converter, `!llvm.ptr` fallback |
| `LowerCIRToMLIR.cpp` | 2349–2372 | Conversion target setup, `applyPartialConversion` |
| `mlir/include/mlir/IR/BuiltinTypes.h` | 413–418 | `BaseMemRefType::isValidElementType` |
