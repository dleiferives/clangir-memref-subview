//====- LowerCIRToMLIR.cpp - Lowering from CIR to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to MLIR.
//
//===----------------------------------------------------------------------===//

#include "LowerToMLIRHelpers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;
using namespace llvm;

namespace cir {

// Mirrors cir::direct::convertLinkage from DirectToLLVM. Kept local here so
// the ThroughMLIR lowering does not need to pull in DirectToLLVM's internals
// just to emit an `llvm.mlir.global` for an opaque-pointer global.
static mlir::LLVM::Linkage
convertCIRLinkageToLLVMForMLIR(cir::GlobalLinkageKind linkage) {
  using CIR = cir::GlobalLinkageKind;
  using L = mlir::LLVM::Linkage;
  switch (linkage) {
  case CIR::AppendingLinkage:         return L::Appending;
  case CIR::AvailableExternallyLinkage: return L::AvailableExternally;
  case CIR::CommonLinkage:            return L::Common;
  case CIR::ExternalLinkage:          return L::External;
  case CIR::ExternalWeakLinkage:      return L::ExternWeak;
  case CIR::InternalLinkage:          return L::Internal;
  case CIR::LinkOnceAnyLinkage:       return L::Linkonce;
  case CIR::LinkOnceODRLinkage:       return L::LinkonceODR;
  case CIR::PrivateLinkage:           return L::Private;
  case CIR::WeakAnyLinkage:           return L::Weak;
  case CIR::WeakODRLinkage:           return L::WeakODR;
  }
  llvm_unreachable("unhandled cir::GlobalLinkageKind");
}

// Convert a memref value to an !llvm.ptr by extracting its aligned pointer.
static mlir::Value memrefToLLVMPtr(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value memrefVal) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto rawIdx = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
      builder, loc, memrefVal);
  auto rawI64 = mlir::arith::IndexCastUIOp::create(builder, loc,
                                                   builder.getI64Type(), rawIdx);
  return mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy,
                                        mlir::ValueRange{rawI64});
}

class CIRReturnLowering : public mlir::OpConversionPattern<cir::ReturnOp> {
public:
  using OpConversionPattern<cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op->getParentOfType<mlir::LLVM::LLVMFuncOp>())
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op,
                                                        adaptor.getOperands());
    else
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                        adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::math::MathDialect,
                    mlir::ptr::PtrDialect, mlir::vector::VectorDialect,
                    mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();
  }
  void runOnOperation() final;

  StringRef getDescription() const override {
    return "Convert the CIR dialect module to MLIR standard dialects";
  }

  StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRCallOpLowering : public mlir::OpConversionPattern<cir::CallOp> {
public:
  using OpConversionPattern<cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return mlir::failure();

    if (!op.isIndirect()) {
      // Currently variadic functions are not supported by the builtin func
      // dialect. For now only basic call to printf are supported by using the
      // llvmir dialect.
      // TODO: remove this and add support for variadic function calls once
      // TODO: supported by the func dialect
      if (op.getCallee()->equals_insensitive("printf")) {
        SmallVector<mlir::Type> operandTypes =
            llvm::to_vector(adaptor.getOperands().getTypes());

        // Drop the initial memref operand type (we replace the memref format
        // string with equivalent llvm.mlir ops)
        operandTypes.erase(operandTypes.begin());

        // Check that the printf attributes can be used in llvmir dialect (i.e
        // they have integer/float type)
        if (!llvm::all_of(operandTypes, [](mlir::Type ty) {
              return mlir::LLVM::isCompatibleType(ty);
            })) {
          return op.emitError()
                 << "lowering of printf attributes having a type that is "
                    "converted to memref in cir-to-mlir lowering (e.g. "
                    "pointers) not supported yet";
        }

        // Currently only versions of printf are supported where the format
        // string is defined inside the printf ==> the lowering of the cir ops
        // will match:
        // %global = memref.get_global %frm_str
        // %* = memref.reinterpret_cast (%global, 0)
        if (auto reinterpret_castOP =
                adaptor.getOperands()[0]
                    .getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
          if (auto getGlobalOp =
                  reinterpret_castOP->getOperand(0)
                      .getDefiningOp<mlir::memref::GetGlobalOp>()) {
            mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

            auto context = rewriter.getContext();

            // Find the memref.global op defining the frm_str
            auto globalOp = parentModule.lookupSymbol<mlir::memref::GlobalOp>(
                getGlobalOp.getNameAttr());

            rewriter.setInsertionPoint(globalOp);

            // Insert a equivalent llvm.mlir.global
            auto initialvalueAttr =
                mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                    globalOp.getInitialValueAttr());

            auto type = mlir::LLVM::LLVMArrayType::get(
                mlir::IntegerType::get(context, 8),
                initialvalueAttr.getNumElements());

            auto llvmglobalOp = mlir::LLVM::GlobalOp::create(
                rewriter, globalOp->getLoc(), type, true,
                mlir::LLVM::Linkage::Internal,
                "printf_format_" + globalOp.getSymName().str(),
                initialvalueAttr, 0);

            rewriter.setInsertionPoint(getGlobalOp);

            // Insert llvmir dialect ops to retrive the !llvm.ptr of the global
            auto globalPtrOp = mlir::LLVM::AddressOfOp::create(
                rewriter, getGlobalOp->getLoc(), llvmglobalOp);

            mlir::Value cst0 = mlir::LLVM::ConstantOp::create(
                rewriter, getGlobalOp->getLoc(), rewriter.getI8Type(),
                rewriter.getIndexAttr(0));
            auto gepPtrOp = mlir::LLVM::GEPOp::create(
                rewriter, getGlobalOp->getLoc(),
                mlir::LLVM::LLVMPointerType::get(context),
                llvmglobalOp.getType(), globalPtrOp,
                ArrayRef<mlir::Value>({cst0, cst0}));

            mlir::ValueRange operands = adaptor.getOperands();

            // Replace the old memref operand with the !llvm.ptr for the frm_str
            mlir::SmallVector<mlir::Value> newOperands;
            newOperands.push_back(gepPtrOp);
            newOperands.append(operands.begin() + 1, operands.end());

            // Create the llvmir dialect function type for printf
            auto llvmI32Ty = mlir::IntegerType::get(context, 32);
            auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
            auto llvmFnType =
                mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                  /*isVarArg=*/true);

            rewriter.setInsertionPoint(op);

            // Insert an llvm.call op with the updated operands to printf
            rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
                op, llvmFnType, op.getCalleeAttr(), newOperands);

            // Cleanup printf frm_str memref ops
            rewriter.eraseOp(reinterpret_castOP);
            rewriter.eraseOp(getGlobalOp);
            rewriter.eraseOp(globalOp);

            return mlir::LogicalResult::success();
          }
        }

        return op.emitError()
               << "lowering of printf function with Format-String"
                  "defined outside of printf is not supported yet";
      }

      // General handler for extern variadic calls (fprintf, polybench timers,
      // etc.). The callee may still be a cir.func (if visited before
      // CIRFuncOpLowering) or already rewritten to an llvm.func; handle both.
      if (auto callee = op.getCallee()) {
        auto parentModule = op->getParentOfType<mlir::ModuleOp>();
        bool isVarArg = false;
        cir::FuncType cirFnType;
        if (auto cirFn = parentModule.lookupSymbol<cir::FuncOp>(*callee)) {
          cirFnType = cirFn.getFunctionType();
          isVarArg = cirFnType.isVarArg();
        } else if (auto llvmFn =
                       parentModule.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
                           *callee)) {
          isVarArg = llvmFn.isVarArg();
        }
        if (isVarArg) {
          auto *context = rewriter.getContext();
          auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);

            // Rebuild LLVM function type (mirrors CIRFuncOpLowering logic).
            // When the original cir.func is already gone we fall back to
            // building the signature from the call's operand/result types,
            // since they carry the same information for the C ABI.
            SmallVector<mlir::Type> llvmParams;
            SmallVector<mlir::Type> paramTypesStorage;
            if (cirFnType)
              llvm::append_range(paramTypesStorage, cirFnType.getInputs());
            else
              llvm::append_range(paramTypesStorage, op.getOperandTypes());
            mlir::ArrayRef<mlir::Type> paramTypes = paramTypesStorage;
            for (auto paramType : paramTypes) {
              auto converted = typeConverter->convertType(paramType);
              if (!converted)
                return mlir::failure();
              llvmParams.push_back(mlir::LLVM::isCompatibleType(converted)
                                       ? converted
                                       : llvmPtrTy);
            }
            mlir::Type retType;
            mlir::Type cirRetType =
                cirFnType ? cirFnType.getReturnType() : mlir::Type{};
            if (!cirRetType && op.getNumResults() > 0)
              cirRetType = op->getResult(0).getType();
            if (!cirRetType || isa<cir::VoidType>(cirRetType))
              retType = mlir::LLVM::LLVMVoidType::get(context);
            else {
              auto converted = typeConverter->convertType(cirRetType);
              retType = (converted && mlir::LLVM::isCompatibleType(converted))
                            ? converted
                            : mlir::LLVM::LLVMVoidType::get(context);
            }
            auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(
                retType, llvmParams, /*isVarArg=*/true);

            // Convert each operand to an LLVM-compatible type.
            SmallVector<mlir::Value> newOperands;
            SmallVector<mlir::Operation *> toErase;
            for (auto convVal : adaptor.getOperands()) {
              auto ty = convVal.getType();
              if (mlir::LLVM::isCompatibleType(ty)) {
                // Already LLVM-compatible (llvm.ptr, i32, f64, …).
                newOperands.push_back(convVal);
              } else if (mlir::isa<mlir::MemRefType>(ty)) {
                // Check for inline string-literal pattern:
                //   memref.reinterpret_cast(memref.get_global(@str))
                if (auto rcOp =
                        convVal.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
                  if (auto ggOp =
                          rcOp->getOperand(0)
                              .getDefiningOp<mlir::memref::GetGlobalOp>()) {
                    auto globalOp = parentModule.lookupSymbol<mlir::memref::GlobalOp>(
                        ggOp.getNameAttr());
                    if (!globalOp)
                      return op.emitError()
                             << "cannot find global " << ggOp.getName()
                             << " for variadic string arg";
                    auto initAttr =
                        mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                            globalOp.getInitialValueAttr());
                    if (!initAttr)
                      return op.emitError()
                             << "unsupported initializer for variadic string arg";

                    rewriter.setInsertionPoint(globalOp);
                    auto llvmArrTy = mlir::LLVM::LLVMArrayType::get(
                        mlir::IntegerType::get(context, 8),
                        initAttr.getNumElements());
                    auto llvmGlobal = mlir::LLVM::GlobalOp::create(
                        rewriter, globalOp->getLoc(), llvmArrTy, true,
                        mlir::LLVM::Linkage::Internal,
                        "vararg_str_" + globalOp.getSymName().str(), initAttr,
                        0);

                    rewriter.setInsertionPoint(ggOp);
                    auto addrOf = mlir::LLVM::AddressOfOp::create(
                        rewriter, ggOp->getLoc(), llvmGlobal);
                    mlir::Value cst0 = mlir::LLVM::ConstantOp::create(
                        rewriter, ggOp->getLoc(), rewriter.getI8Type(),
                        rewriter.getIndexAttr(0));
                    auto gep = mlir::LLVM::GEPOp::create(
                        rewriter, ggOp->getLoc(), llvmPtrTy,
                        llvmGlobal.getType(), addrOf,
                        ArrayRef<mlir::Value>({cst0, cst0}));

                    rewriter.setInsertionPoint(op);
                    newOperands.push_back(gep);
                    toErase.push_back(rcOp);
                    toErase.push_back(ggOp);
                    toErase.push_back(globalOp);
                    continue;
                  }
                }
                // Generic memref → extract aligned pointer.
                rewriter.setInsertionPoint(op);
                newOperands.push_back(
                    memrefToLLVMPtr(rewriter, op.getLoc(), convVal));
              } else {
                return op.emitError()
                       << "unsupported operand type for variadic call: " << ty;
              }
            }

            rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
                op, llvmFnType, op.getCalleeAttr(), newOperands);
            for (auto *deadOp : llvm::reverse(toErase))
              rewriter.eraseOp(deadOp);
            return mlir::LogicalResult::success();
        }
      }

      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, op.getCalleeAttr(), types, adaptor.getOperands());
      return mlir::LogicalResult::success();

    } else {
      // Indirect call. The first CIR operand is the function pointer; the
      // rest are the call arguments. After type conversion the function
      // pointer is expected to be `!llvm.ptr` (the PointerType converter
      // falls back to that for `!cir.ptr<!cir.func<...>>`). Emit `llvm.call`
      // with the pointer as the callee.
      //
      // This path handles both fixed-arity and variadic callees (e.g.
      // polybench's `no_proto` timer functions).
      auto operands = adaptor.getOperands();
      if (operands.empty())
        return op.emitError() << "indirect call with no callee operand";
      mlir::Value callee = operands.front();
      auto calleeTy = callee.getType();
      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(calleeTy))
        return op.emitError()
               << "indirect call callee expected to be !llvm.ptr after "
                  "conversion, got "
               << calleeTy;

      // Recover the LLVM function type from the original CIR pointer-to-func
      // operand. `op.getOperand(0)` is the pre-conversion function pointer.
      auto cirCalleeTy =
          mlir::dyn_cast<cir::PointerType>(op.getOperand(0).getType());
      if (!cirCalleeTy)
        return op.emitError() << "indirect callee is not a cir.ptr";
      auto cirFnTy = mlir::dyn_cast<cir::FuncType>(cirCalleeTy.getPointee());
      if (!cirFnTy)
        return op.emitError() << "indirect callee does not point to a func";

      auto *context = rewriter.getContext();
      auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);

      SmallVector<mlir::Type> llvmParams;
      for (auto paramType : cirFnTy.getInputs()) {
        auto converted = typeConverter->convertType(paramType);
        if (!converted)
          return mlir::failure();
        llvmParams.push_back(mlir::LLVM::isCompatibleType(converted)
                                 ? converted
                                 : llvmPtrTy);
      }
      mlir::Type retType;
      auto cirRetType = cirFnTy.getReturnType();
      if (!cirRetType || mlir::isa<cir::VoidType>(cirRetType))
        retType = mlir::LLVM::LLVMVoidType::get(context);
      else {
        auto converted = typeConverter->convertType(cirRetType);
        retType = (converted && mlir::LLVM::isCompatibleType(converted))
                      ? converted
                      : mlir::LLVM::LLVMVoidType::get(context);
      }
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(
          retType, llvmParams, /*isVarArg=*/cirFnTy.isVarArg());

      // For an indirect LLVM::CallOp the function pointer is the first entry
      // in the operand list and the callee symbol attribute is null.
      SmallVector<mlir::Value> allOperands;
      allOperands.reserve(operands.size());
      allOperands.push_back(callee);
      allOperands.append(operands.begin() + 1, operands.end());

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, llvmFnType, mlir::FlatSymbolRefAttr{}, allOperands);
      return mlir::success();
    }
  }
};

/// Given a type convertor and a data layout, convert the given type to a type
/// that is suitable for memory operations. For example, this can be used to
/// lower cir.bool accesses to i8.
static mlir::Type convertTypeForMemory(const mlir::TypeConverter &converter,
                                       mlir::Type type) {
  // TODO(cir): Handle other types similarly to clang's codegen
  // convertTypeForMemory
  if (isa<cir::BoolType>(type)) {
    // TODO: Use datalayout to get the size of bool
    return mlir::IntegerType::get(type.getContext(), 8);
  }

  return converter.convertType(type);
}

/// Emits the value from memory as expected by its users. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitFromMemory(mlir::ConversionPatternRewriter &rewriter,
                                  cir::LoadOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitFromMemory
  if (isa<cir::BoolType>(op.getType())) {
    // Create trunc of value from i8 to i1
    // TODO: Use datalayout to get the size of bool
    assert(value.getType().isInteger(8));
    return createIntCast(rewriter, value, rewriter.getI1Type());
  }

  return value;
}

/// Emits a value to memory with the expected scalar type. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitToMemory(mlir::ConversionPatternRewriter &rewriter,
                                cir::StoreOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitToMemory
  if (isa<cir::BoolType>(op.getValue().getType())) {
    // Create zext of value from i1 to i8
    // TODO: Use datalayout to get the size of bool
    return createIntCast(rewriter, value, rewriter.getI8Type());
  }

  return value;
}

class CIRAllocaOpLowering : public mlir::OpConversionPattern<cir::AllocaOp> {
public:
  using OpConversionPattern<cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Type mlirType =
        convertTypeForMemory(*getTypeConverter(), adaptor.getAllocaType());

    // FIXME: Some types can not be converted yet (e.g. struct)
    if (!mlirType)
      return mlir::LogicalResult::failure();

    auto memreftype = mlir::dyn_cast<mlir::MemRefType>(mlirType);
    if (mlir::isa<cir::ArrayType>(adaptor.getAllocaType())) {
      if (!memreftype)
        return mlir::LogicalResult::failure();
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(
          op, memreftype, op.getAlignmentAttr());
    } else {
      auto targetType = mlir::cast<mlir::MemRefType>(
          getTypeConverter()->convertType(op.getResult().getType()));
      memreftype = mlir::MemRefType::get({1}, targetType.getElementType(),
                                         targetType.getLayout(),
                                         targetType.getMemorySpace());
      auto allocaOp = mlir::memref::AllocaOp::create(
          rewriter, op.getLoc(), memreftype, op.getAlignmentAttr());
      // Cast from memref<1xMlirType> to memref<?xMlirType>
      // This is needed since Typeconverter produces memref<?xMlirType> for
      // non-array cir.ptrs, The cast will be eliminated later in
      // load/store-lowering.
      auto castOp = mlir::memref::CastOp::create(rewriter, op.getLoc(),
                                                 targetType, allocaOp);
      rewriter.replaceOp(op, castOp);
    }
    return mlir::LogicalResult::success();
  }
};

// Find base and indices from memref.reinterpret_cast
// and put it into eraseList.
static bool findBaseAndIndices(mlir::Value addr, mlir::Value &base,
                               SmallVector<mlir::Value> &indices,
                               SmallVector<mlir::Operation *> &eraseList,
                               mlir::ConversionPatternRewriter &rewriter) {
  while (mlir::Operation *addrOp =
             addr.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
    // Use getMixedOffsets() rather than getOperand(1) so that both static
    // (IndexAttr) and dynamic (Value) offsets are handled correctly.
    auto rcOp = mlir::cast<mlir::memref::ReinterpretCastOp>(addrOp);
    auto mixedOffsets = rcOp.getMixedOffsets();
    assert(!mixedOffsets.empty() && "reinterpret_cast must have an offset");
    if (auto val =
            llvm::dyn_cast_if_present<mlir::Value>(mixedOffsets[0])) {
      indices.push_back(val);
    } else {
      auto intAttr =
          mlir::cast<mlir::IntegerAttr>(mixedOffsets[0].get<mlir::Attribute>());
      indices.push_back(mlir::arith::ConstantIndexOp::create(
          rewriter, addrOp->getLoc(), intAttr.getInt()));
    }
    addr = addrOp->getOperand(0);
    eraseList.push_back(addrOp);
  }
  if (auto castOp = addr.getDefiningOp<mlir::memref::CastOp>()) {
    auto castInput = castOp->getOperand(0);
    if (castInput.getDefiningOp<mlir::memref::AllocaOp>() ||
        castInput.getDefiningOp<mlir::memref::GetGlobalOp>()) {
      // AllocaOp and GetGlobalOp-lowerings produce 1-element memrefs
      indices.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, castOp.getLoc(), 0));
      addr = castInput;
      eraseList.push_back(castOp);
    }
  }
  base = addr;
  if (indices.size() == 0) {
    auto memrefType = mlir::cast<mlir::MemRefType>(base.getType());
    auto rank = memrefType.getRank();
    indices.reserve(rank);
    for (unsigned d = 0; d < rank; ++d) {
      mlir::Value zero = mlir::arith::ConstantIndexOp::create(
          rewriter, base.getLoc(), /*value=*/0);
      indices.push_back(zero);
    }
    return false;
  }
  std::reverse(indices.begin(), indices.end());
  return true;
}

// If the memref.reinterpret_cast has multiple users (i.e the original
// cir.ptr_stride op has multiple users), only erase the operation after the
// last load or store has been generated.
static void eraseIfSafe(mlir::Value oldAddr, mlir::Value newAddr,
                        SmallVector<mlir::Operation *> &eraseList,
                        mlir::ConversionPatternRewriter &rewriter) {

  unsigned oldUsedNum =
      std::distance(oldAddr.getUses().begin(), oldAddr.getUses().end());
  unsigned newUsedNum = 0;
  // Count the uses of the newAddr (the result of the original base alloca) in
  // load/store ops using an forwarded offset from the current
  // memref.reinterpret_cast op
  for (auto *user : newAddr.getUsers()) {
    auto matchesRCOffset = [&](mlir::Value idx) -> bool {
      if (auto reinterpretOp = mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(
              eraseList.back())) {
        auto mixed = reinterpretOp.getMixedOffsets();
        if (mixed.empty())
          return false;
        if (auto dynVal =
                llvm::dyn_cast_if_present<mlir::Value>(mixed[0]))
          return idx == dynVal;
        // Static offset: check that idx is a matching constant.
        auto intAttr = mlir::cast<mlir::IntegerAttr>(
            mixed[0].get<mlir::Attribute>());
        if (auto cst = idx.getDefiningOp<mlir::arith::ConstantIndexOp>())
          return cst.value() == intAttr.getInt();
        return false;
      }
      if (mlir::dyn_cast<mlir::memref::CastOp>(eraseList.back()))
        return true;
      return false;
    };

    if (auto loadOpUser = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(*user)) {
      if (!loadOpUser.getIndices().empty())
        if (matchesRCOffset(loadOpUser.getIndices()[0]))
          ++newUsedNum;
    } else if (auto storeOpUser =
                   mlir::dyn_cast_or_null<mlir::memref::StoreOp>(*user)) {
      if (!storeOpUser.getIndices().empty())
        if (matchesRCOffset(storeOpUser.getIndices()[0]))
          ++newUsedNum;
    }
  }
  // If all load/store ops are using forwarded offsets from the current
  // memref.(reinterpret_)cast ops, erase them
  if (oldUsedNum == newUsedNum) {
    for (auto op : eraseList)
      rewriter.eraseOp(op);
  }
}

static mlir::LogicalResult
prepareReinterpretMetadata(mlir::MemRefType type,
                           mlir::ConversionPatternRewriter &rewriter,
                           llvm::SmallVectorImpl<mlir::OpFoldResult> &sizes,
                           llvm::SmallVectorImpl<mlir::OpFoldResult> &strides,
                           mlir::Operation *anchorOp) {
  sizes.clear();
  strides.clear();

  for (int64_t dim : type.getShape()) {
    sizes.push_back(rewriter.getIndexAttr(dim));
  }

  llvm::SmallVector<int64_t, 4> strideValues;
  int64_t layoutOffset = 0;
  if (mlir::failed(type.getStridesAndOffset(strideValues, layoutOffset))) {
    anchorOp->emitError("expected strided memref layout");
    return mlir::failure();
  }

  for (int64_t stride : strideValues) {
    if (mlir::ShapedType::isDynamic(stride)) {
      anchorOp->emitError("dynamic memref strides are not supported yet");
      return mlir::failure();
    }
    strides.push_back(rewriter.getIndexAttr(stride));
  }

  return mlir::success();
}

class CIRLoadOpLowering : public mlir::OpConversionPattern<cir::LoadOp> {
public:
  using OpConversionPattern<cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // If the address operand is an opaque LLVM pointer (e.g. loading from
    // `llvm.mlir.addressof @stderr`), we cannot use memref.load; emit an
    // LLVM-dialect load instead. The result type is whatever the type
    // converter produced for the load's CIR result (typically !llvm.ptr
    // again for `FILE **` → `FILE *`).
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(adaptor.getAddr().getType())) {
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());
      if (!resultType)
        return mlir::failure();
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, resultType,
                                                      adaptor.getAddr());
      return mlir::success();
    }

    mlir::Value base;
    SmallVector<mlir::Value> indices;
    SmallVector<mlir::Operation *> eraseList;
    mlir::memref::LoadOp newLoad;
    bool eraseIntermediateOp = findBaseAndIndices(adaptor.getAddr(), base,
                                                  indices, eraseList, rewriter);
    newLoad = mlir::memref::LoadOp::create(rewriter, op.getLoc(), base, indices,
                                           op.getIsNontemporal());
    if (eraseIntermediateOp)
      eraseIfSafe(op.getAddr(), base, eraseList, rewriter);

    // Convert adapted result to its original type if needed.
    mlir::Value result = emitFromMemory(rewriter, op, newLoad.getResult());
    rewriter.replaceOp(op, result);
    return mlir::LogicalResult::success();
  }
};

class CIRStoreOpLowering : public mlir::OpConversionPattern<cir::StoreOp> {
public:
  using OpConversionPattern<cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Symmetric to CIRLoadOpLowering: if storing through an opaque LLVM
    // pointer, emit llvm.store rather than memref.store.
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(adaptor.getAddr().getType())) {
      mlir::Value value = emitToMemory(rewriter, op, adaptor.getValue());
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, value,
                                                       adaptor.getAddr());
      return mlir::success();
    }

    mlir::Value base;
    SmallVector<mlir::Value> indices;
    SmallVector<mlir::Operation *> eraseList;

    // Convert adapted value to its memory type if needed.
    mlir::Value value = emitToMemory(rewriter, op, adaptor.getValue());
    bool eraseIntermediateOp = findBaseAndIndices(adaptor.getAddr(), base,
                                                  indices, eraseList, rewriter);
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, value, base, indices,
                                                       op.getIsNontemporal());
    if (eraseIntermediateOp)
      eraseIfSafe(op.getAddr(), base, eraseList, rewriter);

    return mlir::LogicalResult::success();
  }
};

/// Converts CIR unary math ops (e.g., cir::SinOp) to their MLIR equivalents
/// (e.g., math::SinOp) using a generic template to avoid redundant boilerplate
/// matchAndRewrite definitions.

template <typename CIROp, typename MLIROp>
class CIRUnaryMathOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CIROp op,
                  typename mlir::OpConversionPattern<CIROp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MLIROp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

using CIRASinOpLowering =
    CIRUnaryMathOpLowering<cir::ASinOp, mlir::math::AsinOp>;
using CIRSinOpLowering = CIRUnaryMathOpLowering<cir::SinOp, mlir::math::SinOp>;
using CIRExp2OpLowering =
    CIRUnaryMathOpLowering<cir::Exp2Op, mlir::math::Exp2Op>;
using CIRExpOpLowering = CIRUnaryMathOpLowering<cir::ExpOp, mlir::math::ExpOp>;
using CIRRoundOpLowering =
    CIRUnaryMathOpLowering<cir::RoundOp, mlir::math::RoundOp>;
using CIRLog2OpLowering =
    CIRUnaryMathOpLowering<cir::Log2Op, mlir::math::Log2Op>;
using CIRLogOpLowering = CIRUnaryMathOpLowering<cir::LogOp, mlir::math::LogOp>;
using CIRLog10OpLowering =
    CIRUnaryMathOpLowering<cir::Log10Op, mlir::math::Log10Op>;
using CIRCeilOpLowering =
    CIRUnaryMathOpLowering<cir::CeilOp, mlir::math::CeilOp>;
using CIRFloorOpLowering =
    CIRUnaryMathOpLowering<cir::FloorOp, mlir::math::FloorOp>;
using CIRAbsOpLowering = CIRUnaryMathOpLowering<cir::AbsOp, mlir::math::AbsIOp>;
using CIRFAbsOpLowering =
    CIRUnaryMathOpLowering<cir::FAbsOp, mlir::math::AbsFOp>;
using CIRSqrtOpLowering =
    CIRUnaryMathOpLowering<cir::SqrtOp, mlir::math::SqrtOp>;
using CIRCosOpLowering = CIRUnaryMathOpLowering<cir::CosOp, mlir::math::CosOp>;
using CIRATanOpLowering =
    CIRUnaryMathOpLowering<cir::ATanOp, mlir::math::AtanOp>;
using CIRACosOpLowering =
    CIRUnaryMathOpLowering<cir::ACosOp, mlir::math::AcosOp>;
using CIRTanOpLowering = CIRUnaryMathOpLowering<cir::TanOp, mlir::math::TanOp>;

class CIRShiftOpLowering : public mlir::OpConversionPattern<cir::ShiftOp> {
public:
  using mlir::OpConversionPattern<cir::ShiftOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::ShiftOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cirAmtTy = mlir::dyn_cast<cir::IntType>(op.getAmount().getType());
    auto cirValTy = mlir::dyn_cast<cir::IntType>(op.getValue().getType());

    // Operands could also be vector type
    auto cirAmtVTy = mlir::dyn_cast<cir::VectorType>(op.getAmount().getType());
    auto cirValVTy = mlir::dyn_cast<cir::VectorType>(op.getValue().getType());
    auto targetTy = getTypeConverter()->convertType(op.getType());
    mlir::Value amt = adaptor.getAmount();
    mlir::Value val = adaptor.getValue();

    assert(((cirValTy && cirAmtTy) || (cirAmtVTy && cirValVTy)) &&
           "shift input type must be integer or vector type, otherwise NYI");

    assert((cirValTy == op.getType() || cirValVTy == op.getType()) &&
           "inconsistent operands' types NYI");

    // Ensure shift amount is the same type as the value. Some undefined
    // behavior might occur in the casts below as per [C99 6.5.7.3].
    // Vector type shift amount needs no cast as type consistency is expected to
    // already be enforced at CIRGen.
    if (cirAmtTy)
      amt = createIntCast(rewriter, amt, targetTy, cirAmtTy.isSigned());

    // Lower to the proper arithmetic shift operation.
    if (op.getIsShiftleft())
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(op, targetTy, val, amt);
    else {
      bool isUnSigned =
          cirValTy ? !cirValTy.isSigned()
                   : !mlir::cast<cir::IntType>(cirValVTy.getElementType())
                          .isSigned();
      if (isUnSigned)
        rewriter.replaceOpWithNewOp<mlir::arith::ShRUIOp>(op, targetTy, val,
                                                          amt);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ShRSIOp>(op, targetTy, val,
                                                          amt);
    }

    return mlir::success();
  }
};

template <typename CIROp, typename MLIROp>
class CIRCountZerosBitOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CIROp op,
                  typename mlir::OpConversionPattern<CIROp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MLIROp>(op, adaptor.getInput());
    return mlir::LogicalResult::success();
  }
};

using CIRBitClzOpLowering =
    CIRCountZerosBitOpLowering<cir::BitClzOp, mlir::math::CountLeadingZerosOp>;
using CIRBitCtzOpLowering =
    CIRCountZerosBitOpLowering<cir::BitCtzOp, mlir::math::CountTrailingZerosOp>;

class CIRBitClrsbOpLowering
    : public mlir::OpConversionPattern<cir::BitClrsbOp> {
public:
  using OpConversionPattern<cir::BitClrsbOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitClrsbOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto zero = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isNeg = mlir::arith::CmpIOp::create(
        rewriter, op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::slt),
        adaptor.getInput(), zero);

    auto negOne = getConst(rewriter, op.getLoc(), inputTy, -1);
    auto flipped = mlir::arith::XOrIOp::create(rewriter, op.getLoc(),
                                               adaptor.getInput(), negOne);

    auto select = mlir::arith::SelectOp::create(rewriter, op.getLoc(), isNeg,
                                                flipped, adaptor.getInput());

    auto clz =
        mlir::math::CountLeadingZerosOp::create(rewriter, op->getLoc(), select);

    auto one = getConst(rewriter, op.getLoc(), inputTy, 1);
    auto res = mlir::arith::SubIOp::create(rewriter, op.getLoc(), clz, one);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitFfsOpLowering : public mlir::OpConversionPattern<cir::BitFfsOp> {
public:
  using OpConversionPattern<cir::BitFfsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitFfsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto ctz = mlir::math::CountTrailingZerosOp::create(rewriter, op.getLoc(),
                                                        adaptor.getInput());

    auto one = getConst(rewriter, op.getLoc(), inputTy, 1);
    auto ctzAddOne =
        mlir::arith::AddIOp::create(rewriter, op.getLoc(), ctz, one);

    auto zero = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isZero = mlir::arith::CmpIOp::create(
        rewriter, op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::eq),
        adaptor.getInput(), zero);

    auto res = mlir::arith::SelectOp::create(rewriter, op.getLoc(), isZero,
                                             zero, ctzAddOne);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitPopcountOpLowering
    : public mlir::OpConversionPattern<cir::BitPopcountOp> {
public:
  using mlir::OpConversionPattern<cir::BitPopcountOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitPopcountOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::CtPopOp>(op, adaptor.getInput());
    return mlir::LogicalResult::success();
  }
};

class CIRBitParityOpLowering
    : public mlir::OpConversionPattern<cir::BitParityOp> {
public:
  using OpConversionPattern<cir::BitParityOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitParityOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto count =
        mlir::math::CtPopOp::create(rewriter, op.getLoc(), adaptor.getInput());
    auto countMod2 = mlir::arith::AndIOp::create(
        rewriter, op.getLoc(), count,
        getConst(rewriter, op.getLoc(), count.getType(), 1));
    rewriter.replaceOp(op, countMod2);
    return mlir::LogicalResult::success();
  }
};

class CIRConstantOpLowering
    : public mlir::OpConversionPattern<cir::ConstantOp> {
public:
  using OpConversionPattern<cir::ConstantOp>::OpConversionPattern;

private:
  // This code is in a separate function rather than part of matchAndRewrite
  // because it is recursive.  There is currently only one level of recursion;
  // when lowing a vector attribute the attributes for the elements also need
  // to be lowered.
  mlir::TypedAttr
  lowerCirAttrToMlirAttr(mlir::Attribute cirAttr,
                         mlir::ConversionPatternRewriter &rewriter) const {
    assert(mlir::isa<mlir::TypedAttr>(cirAttr) &&
           "Can't lower a non-typed attribute");
    auto mlirType = getTypeConverter()->convertType(
        mlir::cast<mlir::TypedAttr>(cirAttr).getType());
    if (auto vecAttr = mlir::dyn_cast<cir::ConstVectorAttr>(cirAttr)) {
      assert(mlir::isa<mlir::VectorType>(mlirType) &&
             "MLIR type for CIR vector attribute is not mlir::VectorType");
      assert(mlir::isa<mlir::ShapedType>(mlirType) &&
             "mlir::VectorType is not a mlir::ShapedType ??");
      SmallVector<mlir::Attribute> mlirValues;
      for (auto elementAttr : vecAttr.getElts()) {
        mlirValues.push_back(
            this->lowerCirAttrToMlirAttr(elementAttr, rewriter));
      }
      return mlir::DenseElementsAttr::get(
          mlir::cast<mlir::ShapedType>(mlirType), mlirValues);
    } else if (auto zeroAttr = mlir::dyn_cast<cir::ZeroAttr>(cirAttr)) {
      (void)zeroAttr;
      return rewriter.getZeroAttr(mlirType);
    } else if (auto complexAttr = mlir::dyn_cast<cir::ComplexAttr>(cirAttr)) {
      auto vecType = mlir::dyn_cast<mlir::VectorType>(mlirType);
      assert(vecType && "complex attribute lowered type should be a vector");
      SmallVector<mlir::Attribute, 2> elements{
          this->lowerCirAttrToMlirAttr(complexAttr.getReal(), rewriter),
          this->lowerCirAttrToMlirAttr(complexAttr.getImag(), rewriter)};
      return mlir::DenseElementsAttr::get(vecType, elements);
    } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, boolAttr.getValue());
    } else if (auto floatAttr = mlir::dyn_cast<cir::FPAttr>(cirAttr)) {
      return rewriter.getFloatAttr(mlirType, floatAttr.getValue());
    } else if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, intAttr.getValue());
    } else {
      llvm_unreachable("NYI: unsupported attribute kind lowering to MLIR");
      return {};
    }
  }

public:
  mlir::LogicalResult
  matchAndRewrite(cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, getTypeConverter()->convertType(op.getType()),
        this->lowerCirAttrToMlirAttr(op.getValue(), rewriter));
    return mlir::LogicalResult::success();
  }
};

class CIRFuncOpLowering : public mlir::OpConversionPattern<cir::FuncOp> {
public:
  using OpConversionPattern<cir::FuncOp>::OpConversionPattern;

  void lowerFuncOpenCLKernelMetadataToMLIR(
      cir::ExtraFuncAttributesAttr extraAttr,
      SmallVectorImpl<mlir::NamedAttribute> &result) const {
    if (extraAttr.getElements().get(cir::OpenCLKernelAttr::getMnemonic()))
      result.push_back(
          mlir::NamedAttribute(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                               mlir::UnitAttr::get(getContext())));
  }

  void lowerFuncAttributesToMLIR(
      cir::FuncOp func, SmallVectorImpl<mlir::NamedAttribute> &result) const {
    if (auto symVisibilityAttr = func.getSymVisibilityAttr())
      result.push_back(
          mlir::NamedAttribute("sym_visibility", symVisibilityAttr));

    if (auto extraAttr = func.getExtraAttrs())
      lowerFuncOpenCLKernelMetadataToMLIR(extraAttr, result);
  }

  mlir::LogicalResult
  matchAndRewrite(cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto fnType = op.getFunctionType();

    if (fnType.isVarArg()) {
      auto *context = rewriter.getContext();
      auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);

      // Build LLVM types for the fixed parameters.
      SmallVector<mlir::Type> llvmParams;
      for (auto paramType : fnType.getInputs()) {
        auto converted = typeConverter->convertType(paramType);
        if (!converted)
          return mlir::failure();
        llvmParams.push_back(mlir::LLVM::isCompatibleType(converted)
                                 ? converted
                                 : llvmPtrTy);
      }

      // Build LLVM return type.
      mlir::Type retType;
      auto cirRetType = fnType.getReturnType();
      if (!cirRetType || isa<cir::VoidType>(cirRetType)) {
        retType = mlir::LLVM::LLVMVoidType::get(context);
      } else {
        auto converted = typeConverter->convertType(cirRetType);
        retType = (converted && mlir::LLVM::isCompatibleType(converted))
                      ? converted
                      : mlir::LLVM::LLVMVoidType::get(context);
      }

      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(retType, llvmParams,
                                                           /*isVarArg=*/true);

      if (op.isDeclaration()) {
        // Extern variadic declaration: emit llvm.func with no body.
        rewriter.replaceOpWithNewOp<mlir::LLVM::LLVMFuncOp>(op, op.getName(),
                                                             llvmFnType);
      } else {
        // Variadic function definition: emit llvm.func with the body inlined.
        // The body's CIR ops are lowered by existing patterns; cir.return is
        // handled by CIRReturnLowering (which checks for llvm.func parent).
        mlir::TypeConverter::SignatureConversion sigConv(fnType.getNumInputs());
        for (const auto &[i, llvmTy] : llvm::enumerate(llvmParams))
          sigConv.addInputs(i, llvmTy);

        auto fn = mlir::LLVM::LLVMFuncOp::create(rewriter, op.getLoc(),
                                                  op.getName(), llvmFnType);
        if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                               &sigConv)))
          return mlir::failure();
        rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
        rewriter.eraseOp(op);
      }
    } else {
      mlir::TypeConverter::SignatureConversion signatureConversion(
          fnType.getNumInputs());

      for (const auto &argType : enumerate(fnType.getInputs())) {
        auto convertedType = typeConverter->convertType(argType.value());
        if (!convertedType)
          return mlir::failure();
        signatureConversion.addInputs(argType.index(), convertedType);
      }

      SmallVector<mlir::NamedAttribute, 4> passThroughAttrs;
      lowerFuncAttributesToMLIR(op, passThroughAttrs);

      mlir::Type resultType =
          getTypeConverter()->convertType(fnType.getReturnType());
      auto fn = mlir::func::FuncOp::create(
          rewriter, op.getLoc(), op.getName(),
          rewriter.getFunctionType(signatureConversion.getConvertedTypes(),
                                   resultType ? mlir::TypeRange(resultType)
                                              : mlir::TypeRange()),
          passThroughAttrs);

      if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                             &signatureConversion)))
        return mlir::failure();
      rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());

      rewriter.eraseOp(op);
    }
    return mlir::LogicalResult::success();
  }
};

class CIRUnaryOpLowering : public mlir::OpConversionPattern<cir::UnaryOp> {
public:
  using OpConversionPattern<cir::UnaryOp>::OpConversionPattern;

  template <typename OpFloat, typename OpInt, bool rev>
  mlir::Operation *
  replaceImmediateOp(cir::UnaryOp op, mlir::Type type, mlir::Value input,
                     int64_t n,
                     mlir::ConversionPatternRewriter &rewriter) const {
    if (type.isFloat()) {
      auto imm = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(),
          mlir::FloatAttr::get(type, static_cast<double>(n)));
      if constexpr (rev)
        return rewriter.replaceOpWithNewOp<OpFloat>(op, type, imm, input);
      else
        return rewriter.replaceOpWithNewOp<OpFloat>(op, type, input, imm);
    }
    if (type.isInteger()) {
      auto imm = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(), mlir::IntegerAttr::get(type, n));
      if constexpr (rev)
        return rewriter.replaceOpWithNewOp<OpInt>(op, type, imm, input);
      else
        return rewriter.replaceOpWithNewOp<OpInt>(op, type, input, imm);
    }
    op->emitError("Unsupported type: ") << type << " at " << op->getLoc();
    llvm_unreachable("CIRUnaryOpLowering met unsupported type");
    return nullptr;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::UnaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto type = getTypeConverter()->convertType(op.getType());

    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      replaceImmediateOp<mlir::arith::AddFOp, mlir::arith::AddIOp, false>(
          op, type, input, 1, rewriter);
      break;
    }
    case cir::UnaryOpKind::Dec: {
      replaceImmediateOp<mlir::arith::AddFOp, mlir::arith::AddIOp, false>(
          op, type, input, -1, rewriter);
      break;
    }
    case cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, op.getInput());
      break;
    }
    case cir::UnaryOpKind::Minus: {
      replaceImmediateOp<mlir::arith::SubFOp, mlir::arith::SubIOp, true>(
          op, type, input, 0, rewriter);
      break;
    }
    case cir::UnaryOpKind::Not: {
      auto o = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(), mlir::IntegerAttr::get(type, -1));
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, type, o, input);
      break;
    }
    }
    return mlir::LogicalResult::success();
  }
};

class CIRBinOpLowering : public mlir::OpConversionPattern<cir::BinOp> {
public:
  using OpConversionPattern<cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert((adaptor.getLhs().getType() == adaptor.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
    assert((mlir::isa<mlir::IntegerType>(mlirType) ||
            mlir::isa<mlir::FloatType>(mlirType) ||
            mlir::isa<mlir::VectorType>(mlirType)) &&
           "operand type not supported yet");

    auto type = op.getLhs().getType();
    if (auto vecType = mlir::dyn_cast<cir::VectorType>(type)) {
      type = vecType.getElementType();
    }

    switch (op.getKind()) {
    case cir::BinOpKind::Add:
      if (mlir::isa<cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Sub:
      if (mlir::isa<cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Mul:
      if (mlir::isa<cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Div:
      if (auto ty = mlir::dyn_cast<cir::IntType>(type)) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Rem:
      if (auto ty = mlir::dyn_cast<cir::IntType>(type)) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::RemSIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Max:
      llvm_unreachable("BinOpKind::Max lowering through MLIR NYI");
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<cir::CmpOp> {
public:
  using OpConversionPattern<cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getLhs().getType();

    if (auto ty = mlir::dyn_cast<cir::IntType>(type)) {
      auto kind = convertCmpKindToCmpIPredicate(op.getKind(), ty.isSigned());
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
          op, kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ty = mlir::dyn_cast<cir::FPTypeInterface>(type)) {
      auto kind = convertCmpKindToCmpFPredicate(op.getKind());
      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
          op, kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ty = mlir::dyn_cast<cir::PointerType>(type)) {
      llvm_unreachable("pointer comparison not supported yet");
    } else {
      return op.emitError() << "unsupported type for CmpOp: " << type;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpConversionPattern<cir::BrOp> {
public:
  using mlir::OpConversionPattern<cir::BrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                    adaptor.getDestOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRScopeOpLowering : public mlir::OpConversionPattern<cir::ScopeOp> {
public:
  using mlir::OpConversionPattern<cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ScopeOp scopeOp, [[maybe_unused]] OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Empty scope: just remove it.
    // TODO: Remove this logic once CIR uses MLIR infrastructure to remove
    // trivially dead operations
    if (scopeOp.isEmpty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Check if the scope is empty (no operations)
    auto &scopeRegion = scopeOp.getScopeRegion();
    if (scopeRegion.empty() ||
        (scopeRegion.front().empty() ||
         (scopeRegion.front().getOperations().size() == 1 &&
          isa<cir::YieldOp>(scopeRegion.front().front())))) {
      // Drop empty scopes
      rewriter.eraseOp(scopeOp);
      return mlir::LogicalResult::success();
    }

    // For scopes without results, use memref.alloca_scope
    if (scopeOp.getNumResults() == 0) {
      auto allocaScope = mlir::memref::AllocaScopeOp::create(
          rewriter, scopeOp.getLoc(), mlir::TypeRange{});
      rewriter.inlineRegionBefore(scopeOp.getScopeRegion(),
                                  allocaScope.getBodyRegion(),
                                  allocaScope.getBodyRegion().end());
      rewriter.eraseOp(scopeOp);
    } else {
      // For scopes with results, use scf.execute_region
      SmallVector<mlir::Type> types;
      if (mlir::failed(getTypeConverter()->convertTypes(
              scopeOp->getResultTypes(), types)))
        return mlir::failure();
      auto exec =
          mlir::scf::ExecuteRegionOp::create(rewriter, scopeOp.getLoc(), types);
      rewriter.inlineRegionBefore(scopeOp.getScopeRegion(), exec.getRegion(),
                                  exec.getRegion().end());
      rewriter.replaceOp(scopeOp, exec.getResults());
    }
    return mlir::LogicalResult::success();
  }
};

struct CIRBrCondOpLowering : public mlir::OpConversionPattern<cir::BrCondOp> {
  using mlir::OpConversionPattern<cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        brOp, adaptor.getCond(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRTernaryOpLowering : public mlir::OpConversionPattern<cir::TernaryOp> {
public:
  using OpConversionPattern<cir::TernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return mlir::failure();

    auto ifOp = mlir::scf::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                        adaptor.getCond(), true);
    auto *thenBlock = &ifOp.getThenRegion().front();
    auto *elseBlock = &ifOp.getElseRegion().front();
    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), thenBlock,
                               thenBlock->end());
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), elseBlock,
                               elseBlock->end());

    rewriter.replaceOp(op, ifOp);
    return mlir::success();
  }
};

class CIRYieldOpLowering : public mlir::OpConversionPattern<cir::YieldOp> {
public:
  using OpConversionPattern<cir::YieldOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp,
              mlir::scf::ExecuteRegionOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Case<mlir::memref::AllocaScopeOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

class CIRIfOpLowering : public mlir::OpConversionPattern<cir::IfOp> {
public:
  using mlir::OpConversionPattern<cir::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::IfOp ifop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newIfOp =
        mlir::scf::IfOp::create(rewriter, ifop->getLoc(),
                                ifop->getResultTypes(), adaptor.getCondition());
    auto *thenBlock = rewriter.createBlock(&newIfOp.getThenRegion());
    rewriter.inlineBlockBefore(&ifop.getThenRegion().front(), thenBlock,
                               thenBlock->end());
    if (!ifop.getElseRegion().empty()) {
      auto *elseBlock = rewriter.createBlock(&newIfOp.getElseRegion());
      rewriter.inlineBlockBefore(&ifop.getElseRegion().front(), elseBlock,
                                 elseBlock->end());
    }
    rewriter.replaceOp(ifop, newIfOp);
    return mlir::success();
  }
};

class CIRGlobalOpLowering : public mlir::OpConversionPattern<cir::GlobalOp> {
public:
  using OpConversionPattern<cir::GlobalOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::OpBuilder b(moduleOp.getContext());

    const auto CIRSymType = op.getSymType();
    auto convertedType = convertTypeForMemory(*getTypeConverter(), CIRSymType);
    if (!convertedType)
      return mlir::failure();

    // If the converted element type cannot be stored in a memref (e.g.
    // !llvm.ptr produced by the PointerType converter's fallback for
    // pointer-to-struct types like `extern FILE *stderr`), emit an
    // LLVM-dialect global instead. This mirrors DirectToLLVM's handling of
    // the same case. We only support the declaration (no initializer) path
    // here, which is what arises for external globals such as `stderr`;
    // globals with non-memref element types AND a non-trivial initializer
    // would need the full DirectToLLVM initializer-region machinery, which
    // is out of scope for this pass.
    if (!mlir::MemRefType::isValidElementType(convertedType)) {
      if (op.getInitialValue().has_value()) {
        op.emitOpError("cir-to-mlir: global with non-memref element type "
                       "and a non-trivial initializer is not yet supported");
        return mlir::failure();
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, convertedType,
          /*isConstant=*/op.getConstant(),
          convertCIRLinkageToLLVMForMLIR(op.getLinkage()),
          op.getSymName(),
          /*value=*/mlir::Attribute{},
          /*alignment=*/op.getAlignment().value_or(0),
          /*addrSpace=*/0,
          /*dsoLocal=*/op.getDsoLocal(),
          /*threadLocal=*/(bool)op.getTlsModelAttr(),
          /*comdat=*/mlir::SymbolRefAttr{});
      return mlir::success();
    }

    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(convertedType);
    if (!memrefType) {
      auto maybeAddrSpace = getTypeConverter()->convertTypeAttribute(
          CIRSymType, op.getAddrSpaceAttr());
      mlir::Attribute addrSpace = maybeAddrSpace.value_or(mlir::Attribute());
      memrefType = mlir::MemRefType::get(
          {1}, convertedType, mlir::MemRefLayoutAttrInterface(), addrSpace);
    }
    // Add an optional alignment to the global memref.
    mlir::IntegerAttr memrefAlignment =
        op.getAlignment()
            ? mlir::IntegerAttr::get(b.getI64Type(), op.getAlignment().value())
            : mlir::IntegerAttr();
    // Add an optional initial value to the global memref.
    mlir::Attribute initialValue = mlir::Attribute();
    std::optional<mlir::Attribute> init = op.getInitialValue();
    if (init.has_value()) {
      if (auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(init.value())) {
        init = cir::direct::lowerConstArrayAttr(constArr, getTypeConverter());
        if (init.has_value())
          initialValue = init.value();
        else
          llvm_unreachable("GlobalOp lowering array with initial value fail");
      } else if (auto constComplex =
                     mlir::dyn_cast<cir::ComplexAttr>(init.value())) {
        if (auto lowered = cir::direct::lowerConstComplexAttr(
                constComplex, getTypeConverter());
            lowered.has_value())
          initialValue = lowered.value();
        else
          llvm_unreachable(
              "GlobalOp lowering complex with initial value failed");
      } else if (auto zeroAttr = mlir::dyn_cast<cir::ZeroAttr>(init.value())) {
        (void)zeroAttr;
        if (memrefType.getShape().size()) {
          auto elementType = memrefType.getElementType();
          auto rtt =
              mlir::RankedTensorType::get(memrefType.getShape(), elementType);
          if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
            initialValue = mlir::DenseIntElementsAttr::get(
                rtt, llvm::APInt(intTy.getWidth(), 0));
          } else if (mlir::isa<mlir::FloatType>(elementType)) {
            auto floatZero = mlir::FloatAttr::get(elementType, 0.0).getValue();
            initialValue = mlir::DenseFPElementsAttr::get(rtt, floatZero);
          } else
            initialValue = mlir::Attribute();
        } else {
          auto rtt = mlir::RankedTensorType::get({1}, convertedType);
          if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(convertedType)) {
            initialValue = mlir::DenseIntElementsAttr::get(
                rtt, llvm::APInt(intTy.getWidth(), 0));
          } else if (mlir::isa<mlir::FloatType>(convertedType)) {
            auto floatZero =
                mlir::FloatAttr::get(convertedType, 0.0).getValue();
            initialValue = mlir::DenseFPElementsAttr::get(rtt, floatZero);
          } else
            initialValue = mlir::Attribute();
        }
      } else if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({1}, convertedType);
        initialValue = mlir::DenseIntElementsAttr::get(rtt, intAttr.getValue());
      } else if (auto fltAttr = mlir::dyn_cast<cir::FPAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({1}, convertedType);
        initialValue = mlir::DenseFPElementsAttr::get(rtt, fltAttr.getValue());
      } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({1}, convertedType);
        initialValue =
            mlir::DenseIntElementsAttr::get(rtt, (char)boolAttr.getValue());
      } else
        llvm_unreachable(
            "GlobalOp lowering with initial value is not fully supported yet");
    }

    // Add symbol visibility
    std::string sym_visibility = op.isPrivate() ? "private" : "public";

    rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
        op, b.getStringAttr(op.getSymName()),
        /*sym_visibility=*/b.getStringAttr(sym_visibility),
        /*type=*/memrefType, initialValue,
        /*constant=*/op.getConstant(),
        /*alignment=*/memrefAlignment);

    return mlir::success();
  }
};

class CIRGetGlobalOpLowering
    : public mlir::OpConversionPattern<cir::GetGlobalOp> {
public:
  using OpConversionPattern<cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
    // CIRGen should mitigate this and not emit the get_global.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    // If the get_global result is an !llvm.ptr — either because the
    // referenced global has a non-memref element type (e.g.
    // `stderr : !cir.ptr<!rec__IO_FILE>`), or because the referenced symbol
    // is a function (e.g. `@polybench_timer_start` lowered to `llvm.func`) —
    // emit `llvm.mlir.addressof` rather than `memref.get_global`. This must
    // come before the `convertTypeForMemory` check on the pointee, which
    // returns null for non-storage types like `cir::FuncType`.
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(resultType)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, resultType,
                                                           op.getName());
      return mlir::success();
    }

    auto globalOpType =
        convertTypeForMemory(*getTypeConverter(), op.getType().getPointee());
    if (!globalOpType)
      return mlir::failure();

    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(globalOpType);
    if (!memrefType) {
      mlir::MemRefType resultTypeMemref =
          mlir::cast<mlir::MemRefType>(resultType);
      memrefType = mlir::MemRefType::get({1}, resultTypeMemref.getElementType(),
                                         resultTypeMemref.getLayout(),
                                         resultTypeMemref.getMemorySpace());
    }

    auto symbol = op.getName();
    auto getGlobalOp = mlir::memref::GetGlobalOp::create(rewriter, op.getLoc(),
                                                         memrefType, symbol);

    if (isa<cir::ArrayType>(op.getType().getPointee())) {
      rewriter.replaceOp(op, getGlobalOp);
    } else {
      // Cast from memref<1xmlirType> to memref<?xmlirType>. This is needed
      // since Typeconverter produces memref<?xmlirType> for non-array cir.ptrs.
      // The cast will be eliminated later in load/store-lowering.
      auto castOp = mlir::memref::CastOp::create(rewriter, op.getLoc(),
                                                 resultType, getGlobalOp);
      rewriter.replaceOp(op, castOp);
    }
    return mlir::success();
  }
};

class CIRComplexCreateOpLowering
    : public mlir::OpConversionPattern<cir::ComplexCreateOp> {
public:
  using OpConversionPattern<cir::ComplexCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto vecType = mlir::cast<mlir::VectorType>(
        getTypeConverter()->convertType(op.getType()));
    auto zeroAttr = rewriter.getZeroAttr(vecType);
    mlir::Value result =
        mlir::arith::ConstantOp::create(rewriter, loc, vecType, zeroAttr)
            .getResult();
    SmallVector<int64_t, 1> realIdx{0};
    SmallVector<int64_t, 1> imagIdx{1};
    result = mlir::vector::InsertOp::create(rewriter, loc, adaptor.getReal(),
                                            result, realIdx)
                 .getResult();
    result = mlir::vector::InsertOp::create(rewriter, loc, adaptor.getImag(),
                                            result, imagIdx)
                 .getResult();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class CIRComplexRealOpLowering
    : public mlir::OpConversionPattern<cir::ComplexRealOp> {
public:
  using OpConversionPattern<cir::ComplexRealOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexRealOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t, 1> idx{0};
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, adaptor.getOperand(), idx);
    return mlir::success();
  }
};

class CIRComplexImagOpLowering
    : public mlir::OpConversionPattern<cir::ComplexImagOp> {
public:
  using OpConversionPattern<cir::ComplexImagOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexImagOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t, 1> idx{1};
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, adaptor.getOperand(), idx);
    return mlir::success();
  }
};

class CIRVectorCreateLowering
    : public mlir::OpConversionPattern<cir::VecCreateOp> {
public:
  using OpConversionPattern<cir::VecCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType());
    assert(vecTy && "result type of cir.vec.create op is not VectorType");
    auto elementTy = typeConverter->convertType(vecTy.getElementType());
    auto loc = op.getLoc();
    auto zeroElement = rewriter.getZeroAttr(elementTy);
    mlir::Value vectorVal = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::DenseElementsAttr::get(
            mlir::VectorType::get(vecTy.getSize(), elementTy), zeroElement));
    assert(vecTy.getSize() == op.getElements().size() &&
           "cir.vec.create op count doesn't match vector type elements count");
    for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
      SmallVector<int64_t, 1> position{static_cast<int64_t>(i)};
      vectorVal = mlir::vector::InsertOp::create(rewriter, loc,
                                                 adaptor.getElements()[i],
                                                 vectorVal, position)
                      .getResult();
    }
    rewriter.replaceOp(op, vectorVal);
    return mlir::success();
  }
};

class CIRVectorInsertLowering
    : public mlir::OpConversionPattern<cir::VecInsertOp> {
public:
  using OpConversionPattern<cir::VecInsertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecInsertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value index = adaptor.getIndex();
    if (!mlir::isa<mlir::IndexType>(index.getType()))
      index = mlir::arith::IndexCastOp::create(rewriter, op.getLoc(),
                                               rewriter.getIndexType(), index);
    SmallVector<mlir::OpFoldResult, 1> position{index};
    auto newVec = mlir::vector::InsertOp::create(
        rewriter, op.getLoc(), adaptor.getValue(), adaptor.getVec(), position);
    rewriter.replaceOp(op, newVec.getResult());
    return mlir::success();
  }
};

class CIRVectorExtractLowering
    : public mlir::OpConversionPattern<cir::VecExtractOp> {
public:
  using OpConversionPattern<cir::VecExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecExtractOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value index = adaptor.getIndex();
    if (!mlir::isa<mlir::IndexType>(index.getType()))
      index = mlir::arith::IndexCastOp::create(rewriter, op.getLoc(),
                                               rewriter.getIndexType(), index);
    SmallVector<mlir::OpFoldResult, 1> position{index};
    auto extracted = mlir::vector::ExtractOp::create(
        rewriter, op.getLoc(), adaptor.getVec(), position);
    rewriter.replaceOp(op, extracted.getResult());
    return mlir::success();
  }
};

class CIRVectorCmpOpLowering : public mlir::OpConversionPattern<cir::VecCmpOp> {
public:
  using OpConversionPattern<cir::VecCmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(mlir::isa<cir::VectorType>(op.getType()) &&
           mlir::isa<cir::VectorType>(op.getLhs().getType()) &&
           mlir::isa<cir::VectorType>(op.getRhs().getType()) &&
           "Vector compare with non-vector type");
    auto elementType =
        mlir::cast<cir::VectorType>(op.getLhs().getType()).getElementType();
    mlir::Value bitResult;
    if (auto intType = mlir::dyn_cast<cir::IntType>(elementType)) {
      bitResult = mlir::arith::CmpIOp::create(
          rewriter, op.getLoc(),
          convertCmpKindToCmpIPredicate(op.getKind(), intType.isSigned()),
          adaptor.getLhs(), adaptor.getRhs());
    } else if (mlir::isa<cir::FPTypeInterface>(elementType)) {
      bitResult = mlir::arith::CmpFOp::create(
          rewriter, op.getLoc(), convertCmpKindToCmpFPredicate(op.getKind()),
          adaptor.getLhs(), adaptor.getRhs());
    } else {
      return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(
        op, typeConverter->convertType(op.getType()), bitResult);
    return mlir::success();
  }
};

class CIRCastOpLowering : public mlir::OpConversionPattern<cir::CastOp> {
public:
  using OpConversionPattern<cir::CastOp>::OpConversionPattern;

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isa<cir::VectorType>(op.getSrc().getType()))
      llvm_unreachable("CastOp lowering for vector type is not supported yet");
    auto src = adaptor.getSrc();
    auto dstType = op.getType();
    using CIR = cir::CastKind;
    switch (op.getKind()) {
    case CIR::array_to_ptrdecay: {
      auto newDstType = llvm::cast<mlir::MemRefType>(convertTy(dstType));
      auto srcMemref = llvm::cast<mlir::MemRefType>(src.getType());
      auto loc = op.getLoc();

      // Multi-dim sources (e.g. memref<100x100xi32> -> memref<100xi32>) are
      // rank-reducing decays. subview cannot reduce rank, so fall back to
      // reinterpret_cast for those cases.
      if (srcMemref.getRank() != 1) {
        llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
        if (mlir::failed(prepareReinterpretMetadata(newDstType, rewriter, sizes,
                                                    strides, op.getOperation())))
          return mlir::failure();
        rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
            op, newDstType, src, rewriter.getIndexAttr(0), sizes, strides);
        return mlir::success();
      }

      // 1-D source: emit subview so alias-analysis passes can see the
      // offset/size structure. The reinterpret_cast wrapper normalises the
      // result type; findBaseAndIndices peels it at load/store time.
      mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

      mlir::Value size;
      int64_t staticSize = srcMemref.getShape()[0];
      if (mlir::ShapedType::isDynamic(staticSize)) {
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
        size = mlir::memref::DimOp::create(rewriter, loc, src, dimIdx);
      } else {
        size = mlir::arith::ConstantIndexOp::create(rewriter, loc, staticSize);
      }

      llvm::SmallVector<mlir::OpFoldResult> svOffsets = {
          mlir::OpFoldResult(zero)};
      llvm::SmallVector<mlir::OpFoldResult> svSizes = {mlir::OpFoldResult(size)};
      llvm::SmallVector<mlir::OpFoldResult> svStrides = {
          mlir::OpFoldResult(one)};

      auto svType = mlir::memref::SubViewOp::inferResultType(srcMemref,
                                                             svOffsets, svSizes,
                                                             svStrides);
      auto sv = mlir::memref::SubViewOp::create(
          rewriter, loc, llvm::cast<mlir::MemRefType>(svType), src, svOffsets,
          svSizes, svStrides);

      // Build sizes: use the computed Value for dynamic dims, attr for static.
      llvm::SmallVector<mlir::OpFoldResult> rcSizes, rcStrides;
      for (int64_t dim : newDstType.getShape())
        rcSizes.push_back(mlir::ShapedType::isDynamic(dim)
                              ? mlir::OpFoldResult(size)
                              : rewriter.getIndexAttr(dim));
      llvm::SmallVector<int64_t, 4> strideVals;
      int64_t layoutOff;
      (void)newDstType.getStridesAndOffset(strideVals, layoutOff);
      for (int64_t s : strideVals)
        rcStrides.push_back(rewriter.getIndexAttr(s));

      rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
          op, newDstType, sv.getResult(), rewriter.getIndexAttr(0), rcSizes,
          rcStrides);
      return mlir::success();
    }
    case CIR::int_to_bool: {
      auto zero =
          cir::ConstantOp::create(rewriter, src.getLoc(), op.getSrc().getType(),
                                  cir::IntAttr::get(op.getSrc().getType(), 0));
      rewriter.replaceOpWithNewOp<cir::CmpOp>(
          op, cir::BoolType::get(getContext()), cir::CmpOpKind::ne, op.getSrc(),
          zero);
      return mlir::success();
    }
    case CIR::integral: {
      auto newDstType = convertTy(dstType);
      auto srcType = op.getSrc().getType();
      cir::IntType srcIntType = mlir::cast<cir::IntType>(srcType);
      auto newOp =
          createIntCast(rewriter, src, newDstType, srcIntType.isSigned());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::floating: {
      auto newDstType = convertTy(dstType);
      auto srcTy = op.getSrc().getType();
      auto dstTy = op.getType();

      if (!mlir::isa<cir::FPTypeInterface>(dstTy) ||
          !mlir::isa<cir::FPTypeInterface>(srcTy))
        return op.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

      auto getFloatWidth = [](mlir::Type ty) -> unsigned {
        return mlir::cast<cir::FPTypeInterface>(ty).getWidth();
      };

      if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
        rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_bool: {
      auto kind = mlir::arith::CmpFPredicate::UNE;

      // Check if float is not equal to zero.
      auto zeroFloat = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(), src.getType(),
          mlir::FloatAttr::get(src.getType(), 0.0));

      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(op, kind, src,
                                                       zeroFloat);
      return mlir::success();
    }
    case CIR::bool_to_int: {
      auto dstTy = mlir::cast<cir::IntType>(op.getType());
      auto newDstType = mlir::cast<mlir::IntegerType>(convertTy(dstTy));
      auto newOp = createIntCast(rewriter, src, newDstType);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::bool_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::int_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<cir::IntType>(op.getSrc().getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_int: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<cir::IntType>(op.getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::FPToUIOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::bitcast: {
      // Pointer bitcasts. Three shapes arise in practice once !llvm.ptr is
      // produced by the PointerType converter fallback:
      //   1. !llvm.ptr -> !llvm.ptr  (no-op: LLVM pointers are opaque).
      //   2. memref<...> -> !llvm.ptr  (e.g. `free((void*)array)`): extract
      //      the aligned base pointer via memrefToLLVMPtr.
      //   3. !llvm.ptr -> memref<...>: not expected for the current inputs;
      //      reject explicitly so we fail loudly rather than produce an
      //      unresolved materialization.
      auto newDstType = convertTy(dstType);
      auto srcType = src.getType();
      if (!newDstType)
        return mlir::failure();
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(newDstType) &&
          mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
        rewriter.replaceOp(op, src);
        return mlir::success();
      }
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(newDstType) &&
          mlir::isa<mlir::MemRefType>(srcType)) {
        mlir::Value ptrVal = memrefToLLVMPtr(rewriter, op.getLoc(), src);
        rewriter.replaceOp(op, ptrVal);
        return mlir::success();
      }
      if (mlir::isa<mlir::MemRefType>(newDstType) &&
          mlir::isa<mlir::MemRefType>(srcType)) {
        // Same-shape memref bitcasts (e.g. typed-ptr casts that survived
        // into CIR). memref.cast only permits compatible layouts; if types
        // already match, just forward.
        if (srcType == newDstType) {
          rewriter.replaceOp(op, src);
          return mlir::success();
        }
        rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, newDstType, src);
        return mlir::success();
      }
      // !llvm.ptr -> memref<...> : arises when a C function returns `void *`
      // (e.g. `polybench_alloc_data`) and the CIR bitcast narrows the result
      // to a typed pointer. We cannot faithfully rebuild a memref descriptor
      // without shape/stride info, so bridge with an unrealized conversion
      // cast. The cast is legal in the target, and the eventual LLVM
      // lowering materialises the underlying descriptor.
      if (mlir::isa<mlir::MemRefType>(newDstType) &&
          mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
        rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
            op, newDstType, src);
        return mlir::success();
      }
      return op.emitError() << "bitcast lowering from " << srcType << " to "
                            << newDstType << " not supported";
    }
    default:
      break;
    }
    return mlir::failure();
  }
};

class CIRGetElementOpLowering
    : public mlir::OpConversionPattern<cir::GetElementOp> {
  using mlir::OpConversionPattern<cir::GetElementOp>::OpConversionPattern;

  // Rewrite
  //        cir.get_element(%base[%index])
  // to
  //        memref.subview(%base[%index][1][1]) + memref.reinterpret_cast
  //
  // The subview carries structural offset/size information needed by alias
  // analysis passes. The reinterpret_cast wrapper normalises the type to the
  // expected memref<?xT> and is peeled off by findBaseAndIndices at load/store
  // time, leaving the subview alive in the IR.
  mlir::LogicalResult
  matchAndRewrite(cir::GetElementOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Cast the index to the index type, if needed.
    auto index = adaptor.getIndex();
    auto indexType = rewriter.getIndexType();
    auto loc = op.getLoc();
    if (index.getType() != indexType)
      index =
          mlir::arith::IndexCastOp::create(rewriter, loc, indexType, index);

    // Convert the destination type.
    auto dstType =
        cast<mlir::MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto baseMemref =
        llvm::cast<mlir::MemRefType>(adaptor.getBase().getType());

    mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    llvm::SmallVector<mlir::OpFoldResult> svOffsets = {
        mlir::OpFoldResult(index)};
    llvm::SmallVector<mlir::OpFoldResult> svSizes = {mlir::OpFoldResult(one)};
    llvm::SmallVector<mlir::OpFoldResult> svStrides = {mlir::OpFoldResult(one)};

    auto svType = mlir::memref::SubViewOp::inferResultType(baseMemref, svOffsets,
                                                           svSizes, svStrides);
    auto sv = mlir::memref::SubViewOp::create(
        rewriter, loc, llvm::cast<mlir::MemRefType>(svType), adaptor.getBase(),
        svOffsets, svSizes, svStrides);

    llvm::SmallVector<mlir::OpFoldResult> rcSizes, rcStrides;
    for (int64_t dim : dstType.getShape())
      rcSizes.push_back(mlir::ShapedType::isDynamic(dim)
                            ? mlir::OpFoldResult(one)
                            : rewriter.getIndexAttr(dim));
    llvm::SmallVector<int64_t, 4> strideVals;
    int64_t layoutOff;
    (void)dstType.getStridesAndOffset(strideVals, layoutOff);
    for (int64_t s : strideVals)
      rcStrides.push_back(rewriter.getIndexAttr(s));

    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, dstType, sv.getResult(), rewriter.getIndexAttr(0), rcSizes,
        rcStrides);

    return mlir::success();
  }
};

class CIRPtrStrideOpLowering
    : public mlir::OpConversionPattern<cir::PtrStrideOp> {
public:
  using mlir::OpConversionPattern<cir::PtrStrideOp>::OpConversionPattern;

  // Return true if all the PtrStrideOp users are load, store or cast
  // with array_to_ptrdecay kind and they are in the same block.
  inline bool isLoadStoreOrCastArrayToPtrProduer(cir::PtrStrideOp op) const {
    if (op.use_empty())
      return false;
    for (auto *user : op->getUsers()) {
      if (!op->isBeforeInBlock(user))
        return false;
      if (isa<cir::LoadOp, cir::StoreOp, cir::GetElementOp>(*user))
        continue;
      auto castOp = dyn_cast<cir::CastOp>(*user);
      if (castOp && (castOp.getKind() == cir::CastKind::array_to_ptrdecay))
        continue;
      return false;
    }
    return true;
  }

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  // Rewrite
  //        cir.ptr_stride(%base, %stride)
  // to
  //        memref.subview(%base[%stride][dim0-stride][1]) +
  //        memref.reinterpret_cast
  //
  // The subview carries structural offset/size information: its offset is the
  // same SSA value used as a size operand in a preceding array_to_ptrdecay
  // subview, enabling alias analysis passes to detect Form A disjointness.
  mlir::LogicalResult rewritePtrStrideToReinterpret(
      cir::PtrStrideOp op, mlir::Value base, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const {
    auto ptrType = op.getType();
    auto memrefType = llvm::cast<mlir::MemRefType>(convertTy(ptrType));
    auto stride = adaptor.getStride();
    auto indexType = rewriter.getIndexType();
    auto loc = op.getLoc();
    if (stride.getType() != indexType)
      stride =
          mlir::arith::IndexCastOp::create(rewriter, loc, indexType, stride);

    // Compute remaining size = dim(base, 0) - stride.
    auto baseMemref = llvm::cast<mlir::MemRefType>(base.getType());
    mlir::Value totalSize;
    int64_t staticSize = baseMemref.getShape()[0];
    if (mlir::ShapedType::isDynamic(staticSize)) {
      mlir::Value dimIdx =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      totalSize = mlir::memref::DimOp::create(rewriter, loc, base, dimIdx);
    } else {
      totalSize =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, staticSize);
    }
    mlir::Value remaining =
        mlir::arith::SubIOp::create(rewriter, loc, totalSize, stride);

    mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

    llvm::SmallVector<mlir::OpFoldResult> svOffsets = {
        mlir::OpFoldResult(stride)};
    llvm::SmallVector<mlir::OpFoldResult> svSizes = {
        mlir::OpFoldResult(remaining)};
    llvm::SmallVector<mlir::OpFoldResult> svStrides = {mlir::OpFoldResult(one)};

    auto svType = mlir::memref::SubViewOp::inferResultType(baseMemref, svOffsets,
                                                           svSizes, svStrides);
    auto sv = mlir::memref::SubViewOp::create(
        rewriter, loc, llvm::cast<mlir::MemRefType>(svType), base, svOffsets,
        svSizes, svStrides);

    llvm::SmallVector<mlir::OpFoldResult> rcSizes, rcStrides;
    for (int64_t dim : memrefType.getShape())
      rcSizes.push_back(mlir::ShapedType::isDynamic(dim)
                            ? mlir::OpFoldResult(remaining)
                            : rewriter.getIndexAttr(dim));
    llvm::SmallVector<int64_t, 4> strideVals;
    int64_t layoutOff;
    (void)memrefType.getStridesAndOffset(strideVals, layoutOff);
    for (int64_t s : strideVals)
      rcStrides.push_back(rewriter.getIndexAttr(s));

    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, memrefType, sv.getResult(), rewriter.getIndexAttr(0), rcSizes,
        rcStrides);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::PtrStrideOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isLoadStoreOrCastArrayToPtrProduer(op)) {
      if (!isa<cir::ArrayType>(op.getType().getPointee()))
        return rewritePtrStrideToReinterpret(op, adaptor.getBase(), adaptor,
                                             rewriter);
    }

    // Rank-preserving subview for pointer-to-array stride. The base memref may
    // be multi-dimensional (e.g. memref<?xNxT> for int (*p)[N]). Stride on
    // dimension 0; all inner dimensions are preserved at their full extent.
    auto base = adaptor.getBase();
    auto stride = adaptor.getStride();
    auto loc = op.getLoc();
    auto indexType = rewriter.getIndexType();
    auto baseMemref = llvm::cast<mlir::MemRefType>(base.getType());
    int64_t rank = baseMemref.getRank();

    if (stride.getType() != indexType)
      stride =
          mlir::arith::IndexCastOp::create(rewriter, loc, indexType, stride);

    mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    llvm::SmallVector<mlir::OpFoldResult> svOffsets, svSizes, svStrides;

    // Dimension 0: offset = stride, size = total_d0 - stride.
    mlir::Value totalD0;
    int64_t staticD0 = baseMemref.getShape()[0];
    if (mlir::ShapedType::isDynamic(staticD0)) {
      mlir::Value dimIdx =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      totalD0 = mlir::memref::DimOp::create(rewriter, loc, base, dimIdx);
    } else {
      totalD0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, staticD0);
    }
    mlir::Value remaining =
        mlir::arith::SubIOp::create(rewriter, loc, totalD0, stride);
    svOffsets.push_back(mlir::OpFoldResult(stride));
    svSizes.push_back(mlir::OpFoldResult(remaining));
    svStrides.push_back(mlir::OpFoldResult(one));

    // Higher dimensions: offset = 0, size = static extent, stride = 1.
    for (int64_t d = 1; d < rank; ++d) {
      svOffsets.push_back(mlir::OpFoldResult(zero));
      int64_t dimSize = baseMemref.getShape()[d];
      if (mlir::ShapedType::isDynamic(dimSize)) {
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, d);
        mlir::Value dimSizeVal =
            mlir::memref::DimOp::create(rewriter, loc, base, dimIdx);
        svSizes.push_back(mlir::OpFoldResult(dimSizeVal));
      } else {
        svSizes.push_back(rewriter.getIndexAttr(dimSize));
      }
      svStrides.push_back(mlir::OpFoldResult(one));
    }

    auto svType = mlir::memref::SubViewOp::inferResultType(baseMemref, svOffsets,
                                                           svSizes, svStrides);
    auto sv = mlir::memref::SubViewOp::create(
        rewriter, loc, llvm::cast<mlir::MemRefType>(svType), base, svOffsets,
        svSizes, svStrides);

    auto resultMemref =
        llvm::cast<mlir::MemRefType>(convertTy(op.getType()));
    llvm::SmallVector<mlir::OpFoldResult> rcSizes, rcStrides;
    // Dim 0 may be dynamic (use `remaining`); higher dims are static.
    for (auto [d, dim] : llvm::enumerate(resultMemref.getShape())) {
      if (mlir::ShapedType::isDynamic(dim))
        rcSizes.push_back(d == 0 ? mlir::OpFoldResult(remaining)
                                 : mlir::OpFoldResult(
                                       mlir::memref::DimOp::create(
                                           rewriter, loc, sv, (int64_t)d)));
      else
        rcSizes.push_back(rewriter.getIndexAttr(dim));
    }
    llvm::SmallVector<int64_t, 4> strideVals;
    int64_t layoutOff;
    (void)resultMemref.getStridesAndOffset(strideVals, layoutOff);
    for (int64_t s : strideVals)
      rcStrides.push_back(rewriter.getIndexAttr(s));
    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, resultMemref, sv.getResult(), rewriter.getIndexAttr(0), rcSizes,
        rcStrides);
    return mlir::success();
  }
};

class CIRSelectOpLowering : public mlir::OpConversionPattern<cir::SelectOp> {
public:
  using OpConversionPattern<cir::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};

class CIRUnreachableOpLowering
    : public mlir::OpConversionPattern<cir::UnreachableOp> {
public:
  using OpConversionPattern<cir::UnreachableOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::UnreachableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
    return mlir::success();
  }
};

class CIRTrapOpLowering : public mlir::OpConversionPattern<cir::TrapOp> {
public:
  using OpConversionPattern<cir::TrapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TrapOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    auto trapIntrinsicName = rewriter.getStringAttr("llvm.trap");
    mlir::LLVM::CallIntrinsicOp::create(rewriter, op.getLoc(),
                                        trapIntrinsicName,
                                        /*args=*/mlir::ValueRange());
    mlir::LLVM::UnreachableOp::create(rewriter, op.getLoc());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRCopyOpLowering : public mlir::OpConversionPattern<cir::CopyOp> {
public:
  using OpConversionPattern<cir::CopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::CopyOp>(op, adaptor.getSrc(),
                                                      adaptor.getDst());
    return mlir::success();
  }
};

class CIRVAStartOpLowering : public mlir::OpConversionPattern<cir::VAStartOp> {
public:
  using OpConversionPattern<cir::VAStartOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VAStartOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr = memrefToLLVMPtr(rewriter, op.getLoc(), adaptor.getArgList());
    rewriter.replaceOpWithNewOp<mlir::LLVM::VaStartOp>(op, ptr);
    return mlir::success();
  }
};

class CIRVAEndOpLowering : public mlir::OpConversionPattern<cir::VAEndOp> {
public:
  using OpConversionPattern<cir::VAEndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VAEndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr = memrefToLLVMPtr(rewriter, op.getLoc(), adaptor.getArgList());
    rewriter.replaceOpWithNewOp<mlir::LLVM::VaEndOp>(op, ptr);
    return mlir::success();
  }
};

class CIRVACopyOpLowering : public mlir::OpConversionPattern<cir::VACopyOp> {
public:
  using OpConversionPattern<cir::VACopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VACopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = memrefToLLVMPtr(rewriter, op.getLoc(), adaptor.getSrcList());
    auto dst = memrefToLLVMPtr(rewriter, op.getLoc(), adaptor.getDstList());
    rewriter.replaceOpWithNewOp<mlir::LLVM::VaCopyOp>(op, src, dst);
    return mlir::success();
  }
};

class CIRVAArgOpLowering : public mlir::OpConversionPattern<cir::VAArgOp> {
public:
  using OpConversionPattern<cir::VAArgOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VAArgOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return op.emitError("cir.va.arg lowering not yet supported "
                        "(requires ABI-specific va_arg expansion)");
  }
};

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering, CIRBrOpLowering>(patterns.getContext());

  patterns
      .add<CIRATanOpLowering, CIRCmpOpLowering, CIRCallOpLowering,
           CIRUnaryOpLowering, CIRBinOpLowering, CIRLoadOpLowering,
           CIRConstantOpLowering, CIRStoreOpLowering, CIRAllocaOpLowering,
           CIRFuncOpLowering, CIRBrCondOpLowering, CIRTernaryOpLowering,
           CIRYieldOpLowering, CIRCosOpLowering, CIRGlobalOpLowering,
           CIRGetGlobalOpLowering, CIRComplexCreateOpLowering,
           CIRComplexRealOpLowering, CIRComplexImagOpLowering,
           CIRCastOpLowering, CIRPtrStrideOpLowering, CIRSelectOpLowering,
           CIRGetElementOpLowering, CIRSqrtOpLowering, CIRCeilOpLowering,
           CIRExp2OpLowering, CIRExpOpLowering, CIRFAbsOpLowering,
           CIRAbsOpLowering, CIRFloorOpLowering, CIRLog10OpLowering,
           CIRLog2OpLowering, CIRLogOpLowering, CIRRoundOpLowering,
           CIRSinOpLowering, CIRTanOpLowering, CIRShiftOpLowering,
           CIRBitClzOpLowering, CIRBitCtzOpLowering, CIRBitPopcountOpLowering,
           CIRBitClrsbOpLowering, CIRBitFfsOpLowering, CIRBitParityOpLowering,
           CIRIfOpLowering, CIRScopeOpLowering, CIRVectorCreateLowering,
           CIRVectorInsertLowering, CIRVectorExtractLowering,
           CIRVectorCmpOpLowering, CIRACosOpLowering, CIRASinOpLowering,
           CIRUnreachableOpLowering, CIRTrapOpLowering, CIRCopyOpLowering,
           CIRVAStartOpLowering, CIRVAEndOpLowering, CIRVACopyOpLowering,
           CIRVAArgOpLowering>(converter, patterns.getContext());
}

static mlir::Attribute
convertCIRLangAddrSpaceToGPU(cir::LangAddressSpaceAttr addrSpace) {
  auto context = addrSpace.getContext();
  switch (addrSpace.getValue()) {
  case cir::LangAddressSpace::OffloadPrivate:
    return mlir::gpu::AddressSpaceAttr::get(context,
                                            mlir::gpu::AddressSpace::Private);
  case cir::LangAddressSpace::OffloadLocal:
    return mlir::gpu::AddressSpaceAttr::get(context,
                                            mlir::gpu::AddressSpace::Workgroup);
  case cir::LangAddressSpace::OffloadGlobal:
    return mlir::gpu::AddressSpaceAttr::get(context,
                                            mlir::gpu::AddressSpace::Global);
  case cir::LangAddressSpace::OffloadConstant:
  case cir::LangAddressSpace::OffloadGeneric:
  case cir::LangAddressSpace::Default:
    return mlir::Attribute();
  }
}

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    auto ty = convertTypeForMemory(converter, type.getPointee());
    if (!ty)
      // Fallback for unconvertible pointee types (structs, void*, etc.):
      // use an opaque LLVM pointer so calls with FILE* etc. can proceed.
      // TODO: this fallback leaks !llvm.ptr into contexts that cannot accept
      // it (e.g. memref-world globals). CIRGlobalOpLowering and
      // CIRGetGlobalOpLowering branch on this and emit llvm.mlir.global /
      // llvm.mlir.addressof instead; a cleaner fix would confine !llvm.ptr
      // production to variadic/call boundaries via type materializations.
      return mlir::LLVM::LLVMPointerType::get(type.getContext());
    if (isa<cir::ArrayType>(type.getPointee()))
      return ty;
    // If the converted pointee is itself not a valid memref element type
    // (e.g. !llvm.ptr produced recursively for pointer-to-struct-pointer
    // like `FILE **`), fall back to an opaque LLVM pointer rather than
    // asserting inside MemRefType::get.
    if (!mlir::MemRefType::isValidElementType(ty))
      return mlir::LLVM::LLVMPointerType::get(type.getContext());
    auto maybeAddrSpace =
        converter.convertTypeAttribute(type, type.getAddrSpace());
    mlir::Attribute addrSpace = maybeAddrSpace.value_or(mlir::Attribute());
    return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, ty,
                                 mlir::MemRefLayoutAttrInterface(), addrSpace);
  });
  converter.addConversion(
      [&](mlir::IntegerType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::FloatType type) -> mlir::Type { return type; });
  converter.addConversion([&](cir::VoidType type) -> mlir::Type { return {}; });
  converter.addConversion([&](cir::IntType type) -> mlir::Type {
    // arith dialect ops doesn't take signed integer -- drop cir sign here
    return mlir::IntegerType::get(
        type.getContext(), type.getWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  });
  converter.addConversion([&](cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::Float128Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::Float16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::BFloat16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::ArrayType type) -> mlir::Type {
    SmallVector<int64_t> shape;
    mlir::Type curType = type;
    while (auto arrayType = dyn_cast<cir::ArrayType>(curType)) {
      shape.push_back(arrayType.getSize());
      curType = arrayType.getElementType();
    }
    auto elementType = converter.convertType(curType);
    // FIXME: The element type might not be converted (e.g. struct)
    if (!elementType)
      return nullptr;
    return mlir::MemRefType::get(shape, elementType);
  });
  converter.addConversion([&](cir::VectorType type) -> mlir::Type {
    auto ty = converter.convertType(type.getElementType());
    return mlir::VectorType::get(type.getSize(), ty);
  });
  converter.addConversion([&](cir::ComplexType type) -> mlir::Type {
    auto elemTy = converter.convertType(type.getElementType());
    if (!elemTy)
      return nullptr;
    return mlir::VectorType::get(2, elemTy);
  });
  converter.addConversion(
      [&](cir::OpaqueType type) -> mlir::Type { llvm_unreachable("NYI"); });
  converter.addTypeAttributeConversion(
      [](mlir::Type, cir::TargetAddressSpaceAttr memorySpaceAttr) {
        auto targetMemorySpace = memorySpaceAttr.getValue();
        return mlir::IntegerAttr::get(
            mlir::IntegerType::get(memorySpaceAttr.getContext(), 64),
            targetMemorySpace);
      });
  converter.addTypeAttributeConversion(
      [](mlir::Type, cir::LangAddressSpaceAttr memorySpaceAttr) {
        return convertCIRLangAddrSpaceToGPU(memorySpaceAttr);
      });
  return converter;
}

void ConvertCIRToMLIRPass::runOnOperation() {
  mlir::ModuleOp theModule = getOperation();

  auto converter = prepareTypeConverter();

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRLoopToSCFConversionPatterns(patterns, converter);
  populateCIRToMLIRConversionPatterns(patterns, converter);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  // Unrealized conversion casts are used as explicit bridges between the
  // memref world and !llvm.ptr at C-ABI boundaries (e.g. void*-returning
  // allocators); mark them legal so partial conversion does not treat them
  // as unresolved materializations.
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target
      .addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                       mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                       mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                       mlir::ptr::PtrDialect, mlir::math::MathDialect,
                       mlir::vector::VectorDialect, mlir::LLVM::LLVMDialect>();
  auto *context = patterns.getContext();

  // We cannot mark cir dialect as illegal before conversion.
  // The conversion of WhileOp relies on partially preserving operations from
  // cir dialect, for example the `cir.continue`. If we marked cir as illegal
  // here, then MLIR would think any remaining `cir.continue` indicates a
  // failure, which is not what we want.

  patterns.add<CIRCastOpLowering, CIRIfOpLowering, CIRScopeOpLowering,
               CIRYieldOpLowering>(converter, context);

  if (mlir::failed(mlir::applyPartialConversion(theModule, target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

mlir::ModuleOp lowerFromCIRToMLIRToLLVMDialect(mlir::ModuleOp theModule,
                                               mlir::MLIRContext *mlirCtx) {
  llvm::TimeTraceScope scope("Lower from CIR to MLIR To LLVM Dialect");
  if (!mlirCtx) {
    mlirCtx = theModule.getContext();
  }

  mlir::PassManager pm(mlirCtx);

  pm.addPass(createConvertCIRToMLIRPass());
  pm.addPass(createConvertMLIRToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  return theModule;
}

std::unique_ptr<llvm::Module>
lowerFromCIRToMLIRToLLVMIR(mlir::ModuleOp theModule,
                           std::unique_ptr<mlir::MLIRContext> mlirCtx,
                           llvm::LLVMContext &llvmCtx) {
  llvm::TimeTraceScope scope("Lower from CIR to MLIR To LLVM");

  lowerFromCIRToMLIRToLLVMDialect(theModule, mlirCtx.get());

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  mlir::registerOpenMPDialectTranslation(*mlirCtx);

  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}

mlir::ModuleOp lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule,
                                            mlir::MLIRContext *mlirCtx) {
  auto llvmModule = lowerFromCIRToMLIR(theModule, mlirCtx);
  if (!llvmModule.getOperation())
    return {};

  mlir::PassManager pm(mlirCtx);

  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  if (mlir::failed(pm.run(llvmModule))) {
    llvmModule.emitError("The pass manager failed to lower the module");
    return {};
  }

  return llvmModule;
}

std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass() {
  return std::make_unique<ConvertCIRToMLIRPass>();
}

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp theModule,
                                  mlir::MLIRContext *mlirCtx) {
  llvm::TimeTraceScope scope("Lower CIR To MLIR");

  mlir::PassManager pm(mlirCtx);
  pm.addPass(createConvertCIRToMLIRPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to MLIR standard dialects!");
  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error(
        "Verification of the final MLIR in standard dialects failed!");

  return theModule;
}

} // namespace cir
