#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ADDINITIALATTRIBUTEPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct AddInitialAttributePass
    : public IREE::Stream::impl::AddInitialAttributePassBase<
          AddInitialAttributePass> {
  void runOnOperation() override {
    auto rootOp = getOperation();
    if (!rootOp) {
      return signalPassFailure();
    }

    OpBuilder builder(rootOp.getContext());

    // Create a constant value
    auto value = builder.getI32IntegerAttr(36);

    // Add it as a top-level named attribute with arith prefix
    rootOp->setAttr("arith.number", value);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
