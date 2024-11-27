#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MODULOATTRIBUTEPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct ModuloAttributePass
    : public IREE::Stream::impl::ModuloAttributePassBase<ModuloAttributePass> {
  void runOnOperation() override {
    auto rootOp = getOperation();
    if (!rootOp) {
      return signalPassFailure();
    }

    OpBuilder builder(rootOp.getContext());

    auto attr = rootOp->getAttrOfType<mlir::IntegerAttr>("arith.number");
    if (!attr) {
      rootOp.emitError("Could not find arith.number attribute");
      return signalPassFailure();
    }

    // Get the current value and modulo it
    int32_t currentValue = attr.getInt();
    int32_t newValue = currentValue % 7;

    // Create new attribute with new value
    auto newAttr = builder.getI32IntegerAttr(newValue);

    // Replace the old attribute
    rootOp->setAttr("arith.number", newAttr);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
