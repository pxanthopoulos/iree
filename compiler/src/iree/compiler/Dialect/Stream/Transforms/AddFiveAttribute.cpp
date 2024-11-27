#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ADDFIVEATTRIBUTEPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct AddFiveAttributePass
    : public IREE::Stream::impl::AddFiveAttributePassBase<
          AddFiveAttributePass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    if (!moduleOp) {
      return signalPassFailure();
    }

    OpBuilder builder(moduleOp.getContext());

    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      if (llvm::isa<mlir::func::FuncOp, IREE::Util::InitializerOp,
                    IREE::Util::FuncOp>(callableOp)) {
        auto attr =
            callableOp->getAttrOfType<mlir::IntegerAttr>("arith.number");
        if (!attr) {
          callableOp.emitError("Could not find arith.number attribute");
          return signalPassFailure();
        }

        // Get the current value and increment it by the specified amount
        int32_t currentValue = attr.getInt();
        int32_t newValue = currentValue + 5;

        // Create new attribute with incremented value
        auto newAttr = builder.getI32IntegerAttr(newValue);

        // Replace the old attribute
        callableOp->setAttr("arith.number", newAttr);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
