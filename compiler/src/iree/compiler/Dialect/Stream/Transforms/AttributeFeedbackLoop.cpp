#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-feedback-loop"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ATTRIBUTEFEEDBACKLOOPPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct AttributeFeedbackLoopPass
    : public IREE::Stream::impl::AttributeFeedbackLoopPassBase<
          AttributeFeedbackLoopPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    if (!moduleOp) {
      return signalPassFailure();
    }

    OpBuilder builder(moduleOp.getContext());
    auto initialState = moduleOp.clone();

    // Initial increment to 1
    int32_t currentIncrement = 1;
    const int32_t maxIncrement = 4;

    while (currentIncrement <= maxIncrement) {
      moduleOp->getRegion(0).takeBody(initialState->getRegion(0));
      moduleOp->setAttrs(initialState->getAttrDictionary());
      initialState = moduleOp.clone();
      LLVM_DEBUG({
        llvm::dbgs() << "Trying with " << currentIncrement << " increment\n\n";
        llvm::dbgs() << "Below is the module:\n\n";
        moduleOp.dump();
        llvm::dbgs()
            << "\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
               "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
               "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n\n";
      });

      IncrementAttributePassOptions incrementAttributeOptions;
      incrementAttributeOptions.incrementAmount = currentIncrement;

      for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
        if (llvm::isa<mlir::func::FuncOp, IREE::Util::InitializerOp,
                      IREE::Util::FuncOp>(callableOp)) {
          OpPassManager passManager(callableOp->getName());

          passManager.addPass(IREE::Stream::createIncrementAttributePass(
              incrementAttributeOptions));
          passManager.addPass(IREE::Stream::createModuloAttributePass());

          // Run the pipeline
          if (failed(runPipeline(passManager, callableOp))) {
            return signalPassFailure();
          }
        }
      }

      OpPassManager passManager(moduleOp->getName());
      passManager.addPass(IREE::Stream::createAddFiveAttributePass());
      if (failed(runPipeline(passManager, moduleOp))) {
        return signalPassFailure();
      }

      // Check the result
      bool allZero = true;
      for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
        if (llvm::isa<mlir::func::FuncOp, IREE::Util::InitializerOp,
                      IREE::Util::FuncOp>(callableOp)) {
          if (auto attr =
                  callableOp->getAttrOfType<IntegerAttr>("arith.number")) {
            if (attr.getInt() != 5) {
              allZero = false;
              break;
            }
          } else {
            callableOp.emitError(
                "Could not find arith.number attribute after pipeline run");
            return signalPassFailure();
          }
        }
      }
      if (allZero)
        return;

      // Increment by 1 for next iteration
      currentIncrement += 1;
    }

    // If we get here, we've exhausted all increment values and haven't found a
    // solution that gives us 0. In this case, we'll return since the current IR
    // comes from the run with the last increment value and represents our "best
    // effort"

    return;
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
