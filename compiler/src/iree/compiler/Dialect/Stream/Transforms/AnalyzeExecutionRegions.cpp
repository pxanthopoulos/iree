#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

// #include "iree/compiler/Utils/PassUtils.h"
// #include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/Support/Debug.h"
// #include "mlir/IR/Matchers.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/Pass/Pass.h"
// #include "llvm/Support/CommandLine.h"
// #include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-stream-analyze-execution-regions"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANALYZEEXECUTIONREGIONSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct AnalyzeExecutionRegionsPass
    : public IREE::Stream::impl::AnalyzeExecutionRegionsPassBase<
          AnalyzeExecutionRegionsPass> {
  static void analyze(Operation *executeOp,
                      llvm::SmallVector<uint64_t> &maxPartitionValues) {
    uint64_t opCount = 0;
    bool oneDispatch = false;
    bool twoDispatches = false;
    if (executeOp->getRegions().size() > 0) {
      executeOp->walk([&](Operation *nestedOp) {
        if (nestedOp != executeOp) {
          opCount++;
        }
        if (!twoDispatches) {
          auto dispatchOp =
              llvm::dyn_cast<IREE::Stream::AsyncDispatchOp>(nestedOp);
          if (dispatchOp && oneDispatch)
            twoDispatches = true;
          else if (dispatchOp) {
            oneDispatch = true;
          }
        }
      });
    }
    if (twoDispatches) {
      maxPartitionValues.push_back(opCount);
    }
  }

  static std::string
  createAttributeString(const llvm::SmallVector<uint64_t> &values) {
    std::string result;
    llvm::raw_string_ostream os(result);

    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << "1:" << values[i];
    }

    return result;
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    for (auto &parentOp : llvm::make_early_inc_range(moduleOp.getOps())) {
      if (auto asyncFuncOp = dyn_cast<IREE::Stream::AsyncFuncOp>(parentOp)) {
        continue;
      }
      auto callableOp = dyn_cast<CallableOpInterface>(parentOp);
      if (!callableOp || !callableOp.getCallableRegion() ||
          callableOp.getCallableRegion()->empty()) {
        continue;
      }

      llvm::SmallVector<Operation *> operations;
      callableOp.walk([&](Operation *op) { operations.push_back(op); });

      llvm::SmallVector<uint64_t> info;
      for (auto op : operations) {
        auto executeOp = llvm::dyn_cast<IREE::Stream::AsyncExecuteOp>(op);
        if (executeOp) {
          analyze(executeOp, info);
        }
      }

      if (info.empty()) {
        continue;
      }

      std::string attrStr = createAttributeString(info);
      OpBuilder builder(callableOp);
      callableOp->setAttr("iree.stream.partitioning.info",
                          builder.getStringAttr(attrStr));

      LLVM_DEBUG(callableOp.dump());
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream