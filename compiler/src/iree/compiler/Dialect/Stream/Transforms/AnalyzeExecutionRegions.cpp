#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-stream-analyze-execution-regions"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANALYZEEXECUTIONREGIONSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct AnalyzeExecutionRegionsPass
    : public IREE::Stream::impl::AnalyzeExecutionRegionsPassBase<
          AnalyzeExecutionRegionsPass> {
  static void analyze(Operation *executeOp,
                      llvm::SmallVector<int64_t> &maxPartitionValues) {
    int64_t realOps = 0;
    uint64_t totalDispatches = 0;
    uint64_t totalOps = 0;
    if (executeOp->getRegions().size() > 0) {
      executeOp->walk([&](Operation *nestedOp) {
        totalOps++;
        if (nestedOp != executeOp &&
            isa<IREE::Stream::StreamableOpInterface>(nestedOp) &&
            !nestedOp->hasTrait<OpTrait::ConstantLike>() &&
            !isa<IREE::Util::GlobalStoreOpInterface>(nestedOp) &&
            !dyn_cast<IREE::Stream::AsyncConcurrentOp>(nestedOp)) {
          realOps++;
        }
        auto dispatchOp =
            llvm::dyn_cast<IREE::Stream::AsyncDispatchOp>(nestedOp);
        if (dispatchOp)
          totalDispatches++;
      });
    }
    LLVM_DEBUG({
      llvm::dbgs() << "ExecuteOp:\n";
      executeOp->dump();
      llvm::dbgs() << "Real ops: " << realOps << "\n";
      llvm::dbgs() << "Total ops: " << totalOps << "\n";
      llvm::dbgs() << "Total dispatches: " << totalDispatches << "\n";
    });
    if (totalDispatches > totalOps / 5) {
      maxPartitionValues.push_back(realOps);
    }
  }

  static std::string
  createAttributeString(const llvm::SmallVector<int64_t> &values) {
    std::string result;
    llvm::raw_string_ostream os(result);

    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << "1:1:" << values[i] << ":" << values[i];
    }

    LLVM_DEBUG(llvm::dbgs() << "Created partitioning info attribute string: "
                            << result << "\n");

    return result;
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    llvm::SmallVector<int64_t> info;

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

      for (auto op : operations) {
        auto executeOp = llvm::dyn_cast<IREE::Stream::AsyncExecuteOp>(op);
        if (executeOp) {
          analyze(executeOp, info);
        }
      }
    }

    if (info.empty()) {
      return;
    }

    std::string attrStr = createAttributeString(info);
    OpBuilder builder(moduleOp);
    moduleOp->setAttr("iree.stream.partitioning.info",
                      builder.getStringAttr(attrStr));
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream