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
  static void analyze(IREE::Stream::AsyncExecuteOp *executeOp,
                      llvm::SmallVector<uint64_t> &maxPartitionValues) {
    uint64_t opCount = 0;
    bool oneDispatch = false;
    bool twoDispatches = false;
    if (executeOp->getRegions().size() > 0) {
      executeOp->walk([&](Operation *nestedOp) {
        if (nestedOp != executeOp &&
            isa<IREE::Stream::StreamableOpInterface>(nestedOp)) {
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
      OpBuilder builder(moduleOp);
      moduleOp->setAttr("iree.stream.partitioning.info",
                        builder.getStringAttr(attrStr));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream