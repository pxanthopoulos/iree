#include <cstdint>
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "iree-stream-check-partition-memory-limit"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_CHECKPARTITIONMEMORYLIMITPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

static llvm::cl::opt<int64_t> clMemoryAwarePartitioningMemoryLimit(
    "iree-stream-memory-aware-partitioning-memory-limit",
    llvm::cl::desc("Memory limit in Bytes for memory aware partitioning"),
    llvm::cl::init(INT64_MAX));

namespace {

struct CheckPartitionMemoryLimitPass
    : public IREE::Stream::impl::CheckPartitionMemoryLimitPassBase<
          CheckPartitionMemoryLimitPass> {
  static LogicalResult check(Operation *executeOp,
                             llvm::SmallVector<bool> &results) {
    auto predecessorAttr = executeOp->getAttrOfType<IntegerAttr>(
        "iree.stream.partitioning.predecessor");
    if (!predecessorAttr) {
      return failure();
    }
    int64_t predecessor = predecessorAttr.getInt();
    if (predecessor >= results.size()) {
      return failure();
    }

    auto sizeAttr =
        executeOp->getAttrOfType<IntegerAttr>("iree.stream.partitioning.size");
    int64_t size = sizeAttr ? sizeAttr.getInt() : 0;

    LLVM_DEBUG({
      llvm::dbgs() << "predecessor: " << predecessor << "\nsize: " << size
                   << "\n";
      executeOp->dump();
    });

    if (size > clMemoryAwarePartitioningMemoryLimit) {
      results[predecessor] = false;
      LLVM_DEBUG(llvm::dbgs()
                 << "Partition " << predecessor << " exceeds memory limit "
                 << clMemoryAwarePartitioningMemoryLimit << "\n");
    }

    return success();
  }

  static std::string createAttributeString(
      const llvm::SmallVector<std::pair<int64_t, int64_t>> &values) {
    std::string result;
    llvm::raw_string_ostream os(result);

    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << values[i].first << ":" << values[i].second;
    }

    return result;
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    auto partitioningInfoAttr =
        moduleOp->getAttrOfType<StringAttr>("iree.stream.partitioning.info");
    if (!partitioningInfoAttr) {
      LLVM_DEBUG(llvm::dbgs() << "No partitioning info attribute found\n");
      return signalPassFailure();
    }

    llvm::StringRef partitioningInfoStr = partitioningInfoAttr.getValue();
    llvm::SmallVector<std::pair<int64_t, int64_t>> partitioningInfo;
    llvm::SmallVector<llvm::StringRef> pairStrs;
    partitioningInfoStr.split(pairStrs, ',');

    for (auto pairStr : pairStrs) {
      llvm::SmallVector<llvm::StringRef> nums;
      pairStr.split(nums, ':');

      if (nums.size() != 2) {
        LLVM_DEBUG(llvm::dbgs() << "Invalid pair format: " << pairStr << "\n");
        return signalPassFailure();
      }

      int64_t first, second;
      if (nums[0].getAsInteger(10, first) || nums[1].getAsInteger(10, second)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to parse integers: " << pairStr << "\n");
        return signalPassFailure();
      }

      partitioningInfo.emplace_back(first, second);
    }

    llvm::SmallVector<bool> results(partitioningInfo.size(), true);

    for (auto &parentOp : llvm::make_early_inc_range(moduleOp.getOps())) {
      auto callableOp = dyn_cast<CallableOpInterface>(parentOp);
      if (!callableOp || !callableOp.getCallableRegion() ||
          callableOp.getCallableRegion()->empty()) {
        continue;
      }

      llvm::SmallVector<Operation *> operations;
      callableOp.walk([&](Operation *op) { operations.push_back(op); });

      for (auto op : operations) {
        auto executeOp = llvm::dyn_cast<IREE::Stream::CmdExecuteOp>(op);
        if (executeOp) {
          if (failed(check(executeOp, results))) {
            return signalPassFailure();
          }
        }
      }
    }

    bool oneFailure = false;
    for (size_t i = 0; i < results.size(); ++i) {
      if (!results[i]) {
        partitioningInfo[i].first++;
        if (partitioningInfo[i].first > partitioningInfo[i].second) {
          return signalPassFailure();
        }
        oneFailure = true;
      }
    }
    if (!oneFailure) {
      OpBuilder builder(moduleOp);
      moduleOp->setAttr("iree.stream.partitioning.info",
                        builder.getStringAttr("pass"));
    } else {
      std::string attrStr = createAttributeString(partitioningInfo);
      OpBuilder builder(moduleOp);
      moduleOp->setAttr("iree.stream.partitioning.info",
                        builder.getStringAttr(attrStr));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream