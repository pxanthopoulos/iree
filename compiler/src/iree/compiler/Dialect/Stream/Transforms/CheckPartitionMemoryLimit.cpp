// #include <cstdint>
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
      LLVM_DEBUG(llvm::dbgs() << "No predecessor attribute\n");
      return failure();
    }
    int64_t predecessor = predecessorAttr.getInt();
    if (predecessor >= results.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Predecessor attribute is larger than the vector size\n");
      return failure();
    }

    auto sizeAttr =
        executeOp->getAttrOfType<IntegerAttr>("iree.stream.partitioning.size");
    int64_t size = sizeAttr ? sizeAttr.getInt() : 0;

    if (size > clMemoryAwarePartitioningMemoryLimit) {
      results[predecessor] = false;
      LLVM_DEBUG(llvm::dbgs()
                 << "Partition " << predecessor << " exceeds memory limit "
                 << clMemoryAwarePartitioningMemoryLimit << "\n");
    }

    return success();
  }

  static std::string createAttributeString(
      const llvm::SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
          &values) {
    std::string result;
    llvm::raw_string_ostream os(result);

    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << std::get<0>(values[i]) << ":" << std::get<1>(values[i]) << ":"
         << std::get<2>(values[i]) << ":" << std::get<3>(values[i]);
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

    auto constEvalAttr = moduleOp->getAttr("iree.consteval");
    if (constEvalAttr) {
      OpBuilder builder(moduleOp);
      moduleOp->setAttr("iree.stream.partitioning.info",
                        builder.getStringAttr("pass"));
      return;
    }

    llvm::StringRef partitioningInfoStr = partitioningInfoAttr.getValue();
    llvm::SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
        partitioningInfo;
    llvm::SmallVector<llvm::StringRef> pairStrs;
    partitioningInfoStr.split(pairStrs, ',');

    for (auto pairStr : pairStrs) {
      llvm::SmallVector<llvm::StringRef> nums;
      pairStr.split(nums, ':');

      if (nums.size() != 4) {
        LLVM_DEBUG(llvm::dbgs() << "Invalid pair format: " << pairStr << "\n");
        return signalPassFailure();
      }

      int64_t first, second, third, fourth;
      if (nums[0].getAsInteger(10, first) || nums[1].getAsInteger(10, second) ||
          nums[2].getAsInteger(10, third) || nums[3].getAsInteger(10, fourth)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to parse integers: " << pairStr << "\n");
        return signalPassFailure();
      }

      partitioningInfo.emplace_back(first, second, third, fourth);
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

    bool runAgain = false;
    for (size_t i = 0; i < results.size(); ++i) {
      // If the predecessor of this execution region needs to be partitioned
      // further
      if (!results[i]) {
        // If mid == top for the binary search
        if (std::get<1>(partitioningInfo[i]) ==
            std::get<2>(partitioningInfo[i])) {
          // Most likely, we reached the max number of partitions and must fail
          if (std::get<2>(partitioningInfo[i]) ==
              std::get<3>(partitioningInfo[i])) {
            return signalPassFailure();
          }
          // Edge case: Top signifies the biggest number of partitions that
          // succeeded. Mid signifies the current number of partitions that was
          // attempted. But mid failed and mid == top? This can happen due to
          // randomness in the partition algorithm. In this case, we will try
          // higher partition numbers linearly, until one succeeds
          else {
            runAgain = true;
            std::get<0>(partitioningInfo[i])++;
            std::get<1>(partitioningInfo[i])++;
            std::get<2>(partitioningInfo[i])++;
          }
        }
        // Change bottom and mid accordingly and try again
        else {
          runAgain = true;
          std::get<0>(partitioningInfo[i]) = std::get<1>(partitioningInfo[i]);
          std::get<1>(partitioningInfo[i]) = std::get<0>(partitioningInfo[i]) +
                                             std::get<2>(partitioningInfo[i]);
          std::get<1>(partitioningInfo[i]) =
              (std::get<1>(partitioningInfo[i]) + 1) / 2;
        }
      }
      // If the precessor of this execution region need not be partitioned
      // further, we can try lower parition numbers
      else {
        // If mid != top, we can update mid and top accordingly and try again
        if (std::get<1>(partitioningInfo[i]) !=
            std::get<2>(partitioningInfo[i])) {
          runAgain = true;
          std::get<2>(partitioningInfo[i]) = std::get<1>(partitioningInfo[i]);
          std::get<1>(partitioningInfo[i]) = std::get<0>(partitioningInfo[i]) +
                                             std::get<2>(partitioningInfo[i]);
          std::get<1>(partitioningInfo[i]) =
              (std::get<1>(partitioningInfo[i]) + 1) / 2;
        }
        // Else, we do not have any more parition numbers to check and we should
        // exit the loop
      }
    }

    if (!runAgain) {
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