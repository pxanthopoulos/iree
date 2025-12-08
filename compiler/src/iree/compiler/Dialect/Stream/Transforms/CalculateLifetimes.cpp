#include <sys/types.h>
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Dominance.h"

#include <fstream>
#include <string>

#define DEBUG_TYPE "iree-stream-calculate-lifetimes"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_CALCULATELIFETIMESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct CalculateLifetimesPass
    : public IREE::Stream::impl::CalculateLifetimesPassBase<
          CalculateLifetimesPass> {
  using IREE::Stream::impl::CalculateLifetimesPassBase<
      CalculateLifetimesPass>::CalculateLifetimesPassBase;

  uint64_t gatherOpLifetimes(
      mlir::CallableOpInterface callableOp,
      DenseMap<Operation *, std::pair<uint64_t, uint64_t>> &executeOpLifetimes,
      DenseMap<Operation *, uint64_t> &operationTimestamps) {
    uint64_t time = 0;
    callableOp.walk([&](Operation *op) {
      const auto &executeOp = dyn_cast<IREE::Stream::AsyncExecuteOp>(op);
      if (!executeOp) {
        return;
      }

      uint64_t startTime = time;
      executeOpLifetimes[executeOp].first = startTime;

      if (executeOp->getRegions().size() > 0) {
        executeOp->walk([&](Operation *nestedOp) {
          if (nestedOp != executeOp &&
              isa<IREE::Stream::StreamableOpInterface>(nestedOp) &&
              !nestedOp->hasTrait<OpTrait::ConstantLike>() &&
              !isa<IREE::Util::GlobalStoreOpInterface>(nestedOp) &&
              !dyn_cast<IREE::Stream::AsyncConcurrentOp>(nestedOp)) {
            operationTimestamps[nestedOp] = time++;
          }
        });
      }

      executeOpLifetimes[op].second = std::max(time - 1, startTime);
    });

    return time - 2;
  }

  void gatherTensorLifetimes(
      const DenseMap<Operation *, uint64_t> &operationTimestamps,
      const DenseMap<Operation *, std::pair<uint64_t, uint64_t>>
          &executeOpLifetimes,
      uint64_t maxExecuteOpLifetimeEnd, llvm::raw_string_ostream &buffer) {
    for (const auto &[op, timestamp] : operationTimestamps) {
      auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(*op);
      auto sizeAwareOp = dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(op);

      auto executeOp =
          dyn_cast<IREE::Stream::AsyncExecuteOp>(op->getParentOp());
      auto &entryBlock = executeOp.getBody().front();
      auto yieldOp = cast<IREE::Stream::YieldOp>(entryBlock.back());

      for (const auto &[resultIndex, result] :
           llvm::enumerate(op->getResults())) {
        if (!llvm::isa<IREE::Stream::ResourceType>(result.getType())) {
          buffer << "WARNING: Result below is not a stream resource, "
                    "skipping ...\n";
          result.print(buffer);
          buffer << "\n";
          continue;
        }

        if (tiedOp) {
          const auto &tiedOperand = tiedOp.getTiedResultOperand(result);
          if (tiedOperand) {
            buffer << "WARNING: Result below is tied, skipping ...\n";
            result.print(buffer);
            buffer << "\n";
            continue;
          }
        }

        if (!sizeAwareOp) {
          buffer << "WARNING: Result below is not size-aware, skipping ...\n";
          result.print(buffer);
          buffer << "\n";
          continue;
        }

        // dummy default value, partitioning only works for static sizes
        int64_t actualSize = 50;
        const auto &resultSize = sizeAwareOp.getResultSize(resultIndex);
        const auto &valueTypedAttr =
            resultSize.getDefiningOp()->getAttrOfType<TypedAttr>("value");

        if (resultSize.getType().isIntOrIndex() && valueTypedAttr) {
          const auto &integerAttr =
              llvm::dyn_cast<mlir::IntegerAttr>(valueTypedAttr);
          if (integerAttr) {
            actualSize = integerAttr.getInt();
          } else {
            buffer << "WARNING: Failed to get integer from value of result "
                      "size below, skipping ...\n";
            resultSize.print(buffer);
            buffer << "\n";
            continue;
          }
        } else {
          buffer << "WARNING: Size of result below is not int or index, "
                    "or value does not exist, "
                    "skipping ...\n";
          resultSize.print(buffer);
          buffer << "\n";
          continue;
        }

        buffer << "TENSOR\n";
        result.print(buffer);
        buffer << "\nSIZE " << actualSize << "\n";

        bool usedOutside = false;
        size_t yieldOperandIndex;
        for (const auto &[operandIndex, operand] :
             llvm::enumerate(yieldOp.getResourceOperands())) {
          if (result == operand) {
            usedOutside = true;
            yieldOperandIndex = operandIndex;
            break;
          }
        }

        if (!usedOutside) {
          buffer << "START " << timestamp << "\n";
          uint64_t lifetimeEnd = 0;
          for (const auto &use : result.getUses()) {
            lifetimeEnd =
                std::max(lifetimeEnd, operationTimestamps.at(use.getOwner()));
          }
          buffer << "END " << lifetimeEnd << "\n";
        } else {
          buffer << "START " << executeOpLifetimes.at(executeOp).first << "\n";
          auto executeOpResultValue = executeOp.getResults()[yieldOperandIndex];
          uint64_t lifetimeEnd = 0;
          for (const auto &use : executeOpResultValue.getUses()) {
            if (executeOpLifetimes.count(use.getOwner())) {
              lifetimeEnd = std::max(
                  lifetimeEnd, executeOpLifetimes.at(use.getOwner()).second);
            } else {
              lifetimeEnd = maxExecuteOpLifetimeEnd + 1;
              break;
            }
          }
          buffer << "END " << lifetimeEnd << "\n";
        }
      }

      buffer << "\n\n";
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    for (auto &parentOp : llvm::make_early_inc_range(moduleOp.getOps())) {
      auto callableOp = dyn_cast<CallableOpInterface>(parentOp);
      if (!callableOp || !callableOp.getCallableRegion() ||
          callableOp.getCallableRegion()->empty()) {
        continue;
      }

      auto conductLifetimeAnalysisAttr = callableOp->getAttrOfType<IntegerAttr>(
          "iree.stream.conduct_lifetime_analysis");
      if (!conductLifetimeAnalysisAttr) {
        LLVM_DEBUG(llvm::dbgs() << "No conduct lifetime analysis attr\n");
        continue;
      }

      int64_t conductLifetimeAnalysis = conductLifetimeAnalysisAttr.getInt();
      if (conductLifetimeAnalysis != 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping lifetime analysis as per attribute\n");
        continue;
      }

      DenseMap<Operation *, std::pair<uint64_t, uint64_t>> executeOpLifetimes;
      DenseMap<Operation *, uint64_t> operationTimestamps;

      uint64_t maxExecuteOpLifetimeEnd = gatherOpLifetimes(
          callableOp, executeOpLifetimes, operationTimestamps);

      std::string bufferStr;
      llvm::raw_string_ostream buffer(bufferStr);

      gatherTensorLifetimes(operationTimestamps, executeOpLifetimes,
                            maxExecuteOpLifetimeEnd, buffer);

      buffer.flush();

      std::string filename = std::string("tensor_lifetimes_") +
                             std::to_string(fileSuffixId) + std::string(".txt");
      std::ofstream file(filename);
      if (!file.is_open()) {
        llvm::errs() << "Error: Could not open file " << filename << "\n";
        return signalPassFailure();
      }

      file << bufferStr;
      file.close();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream