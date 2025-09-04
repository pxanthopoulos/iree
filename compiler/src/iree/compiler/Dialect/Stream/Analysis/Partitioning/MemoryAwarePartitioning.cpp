#include "Graph.h"
#include "RecursivePartitioner.h"

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"

#include <fstream>

#define DEBUG_TYPE "iree-stream-memory-aware-partitioning"

namespace mlir::iree_compiler::IREE::Stream {

static llvm::cl::opt<std::string> clMemoryAwarePartitioningIODir(
    "iree-stream-memory-aware-partitioning-io-dir",
    llvm::cl::desc("Memory aware partitioning IO dir for dot graphs"),
    llvm::cl::init("."));

static std::unique_ptr<AsmState> getRootAsmState(Block *block) {
  auto *rootOp = block->getParentOp();
  while (auto parentOp = rootOp->getParentOp()) {
    if (!isa<IREE::Stream::TimelineOpInterface>(parentOp) &&
        parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      rootOp = parentOp;
      break;
    }
    rootOp = parentOp;
  }
  return std::make_unique<AsmState>(rootOp);
}

bool checkSomeDispatches(Partition partition) {
  for (const auto &op : partition.ops) {
    auto dispatchOp = llvm::dyn_cast<IREE::Stream::AsyncDispatchOp>(op);
    if (dispatchOp)
      return true;
  }
  return false;
}

bool partitionToDotGraph(size_t partitionIndex, Partition &partition) {
  std::ofstream outFile(clMemoryAwarePartitioningIODir + "/partition-graph-" +
                            std::to_string(partitionIndex) + ".dot",
                        std::ios::trunc);
  if (!outFile) {
    llvm::errs() << "Failed to open file: " << clMemoryAwarePartitioningIODir
                 << "/partition-graph-" << partitionIndex
                 << ".dot\nReturning initial partitions.\n";
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "File: " << clMemoryAwarePartitioningIODir
                          << "/partition-graph-" << partitionIndex
                          << ".dot opened successfully.\n");

  outFile << "digraph cfg {\n";

  llvm::DenseMap<unsigned, Operation *> opMap;
  llvm::DenseMap<Operation *, unsigned> inverseOpMap;
  for (const auto [opIndex, op] :
       llvm::enumerate(llvm::reverse(partition.ops))) {
    opMap[opIndex] = op;
    inverseOpMap[op] = opIndex;
    int64_t opWeight = 0;
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(*op);
    auto sizeAwareOp = dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(op);

    for (const auto &[resultIndex, result] :
         llvm::enumerate(op->getResults())) {
      if (!llvm::isa<IREE::Stream::ResourceType>(result.getType())) {
        continue;
      }

      if (tiedOp) {
        auto tiedOperand = tiedOp.getTiedResultOperand(result);
        if (tiedOperand) {
          continue;
        }
      }

      // dummy default value, partitioning only works for static sizes

      int64_t actualSize = 50;
      if (sizeAwareOp) {
        auto resultSize = sizeAwareOp.getResultSizeFromValue(result);
        auto valueTypedAttr =
            resultSize.getDefiningOp()->getAttrOfType<TypedAttr>("value");

        if (resultSize.getType().isIntOrIndex() && valueTypedAttr) {
          auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueTypedAttr);
          if (integerAttr) {
            actualSize = integerAttr.getInt();
          }
        }
      }
      opWeight += actualSize;
    }
    outFile << opIndex << "[weight=" << opWeight << "];\n";
  }

  for (const auto op : llvm::reverse(partition.ops)) {
    for (const Value &value : op->getOperands()) {
      Operation *definingOp = value.getDefiningOp();
      if (partition.ops.contains(definingOp)) {
        int64_t actualSize = 50;

        auto sizeAwareOp =
            dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(definingOp);
        if (sizeAwareOp) {
          auto resultSize = sizeAwareOp.getResultSizeFromValue(value);
          auto valueTypedAttr =
              resultSize.getDefiningOp()->getAttrOfType<TypedAttr>("value");
          if (resultSize.getType().isIntOrIndex() && valueTypedAttr) {
            actualSize =
                llvm::dyn_cast<mlir::IntegerAttr>(valueTypedAttr).getInt();
          }
        }

        unsigned parentOpIndex = inverseOpMap[definingOp];
        unsigned opIndex = inverseOpMap[op];
        outFile << parentOpIndex << "->" << opIndex << "[weight=" << actualSize
                << "];\n";
      }
    }
  }
  outFile << "}\n";
  outFile.close();
  return true;
}

PartitionSet memoryAwarePartition(PartitionSet initialPartitions,
                                  Block *block) {
  PartitionSet result;

  for (const auto &[partitionIndex, partition] :
       llvm::enumerate(initialPartitions.partitions)) {
    if (!checkSomeDispatches(partition)) {
      result.partitions.push_back(std::move(partition));
      continue;
    }

    if (!partitionToDotGraph(partitionIndex, partition))
      result.partitions.push_back(std::move(partition));
    else {
      dag_partitioning::core::Graph graph = dag_partitioning::core::readDotFile(
          clMemoryAwarePartitioningIODir + "/partition-graph-" +
              std::to_string(partitionIndex) + ".dot",
          clMemoryAwarePartitioningIODir + "/partition-graph-" +
              std::to_string(partitionIndex) + ".dot.nodemappings");

      const uint64_t partitionNum =
          std::min((uint64_t)10, partition.ops.size());
      const uint64_t maxLevel = 20;
      const uint64_t minSize = 50 * partitionNum;
      const double vertRatio = 0.9;
      const double maxImbalance = 1.1;
      const uint64_t maxPasses = 10;
      const bool enableParallel = true;

      dag_partitioning::driver::RecursivePartitioner partitioner(
          graph, partitionNum, "HYB", maxLevel, minSize, vertRatio, "UNDIRBOTH",
          maxImbalance, "MIXED", maxPasses, enableParallel);

      auto [partitionInfo, cutSize] = partitioner.run();

      llvm::dbgs() << "Cut size: " << cutSize << "\n";
      result.partitions.push_back(std::move(partition));
    }
  }

  return result;
}

} // namespace mlir::iree_compiler::IREE::Stream