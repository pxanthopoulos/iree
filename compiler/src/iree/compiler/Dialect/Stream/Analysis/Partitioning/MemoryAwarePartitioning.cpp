#include "Graph.h"
#include "RecursivePartitioner.h"
#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/AsmState.h"

#include <filesystem>
#include <fstream>

#define DEBUG_TYPE "iree-stream-memory-aware-partitioning-partitioner"

namespace mlir::iree_compiler::IREE::Stream {

static llvm::cl::opt<std::string> clMemoryAwarePartitioningIODir(
    "iree-stream-memory-aware-partitioning-io-dir",
    llvm::cl::desc("Memory aware partitioning IO dir for dot graphs"),
    llvm::cl::init("."));

struct MemoryAwarePartitioningConfig {
  ClusteringMethod clusteringMethod;
  uint64_t maxClusteringLevel;
  double minClusteringVertexRatio;
  BisectionMethod bisectionMethod;
  double maxImbalance;
  RefinementMethod refinementMethod;
  uint64_t maxRefinementPasses;

  MemoryAwarePartitioningConfig(const char *V = "HYB,20,0.9,BOTH,1.1,MIX,10") {
    clusteringMethod = ClusteringMethod::HYB;
    maxClusteringLevel = 20;
    minClusteringVertexRatio = 0.9;
    bisectionMethod = BisectionMethod::UNDIRBOTH;
    maxImbalance = 1.1;
    refinementMethod = RefinementMethod::MIXED;
    maxRefinementPasses = 10;
  }

  void print() const {
    LLVM_DEBUG({
      llvm::dbgs() << "Memory-Aware Partitioning Configuration:\n"
                   << "  Clustering Method: ";

      switch (clusteringMethod) {
      case ClusteringMethod::FORB:
        llvm::dbgs() << "FORB (Forbidden edges)";
        break;
      case ClusteringMethod::CYC:
        llvm::dbgs() << "CYC (Cycle detection)";
        break;
      case ClusteringMethod::HYB:
        llvm::dbgs() << "HYB (Hybrid approach)";
        break;
      }

      llvm::dbgs() << "\n  Max Clustering Level: " << maxClusteringLevel
                   << "\n  Min Clustering Vertex Ratio: "
                   << minClusteringVertexRatio << "\n  Bisection Method: ";

      switch (bisectionMethod) {
      case BisectionMethod::GGG:
        llvm::dbgs() << "GGG (Greedy directed graph growing)";
        break;
      case BisectionMethod::UNDIRMETIS:
        llvm::dbgs() << "UNDIRMETIS (Undirected METIS)";
        break;
      case BisectionMethod::UNDIRSCOTCH:
        llvm::dbgs() << "UNDIRSCOTCH (Undirected Scotch)";
        break;
      case BisectionMethod::UNDIRBOTH:
        llvm::dbgs() << "UNDIRBOTH (Try both METIS and Scotch)";
        break;
      }

      llvm::dbgs() << "\n  Max Imbalance: " << maxImbalance
                   << "\n  Refinement Method: ";

      switch (refinementMethod) {
      case RefinementMethod::BOUNDARYFM:
        llvm::dbgs() << "BOUNDARYFM (Boundary FM adaptation)";
        break;
      case RefinementMethod::BOUNDARYKL:
        llvm::dbgs() << "BOUNDARYKL (Boundary KL adaptation)";
        break;
      case RefinementMethod::MIXED:
        llvm::dbgs() << "MIXED (KL + FM)";
        break;
      }

      llvm::dbgs() << "\n  Max Refinement Passes: " << maxRefinementPasses
                   << "\n";
    });
  }
};

struct MemoryAwarePartitioningConfigParser
    : public llvm::cl::parser<MemoryAwarePartitioningConfig> {
  MemoryAwarePartitioningConfigParser(llvm::cl::Option &O)
      : llvm::cl::parser<MemoryAwarePartitioningConfig>(O) {}

  bool parse(llvm::cl::Option &O, StringRef ArgName, StringRef Arg,
             MemoryAwarePartitioningConfig &V) {
    SmallVector<StringRef> Parts;
    Arg.split(Parts, ',');

    if (Parts.size() != 7)
      return O.error("expected 7 comma-separated values");

    if (Parts[0].compare("FORB") == 0) {
      V.clusteringMethod = ClusteringMethod::FORB;
    } else if (Parts[0].compare("CYC") == 0) {
      V.clusteringMethod = ClusteringMethod::CYC;
    } else if (Parts[0].compare("HYB") == 0) {
      V.clusteringMethod = ClusteringMethod::HYB;
    } else {
      return O.error("invalid clustering method (expected: FORB|CYC|HYB)");
    }

    if (Parts[1].getAsInteger(10, V.maxClusteringLevel))
      return O.error("invalid clustering level");

    double Ratio;
    if (Parts[2].getAsDouble(Ratio) || Ratio <= 0.0 || Ratio >= 1.0)
      return O.error("clustering ratio must be between 0 and 1");
    V.minClusteringVertexRatio = Ratio;

    if (Parts[3].compare("GGG") == 0) {
      V.bisectionMethod = BisectionMethod::GGG;
    } else if (Parts[3].compare("SCOTCH") == 0) {
      V.bisectionMethod = BisectionMethod::UNDIRSCOTCH;
    } else if (Parts[3].compare("METIS") == 0) {
      V.bisectionMethod = BisectionMethod::UNDIRMETIS;
    } else if (Parts[3].compare("BOTH") == 0) {
      V.bisectionMethod = BisectionMethod::UNDIRBOTH;
    } else {
      return O.error(
          "invalid bisection method (expected: GGG|SCOTCH|METIS|BOTH)");
    }

    double Imbalance;
    if (Parts[4].getAsDouble(Imbalance) || Imbalance <= 1.0)
      return O.error("imbalance must be greater than 1");
    V.maxImbalance = Imbalance;

    if (Parts[5].compare("FM") == 0) {
      V.refinementMethod = RefinementMethod::BOUNDARYFM;
    } else if (Parts[5].compare("KL") == 0) {
      V.refinementMethod = RefinementMethod::BOUNDARYKL;
    } else if (Parts[5].compare("MIX") == 0) {
      V.refinementMethod = RefinementMethod::MIXED;
    } else {
      return O.error("invalid refinement method (expected: FM|KL|MIX)");
    }

    if (Parts[6].getAsInteger(10, V.maxRefinementPasses))
      return O.error("invalid refinement passes");

    return false;
  }
};

static llvm::cl::opt<MemoryAwarePartitioningConfig, false,
                     MemoryAwarePartitioningConfigParser>
    clMemoryAwarePartitioningConfig(
        "iree-stream-memory-aware-partitioning-config",
        llvm::cl::desc(
            "--iree-stream-memory-aware-partitioning-config=cm,cl,vr,bm,i,"
            "rm,rp\n"
            "Comma-separated list of configuration variables for the "
            "partitioning phase:\n"
            "  cm = clusteringMethod (FORB|CYC|HYB)\n"
            "  cl = maxClusteringLevel (int)\n"
            "  vr = minClusteringVertexRatio (float, 0.0 < vr < 1.0)\n"
            "  bm = bisectionMethod (GGG|SCOTCH|METIS|BOTH)\n"
            "  i  = maxImbalance (float, 1.0 < i)\n"
            "  rm = refinementMethod (FM|KL|MIX)\n"
            "  rp = maxRefinementPasses (int)\n"
            "Example: "
            "--iree-stream-memory-aware-partitioning-config=HYB,20,0.9,BOTH,1."
            "1,MIX,10"),
        llvm::cl::init("HYB,20,0.9,BOTH,1.1,MIX,10"));

static std::unique_ptr<AsmState> getRootAsmState(Block *block) {
  LLVM_DEBUG({
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
  });
  return nullptr;
}

llvm::Expected<llvm::SmallVector<int64_t>>
extractNumPartitionsFromAttr(Block *block) {
  llvm::SmallVector<int64_t> result;

  auto *rootOp = block->getParentOp();
  while (auto parentOp = rootOp->getParentOp()) {
    if (!isa<IREE::Stream::TimelineOpInterface>(parentOp) &&
        parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      rootOp = parentOp;
      break;
    }
    rootOp = parentOp;
  }

  auto partitioningInfoAttr =
      rootOp->getAttrOfType<StringAttr>("iree.stream.partitioning.info");
  if (!partitioningInfoAttr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No partitioning info attribute found\n");
  }

  llvm::StringRef partitioningInfoStr = partitioningInfoAttr.getValue();
  llvm::SmallVector<llvm::StringRef> pairStrs;
  partitioningInfoStr.split(pairStrs, ',');

  for (auto pairStr : pairStrs) {
    llvm::SmallVector<llvm::StringRef> nums;
    pairStr.split(nums, ':');

    if (nums.size() != 4) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          llvm::formatv("Invalid pair format: {0}\n", pairStr));
    }

    int64_t first, second, third, fourth;
    if (nums[0].getAsInteger(10, first) || nums[1].getAsInteger(10, second) ||
        nums[2].getAsInteger(10, third) || nums[3].getAsInteger(10, fourth)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          llvm::formatv("Failed to parse integers: {0}\n", pairStr));
    }

    result.emplace_back(second);
  }

  return result;
}

bool checkOneDispatch(Partition partition) {
  for (const auto &op : partition.ops) {
    auto dispatchOp = llvm::dyn_cast<IREE::Stream::AsyncDispatchOp>(op);
    if (dispatchOp)
      return true;
  }
  return false;
}

llvm::Expected<DenseMap<unsigned, Operation *>>
partitionToDotGraph(int64_t partitionIndex, Partition &partition) {
  std::ofstream outFile(clMemoryAwarePartitioningIODir + "/partition-graph-" +
                            std::to_string(partitionIndex) + ".dot",
                        std::ios::trunc);
  if (!outFile) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::formatv(
            "Failed to open file: {0}/partition-graph-{1}.dot\nReturning "
            "initial partitions.\n",
            clMemoryAwarePartitioningIODir, partitionIndex));
  }

  LLVM_DEBUG(llvm::dbgs() << "File: " << clMemoryAwarePartitioningIODir
                          << "/partition-graph-" << partitionIndex
                          << ".dot opened successfully.\n");

  outFile << "// size=" << partition.ops.size() << "\n";
  outFile << "digraph cfg {\n";

  DenseMap<unsigned, Operation *> opMap;
  DenseMap<Operation *, unsigned> inverseOpMap;
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

  std::map<std::pair<unsigned, unsigned>, int64_t> edgeWeights;

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
        std::pair<unsigned, unsigned> edge = {parentOpIndex, opIndex};
        edgeWeights[edge] += actualSize;
      }
    }
  }

  for (const auto &edgeEntry : edgeWeights) {
    const auto &edge = edgeEntry.first;
    int64_t totalWeight = edgeEntry.second;

    outFile << edge.first << "->" << edge.second << "[weight=" << totalWeight
            << "];\n";
  }

  outFile << "}\n";
  outFile.close();
  return llvm::Expected<DenseMap<unsigned, Operation *>>(std::move(opMap));
}

uint64_t calculateMaxPartSize(const std::vector<uint64_t> &partitionInfo,
                              const Graph &graph, uint64_t numPartitions) {
  std::vector<uint64_t> partSizes(numPartitions, 0);
  for (uint64_t i = 0; i < partitionInfo.size(); ++i) {
    partSizes[partitionInfo[i]] += graph.nodeWeights[i];
  }
  return *std::max_element(partSizes.begin(), partSizes.end());
}

SmallVector<SetVector<Operation *>>
createOpGroups(const std::vector<uint64_t> &partitionInfo,
               const DenseMap<unsigned, Operation *> &opMap,
               const std::vector<uint64_t> &topSort,
               const std::vector<uint64_t> &groupedTopSortPositions) {
  struct PartitionData {
    SmallVector<std::pair<Operation *, uint64_t>> ops;
    uint64_t groupedTopSortPosition = 0;
  };
  DenseMap<uint64_t, PartitionData> partitionData;

  std::vector<uint64_t> topSortPositions(partitionInfo.size());
  for (unsigned i = 0; i < topSortPositions.size(); ++i) {
    topSortPositions[topSort[i]] = i;
  }

  for (unsigned i = 0; i < partitionInfo.size(); i++) {
    if (auto op = opMap.lookup(i)) {
      uint64_t partId = partitionInfo[i];
      auto &data = partitionData[partId];
      data.ops.push_back({op, topSortPositions[i]});
      data.groupedTopSortPosition = groupedTopSortPositions[i];
    }
  }

  SmallVector<uint64_t> sortedPartitions;
  for (const auto &entry : partitionData)
    sortedPartitions.push_back(entry.first);

  llvm::sort(sortedPartitions, [&](uint64_t a, uint64_t b) {
    return partitionData[a].groupedTopSortPosition <
           partitionData[b].groupedTopSortPosition;
  });

  SmallVector<SetVector<Operation *>> result;
  for (uint64_t partitionId : sortedPartitions) {
    auto &ops = partitionData[partitionId].ops;
    llvm::sort(
        ops, [](const auto &a, const auto &b) { return a.second < b.second; });

    SetVector<Operation *> partitionOps;
    for (const auto &op : ops)
      partitionOps.insert(op.first);

    result.push_back(std::move(partitionOps));
  }

  LLVM_DEBUG({
    DenseMap<Operation *, unsigned> inverseOpMap;
    for (const auto &pair : opMap) {
      unsigned key = pair.first;
      Operation *value = pair.second;
      inverseOpMap[value] = key;
    }

    llvm::dbgs() << "Partition Data:\n";
    llvm::dbgs() << "==============\n\n";

    for (uint64_t partId : sortedPartitions) {
      const auto &data = partitionData[partId];

      llvm::dbgs() << "Partition " << partId << ":\n";
      llvm::dbgs() << "  Grouped TopSort position: "
                   << data.groupedTopSortPosition << "\n";
      llvm::dbgs() << "  Operations (" << data.ops.size() << "):\n";

      for (const auto &[op, topSortPos] : data.ops) {
        llvm::dbgs() << "    [" << inverseOpMap[op] << "][" << topSortPos
                     << "] ";
        op->dump();
      }

      llvm::dbgs() << "\n";
    }
  });

  return result;
}

SmallVector<Partition>
createPartitions(const std::vector<uint64_t> &partitionInfo,
                 const DenseMap<unsigned, Operation *> &opMap,
                 const std::vector<uint64_t> &topSort,
                 const std::vector<uint64_t> &groupedTopSortPositions,
                 IREE::Stream::AffinityAttr affinity, int64_t partitionIndex) {
  auto opGroups =
      createOpGroups(partitionInfo, opMap, topSort, groupedTopSortPositions);
  SmallVector<Partition> result;

  for (auto &opGroup : opGroups) {
    Partition partition;

    SetVector<Value> consumedValues;
    SetVector<Value> producedValues;
    SetVector<Value> escapingValues;
    for (auto *op : opGroup) {
      for (auto operand : op->getOperands()) {
        consumedValues.insert(operand);
      }
      for (auto result : op->getResults()) {
        producedValues.insert(result);

        for (auto user : result.getUsers()) {
          if (!opGroup.contains(user)) {
            escapingValues.insert(result);
          }
        }
      }
    }
    consumedValues.set_subtract(producedValues);
    partition.affinity = affinity;
    partition.ins = consumedValues;
    partition.outs = escapingValues;

    partition.ops = std::move(opGroup);
    partition.predecessorPartition = partitionIndex;
    result.push_back(std::move(partition));
  }

  return result;
}

PartitionSet memoryAwarePartition(PartitionSet initialPartitions,
                                  Block *block) {
  auto asmState = getRootAsmState(block);
  PartitionSet result;

  llvm::SmallVector<int64_t> numPartitionsFromAttr;
  auto numPartitionsFromAttrPtr = extractNumPartitionsFromAttr(block);
  if (numPartitionsFromAttrPtr) {
    numPartitionsFromAttr = *numPartitionsFromAttrPtr;
  } else {
    LLVM_DEBUG({
      llvm::dbgs() << "Failed to extract partition info: "
                   << llvm::toString(numPartitionsFromAttrPtr.takeError())
                   << "\n";
    });
    return initialPartitions;
  }

  int64_t partitionIndex = 0;
  for (auto &partition : initialPartitions.partitions) {
    if (!checkOneDispatch(partition)) {
      result.partitions.push_back(std::move(partition));
      continue;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "On partition:\n\n";
      partition.dump(*asmState);
      llvm::dbgs() << "\n\n";
    });

    auto opMapPtr = partitionToDotGraph(partitionIndex, partition);
    if (!opMapPtr) {
      result.partitions.push_back(std::move(partition));
      LLVM_DEBUG({
        llvm::dbgs() << "Failed to create dot file: "
                     << llvm::toString(opMapPtr.takeError()) << "\n";
      });
      continue;
    }

    auto opMap = *opMapPtr;
    Graph graph =
        readDotFile(clMemoryAwarePartitioningIODir + "/partition-graph-" +
                        std::to_string(partitionIndex) + ".dot",
                    clMemoryAwarePartitioningIODir + "/partition-graph-" +
                        std::to_string(partitionIndex) + ".dot.nodemappings");

    uint64_t numPartitions = std::min(numPartitionsFromAttr[partitionIndex],
                                      (int64_t)partition.ops.size());

    RecursivePartitioner partitioner(
        graph, numPartitions, clMemoryAwarePartitioningConfig.clusteringMethod,
        clMemoryAwarePartitioningConfig.maxClusteringLevel, 50 * numPartitions,
        clMemoryAwarePartitioningConfig.minClusteringVertexRatio,
        clMemoryAwarePartitioningConfig.bisectionMethod,
        clMemoryAwarePartitioningConfig.maxImbalance,
        clMemoryAwarePartitioningConfig.refinementMethod,
        clMemoryAwarePartitioningConfig.maxRefinementPasses);

    auto [partitionInfo, cutSize] = partitioner.run();

    std::filesystem::remove(clMemoryAwarePartitioningIODir +
                            "/partition-graph-" +
                            std::to_string(partitionIndex) + ".dot");
    std::filesystem::remove(
        clMemoryAwarePartitioningIODir + "/partition-graph-" +
        std::to_string(partitionIndex) + ".dot.nodemappings");

    LLVM_DEBUG({
      uint64_t maxPartSize =
          calculateMaxPartSize(partitionInfo, graph, numPartitions);
      double imbalance =
          std::abs(((double)maxPartSize / (double)graph.totalWeight) * 100 -
                   ((double)100 / (double)numPartitions));
      imbalance = round(imbalance * 10) / 10;

      llvm::dbgs() << "Cut size: " << cutSize << "\nImbalance: " << imbalance
                   << "\n";
      clMemoryAwarePartitioningConfig.print();
      for (int node = 0; node < partitionInfo.size(); node++) {
        llvm::dbgs() << "Node " << node << "\nOp: ";
        opMap[node]->dump();
        llvm::dbgs() << "In partition " << partitionInfo[node] << "\n\n\n";
      }
    });

    SmallVector<Partition> partitions =
        createPartitions(partitionInfo, opMap, graph.topologicalSort(),
                         graph.groupedTopSortPositions(partitionInfo),
                         partition.affinity, partitionIndex);

    for (auto &partition : partitions) {
      result.partitions.push_back(std::move(partition));
    }

    partitionIndex++;
  }

  return result;
}

} // namespace mlir::iree_compiler::IREE::Stream