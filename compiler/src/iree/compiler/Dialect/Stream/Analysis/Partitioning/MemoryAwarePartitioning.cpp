#include "Graph.h"
#include "RecursivePartitioner.h"
#include "Scheduling.h"
#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/Analysis/StaticSlabPacking.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/AsmState.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

#define DEBUG_TYPE "iree-stream-memory-aware-partitioning-partitioner"

namespace mlir::iree_compiler::IREE::Stream {

enum class InterRegionScheduler { Off, CPSAT, RPO };

static llvm::cl::opt<InterRegionScheduler> clInterRegionScheduler(
    "iree-stream-memory-aware-partitioning-inter-region-scheduler",
    llvm::cl::desc("How to order the final memory-aware partitions."),
    llvm::cl::values(clEnumValN(InterRegionScheduler::Off, "off",
                                "Their topological order (default)."),
                     clEnumValN(InterRegionScheduler::CPSAT, "cpsat",
                                "CP-SAT, warm-started with the RPO heuristic."),
                     clEnumValN(InterRegionScheduler::RPO, "rpo",
                                "The RPO heuristic alone.")),
    llvm::cl::init(InterRegionScheduler::Off));

bool isInterRegionSchedulingEnabled() {
  return clInterRegionScheduler != InterRegionScheduler::Off;
}

struct MemoryAwarePartitioningConfig {
  std::string clusteringMethod;
  uint64_t maxClusteringRounds;
  double clusteringVertexRatio;
  std::string bisectionMethod;
  double imbalanceRatio;
  std::string refinementMethod;
  uint64_t refinementPasses;
  bool enableParallel;
  uint64_t minSizeForParallel;
  uint64_t maxParallelDepth;

  MemoryAwarePartitioningConfig(
      const char *V = "HYB,20,0.9,UNDIRBOTH,1.1,MIXED,10,1,100,10") {
    clusteringMethod = "HYB";
    maxClusteringRounds = 20;
    clusteringVertexRatio = 0.9;
    bisectionMethod = "UNDIRBOTH";
    imbalanceRatio = 1.1;
    refinementMethod = "MIXED";
    refinementPasses = 10;
    enableParallel = true;
    minSizeForParallel = 100;
    maxParallelDepth = 10;
  }

  void print() const {
    LLVM_DEBUG({
      llvm::dbgs() << "Memory-Aware Partitioning Configuration:\n"
                   << "  Clustering Method: ";

      if (clusteringMethod == "FORB") {
        llvm::dbgs() << "FORB (Forbidden edges)";
      } else if (clusteringMethod == "CYC") {
        llvm::dbgs() << "CYC (Cycle detection)";
      } else if (clusteringMethod == "HYB") {
        llvm::dbgs() << "HYB (Hybrid approach)";
      }

      llvm::dbgs() << "\n  Max Clustering Rounds: " << maxClusteringRounds
                   << "\n  Min Clustering Vertex Ratio: "
                   << clusteringVertexRatio << "\n  Bisection Method: ";

      if (bisectionMethod == "GGG") {
        llvm::dbgs() << "GGG (Greedy directed graph growing)";
      } else if (bisectionMethod == "UNDIRMETIS") {
        llvm::dbgs() << "UNDIRMETIS (Undirected METIS)";
      } else if (bisectionMethod == "UNDIRSCOTCH") {
        llvm::dbgs() << "UNDIRSCOTCH (Undirected Scotch)";
      } else if (bisectionMethod == "UNDIRBOTH") {
        llvm::dbgs() << "UNDIRBOTH (Try both METIS and Scotch)";
      }

      llvm::dbgs() << "\n  Max Imbalance Ratio: " << imbalanceRatio
                   << "\n  Refinement Method: ";

      if (refinementMethod == "BOUNDARYFM") {
        llvm::dbgs() << "BOUNDARYFM (Boundary FM adaptation)";
      } else if (refinementMethod == "BOUNDARYKL") {
        llvm::dbgs() << "BOUNDARYKL (Boundary KL adaptation)";
      } else if (refinementMethod == "MIXED") {
        llvm::dbgs() << "MIXED (KL + FM)";
      }

      llvm::dbgs() << "\n  Max Refinement Passes: " << refinementPasses << "\n";
      llvm::dbgs() << "  Parallel Execution: "
                   << (enableParallel ? "Enabled" : "Disabled") << "\n";
      llvm::dbgs() << "  Min Size for Parallel: " << minSizeForParallel << "\n";
      llvm::dbgs() << "  Max Parallel Depth: " << maxParallelDepth << "\n";
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

    if (Parts.size() != 10)
      return O.error("expected 10 comma-separated values");

    if (Parts[0].compare("FORB") == 0) {
      V.clusteringMethod = "FORB";
    } else if (Parts[0].compare("CYC") == 0) {
      V.clusteringMethod = "CYC";
    } else if (Parts[0].compare("HYB") == 0) {
      V.clusteringMethod = "HYB";
    } else {
      return O.error("invalid clustering method (expected: FORB|CYC|HYB)");
    }

    if (Parts[1].getAsInteger(10, V.maxClusteringRounds))
      return O.error("invalid max clustering rounds");

    double Ratio;
    if (Parts[2].getAsDouble(Ratio) || Ratio <= 0.0 || Ratio >= 1.0)
      return O.error("min clustering vertex ratio must be between 0 and 1");
    V.clusteringVertexRatio = Ratio;

    if (Parts[3].compare("GGG") == 0) {
      V.bisectionMethod = "GGG";
    } else if (Parts[3].compare("UNDIRSCOTCH") == 0) {
      V.bisectionMethod = "UNDIRSCOTCH";
    } else if (Parts[3].compare("UNDIRMETIS") == 0) {
      V.bisectionMethod = "UNDIRMETIS";
    } else if (Parts[3].compare("UNDIRBOTH") == 0) {
      V.bisectionMethod = "UNDIRBOTH";
    } else {
      return O.error("invalid bisection method (expected: "
                     "GGG|UNDIRSCOTCH|UNDIRMETIS|UNDIRBOTH)");
    }

    double Imbalance;
    if (Parts[4].getAsDouble(Imbalance) || Imbalance <= 1.0)
      return O.error("max imbalance ratio must be greater than 1");
    V.imbalanceRatio = Imbalance;

    if (Parts[5].compare("BOUNDARYFM") == 0) {
      V.refinementMethod = "BOUNDARYFM";
    } else if (Parts[5].compare("BOUNDARYKL") == 0) {
      V.refinementMethod = "BOUNDARYKL";
    } else if (Parts[5].compare("MIXED") == 0) {
      V.refinementMethod = "MIXED";
    } else {
      return O.error(
          "invalid refinement method (expected: BOUNDARYFM|BOUNDARYKL|MIXED)");
    }

    if (Parts[6].getAsInteger(10, V.refinementPasses))
      return O.error("invalid max refinement passes");

    if (Parts[7].getAsInteger(10, V.enableParallel))
      return O.error("invalid parallel execution flag");

    if (Parts[8].getAsInteger(10, V.minSizeForParallel))
      return O.error("invalid min size for parallel");

    if (Parts[9].getAsInteger(10, V.maxParallelDepth))
      return O.error("invalid max parallel depth");

    return false;
  }
};

static llvm::cl::opt<MemoryAwarePartitioningConfig, false,
                     MemoryAwarePartitioningConfigParser>
    clMemoryAwarePartitioningConfig(
        "iree-stream-memory-aware-partitioning-config",
        llvm::cl::desc(
            "--iree-stream-memory-aware-partitioning-config=cm,cl,vr,bm,i,"
            "rm,rp,ep,ms,md\n"
            "Comma-separated list of configuration variables for the "
            "partitioning phase:\n"
            "  cm = clusteringMethod (FORB|CYC|HYB)\n"
            "  cl = maxClusteringRounds (int)\n"
            "  vr = clusteringVertexRatio (float, 0.0 < vr < 1.0)\n"
            "  bm = bisectionMethod (GGG|UNDIRSCOTCH|UNDIRMETIS|UNDIRBOTH)\n"
            "  i  = imbalanceRatio (float, 1.0 < i)\n"
            "  rm = refinementMethod (BOUNDARYFM|BOUNDARYKL|MIXED)\n"
            "  rp = refinementPasses (int)\n"
            "  ep = enableParallel (0|1)\n"
            "  ms = minSizeForParallel (int)\n"
            "  md = maxParallelDepth (int)\n"
            "Example: "
            "--iree-stream-memory-aware-partitioning-config=HYB,20,0.9,BOTH,1."
            "1,MIX,10,1,100,10"),
        llvm::cl::init("HYB,20,0.9,BOTH,1.1,MIX,10,1,100,10"));

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

llvm::Expected<llvm::SmallVector<std::pair<int64_t, bool>>>
extractNumPartitionsFromAttr(Block *block) {
  llvm::SmallVector<std::pair<int64_t, bool>> result;

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

    result.emplace_back(second, third - first == 1);
  }

  return result;
}

bool checkMoreDispatches(Partition partition) {
  uint64_t totalDispatches = 0;
  uint64_t realOps = 0;
  for (const auto &op : partition.ops) {
    if (isa<IREE::Stream::StreamableOpInterface>(op) &&
        !op->hasTrait<OpTrait::ConstantLike>() &&
        !isa<IREE::Util::GlobalStoreOpInterface>(op) &&
        !dyn_cast<IREE::Stream::AsyncConcurrentOp>(op) &&
        !dyn_cast<IREE::Stream::AsyncUpdateOp>(op)) {
      realOps++;
    }
    auto dispatchOp = llvm::dyn_cast<IREE::Stream::AsyncDispatchOp>(op);
    if (dispatchOp)
      totalDispatches++;
  }
  if (totalDispatches > realOps / 5)
    return true;
  return false;
}

// Returns the size of the resource |value| defined by |op|, or 50 if it is not
// a static size (partitioning only works for static sizes)
static int64_t getStaticResourceSize(Operation *op, Value value) {
  int64_t actualSize = 50;
  auto sizeAwareOp = dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(op);
  if (!sizeAwareOp)
    return actualSize;
  auto resultSize = sizeAwareOp.getResultSizeFromValue(value);
  auto valueTypedAttr =
      resultSize.getDefiningOp()->getAttrOfType<TypedAttr>("value");
  if (resultSize.getType().isIntOrIndex() && valueTypedAttr) {
    if (auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueTypedAttr))
      actualSize = integerAttr.getInt();
  }
  return actualSize;
}

// Node ids are the positions of the ops in block order. The scheduler relies
// on the ids being topological.
static DenseMap<unsigned, Operation *> buildOpMap(Partition &partition) {
  DenseMap<unsigned, Operation *> opMap;
  for (const auto [opIndex, op] :
       llvm::enumerate(llvm::reverse(partition.ops))) {
    opMap[opIndex] = op;
  }
  return opMap;
}

// Builds the dagP graph of |partition|: a node per op, weighted by the size of
// the resources it defines, and an edge per producer-consumer pair, weighted by
// the size of the resources passed along it.
static dag_partitioning::core::Graph buildGraph(Partition &partition) {
  dag_partitioning::core::Graph graph(partition.ops.size());

  DenseMap<Operation *, unsigned> inverseOpMap;
  for (const auto [opIndex, op] :
       llvm::enumerate(llvm::reverse(partition.ops))) {
    inverseOpMap[op] = opIndex;
    int64_t opWeight = 0;
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(*op);

    for (Value result : op->getResults()) {
      if (!llvm::isa<IREE::Stream::ResourceType>(result.getType())) {
        continue;
      }
      if (tiedOp && tiedOp.getTiedResultOperand(result)) {
        continue;
      }
      opWeight += getStaticResourceSize(op, result);
    }
    graph.addNode(opIndex, static_cast<uint64_t>(opWeight));
  }

  // Edges are added sorted by (from, to), the order the dot file had
  std::map<std::pair<unsigned, unsigned>, int64_t> edgeWeights;
  for (const auto op : llvm::reverse(partition.ops)) {
    for (Value value : op->getOperands()) {
      Operation *definingOp = value.getDefiningOp();
      if (partition.ops.contains(definingOp)) {
        edgeWeights[{inverseOpMap[definingOp], inverseOpMap[op]}] +=
            getStaticResourceSize(definingOp, value);
      }
    }
  }
  for (const auto &[edge, weight] : edgeWeights) {
    graph.addEdge(edge.first, edge.second, static_cast<uint64_t>(weight));
  }

  return graph;
}

namespace {

// dagP graphs of the regions of the modules between
// beginMemoryAwarePartitioningGraphCache and
// endMemoryAwarePartitioningGraphCache. A region is identified by its module,
// the position of its callable in the module, the position of its block in the
// callable and its index among the regions memoryAwarePartition handles in the
// block. Callables are partitioned in parallel, hence the mutex.
struct GraphCache {
  using Key = std::tuple<Operation *, size_t, size_t, int64_t>;
  std::mutex mutex;
  llvm::DenseSet<Operation *> activeModules;
  std::map<Key, std::shared_ptr<const dag_partitioning::core::Graph>> graphs;
};

GraphCache &getGraphCache() {
  static GraphCache cache;
  return cache;
}

} // namespace

void beginMemoryAwarePartitioningGraphCache(Operation *moduleOp) {
  auto &cache = getGraphCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  cache.activeModules.insert(moduleOp);
}

void endMemoryAwarePartitioningGraphCache(Operation *moduleOp) {
  auto &cache = getGraphCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  cache.activeModules.erase(moduleOp);
  for (auto it = cache.graphs.begin(); it != cache.graphs.end();) {
    if (std::get<0>(it->first) == moduleOp) {
      it = cache.graphs.erase(it);
    } else {
      ++it;
    }
  }
}

// Returns the cache key of the regions of |block|, or std::nullopt if the
// cache is not active for its module
static std::optional<GraphCache::Key> getGraphCacheKey(Block *block) {
  Operation *callableOp = block->getParentOp();
  Operation *moduleOp = callableOp->getParentOp();
  if (!moduleOp || !callableOp->getBlock())
    return std::nullopt;
  {
    auto &cache = getGraphCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!cache.activeModules.contains(moduleOp))
      return std::nullopt;
  }
  auto &ops = callableOp->getBlock()->getOperations();
  size_t callableIndex =
      std::distance(ops.begin(), Block::iterator(callableOp));
  auto &blocks = block->getParent()->getBlocks();
  size_t blockIndex = std::distance(blocks.begin(), Region::iterator(block));
  return GraphCache::Key{moduleOp, callableIndex, blockIndex, 0};
}

// Returns the dagP graph of |partition|, from the cache when |key| is set and
// the graph is cached already
static std::shared_ptr<const dag_partitioning::core::Graph>
getGraph(Partition &partition, std::optional<GraphCache::Key> key) {
  auto &cache = getGraphCache();
  if (key) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.graphs.find(*key);
    if (it != cache.graphs.end() && it->second->size == partition.ops.size())
      return it->second;
  }
  auto graph = std::make_shared<const dag_partitioning::core::Graph>(
      buildGraph(partition));
  if (key) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.graphs[*key] = graph;
  }
  return graph;
}

uint64_t calculateMaxPartSize(const std::vector<uint64_t> &partitionInfo,
                              const dag_partitioning::core::Graph &graph,
                              uint64_t numPartitions) {
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
               const std::vector<uint64_t> &schedule) {
  struct PartitionData {
    SmallVector<std::pair<Operation *, uint64_t>> ops;
    uint64_t schedulePosition = 0;
  };
  DenseMap<uint64_t, PartitionData> partitionData;

  std::vector<uint64_t> topSortPositions(partitionInfo.size());
  for (unsigned i = 0; i < topSortPositions.size(); ++i) {
    topSortPositions[topSort[i]] = i;
  }

  std::vector<uint64_t> schedulePositions(schedule.size());
  for (unsigned i = 0; i < schedule.size(); ++i) {
    schedulePositions[schedule[i]] = i;
  }

  for (unsigned i = 0; i < partitionInfo.size(); i++) {
    if (auto op = opMap.lookup(i)) {
      uint64_t partId = partitionInfo[i];
      auto &data = partitionData[partId];
      data.ops.push_back({op, topSortPositions[i]});
      data.schedulePosition = schedulePositions[partId];
    }
  }

  SmallVector<uint64_t> sortedPartitions;
  for (const auto &entry : partitionData)
    sortedPartitions.push_back(entry.first);

  llvm::sort(sortedPartitions, [&](uint64_t a, uint64_t b) {
    return partitionData[a].schedulePosition <
           partitionData[b].schedulePosition;
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

  llvm::SmallVector<std::pair<int64_t, bool>> numPartitionsFromAttr;
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

  auto cacheKey = getGraphCacheKey(block);

  int64_t partitionIndex = 0;
  for (auto &partition : initialPartitions.partitions) {
    if (!checkMoreDispatches(partition)) {
      result.partitions.push_back(std::move(partition));
      continue;
    }

    // Graph building covers building the graph, or looking it up in the
    // cache, and rebuilding the node id -> op map (the restored module has new
    // ops)
    auto graphStart = std::chrono::steady_clock::now();
    if (cacheKey)
      std::get<3>(*cacheKey) = partitionIndex;
    auto graphPtr = getGraph(partition, cacheKey);
    const dag_partitioning::core::Graph &graph = *graphPtr;
    auto opMap = buildOpMap(partition);
    std::chrono::duration<double> graphSeconds =
        std::chrono::steady_clock::now() - graphStart;

    uint64_t numPartitions =
        std::min(numPartitionsFromAttr[partitionIndex].first,
                 (int64_t)partition.ops.size());

    const char *env_var = std::getenv("NUM_PARTITIONS");
    if (env_var) {
      numPartitions = std::stoull(env_var);
      LLVM_DEBUG(llvm::dbgs() << "Environment variable NUM_PARTITIONS is set. "
                                 "Overriding numPartitions to "
                              << numPartitions << "\n");
    }

    dag_partitioning::driver::RecursivePartitioner partitioner(
        graph, numPartitions, clMemoryAwarePartitioningConfig.clusteringMethod,
        clMemoryAwarePartitioningConfig.maxClusteringRounds, 50 * numPartitions,
        clMemoryAwarePartitioningConfig.clusteringVertexRatio,
        clMemoryAwarePartitioningConfig.bisectionMethod,
        clMemoryAwarePartitioningConfig.imbalanceRatio,
        clMemoryAwarePartitioningConfig.refinementMethod,
        clMemoryAwarePartitioningConfig.refinementPasses,
        clMemoryAwarePartitioningConfig.enableParallel,
        clMemoryAwarePartitioningConfig.minSizeForParallel,
        clMemoryAwarePartitioningConfig.maxParallelDepth);

    auto partitioningStart = std::chrono::steady_clock::now();
    auto [partitionMapping, cutSize] = partitioner.run();
    std::chrono::duration<double> partitioningSeconds =
        std::chrono::steady_clock::now() - partitioningStart;

    std::vector<uint64_t> schedule;
    if (isInterRegionSchedulingEnabled() &&
        numPartitionsFromAttr[partitionIndex].second) {
      // Weigh partitions with the packer LayoutSlices lays out their
      // transient slabs with, and the alignments ScheduleAllocation uses
      auto resourceConfig =
          IREE::Stream::ResourceConfigAttr::lookup(partition.ops.front());
      int64_t offsetAlignment = resourceConfig.getMinBufferOffsetAlignment();
      int64_t rangeAlignment = resourceConfig.getMinBufferRangeAlignment();
      Location loc = partition.ops.front()->getLoc();
      auto packer =
          [offsetAlignment, rangeAlignment, loc](
              std::vector<dag_partitioning::scheduling::utils::Tensor> &tensors)
          -> uint64_t {
        SmallVector<IREE::Stream::StaticSlabSlice> slices;
        slices.reserve(tensors.size());
        for (auto &tensor : tensors) {
          slices.push_back({static_cast<int64_t>(tensor.birth),
                            static_cast<int64_t>(tensor.death),
                            static_cast<int64_t>(tensor.size)});
        }
        FailureOr<IREE::Stream::StaticSlabLayout> layout =
            IREE::Stream::packStaticSlab(slices, offsetAlignment,
                                         rangeAlignment, loc);
        if (failed(layout)) {
          // idealloc was asked for but is not built in; LayoutSlices reports
          // it and fails the compilation
          return dag_partitioning::scheduling::utils::packBestFit(
              tensors, offsetAlignment, rangeAlignment);
        }
        for (auto [tensor, offset, size] :
             llvm::zip_equal(tensors, layout->offsets, layout->sizes)) {
          tensor.offset = offset;
          tensor.size = size;
        }
        return layout->size;
      };
      dag_partitioning::scheduling::Scheduler scheduler(
          graph, partitionMapping, true, false, offsetAlignment, rangeAlignment,
          packer);
      auto result = scheduler.run(
          3600, 32,
          clInterRegionScheduler == InterRegionScheduler::RPO
              ? dag_partitioning::scheduling::Solver::RPO
              : dag_partitioning::scheduling::Solver::CPSAT);
      schedule = result.first;
    } else {
      schedule = graph.partitionTopologicalOrder(partitionMapping);
    }

    auto creationStart = std::chrono::steady_clock::now();
    SmallVector<Partition> partitions =
        createPartitions(partitionMapping, opMap, graph.topologicalSort(),
                         schedule, partition.affinity, partitionIndex);
    std::chrono::duration<double> creationSeconds =
        std::chrono::steady_clock::now() - creationStart;

    llvm::errs() << "Memory-aware partitioning region " << partitionIndex
                 << ": " << numPartitions << " parts, graph building "
                 << llvm::format("%.6f", graphSeconds.count())
                 << " seconds, partitioning "
                 << llvm::format("%.6f", partitioningSeconds.count())
                 << " seconds, partition creation "
                 << llvm::format("%.6f", creationSeconds.count())
                 << " seconds\n";

    for (auto &partition : partitions) {
      result.partitions.push_back(std::move(partition));
    }

    partitionIndex++;
  }

  return result;
}

} // namespace mlir::iree_compiler::IREE::Stream