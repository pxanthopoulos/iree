#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/Passes.h"

#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>

#define DEBUG_TYPE "iree-stream-memory-aware-partitioning-feedback-loop"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MEMORYAWAREPARTITIONINGFEEDBACKLOOPPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

static llvm::cl::opt<bool> clMemoryAwarePartitioningEnableLifetimeAnalysis(
    "iree-stream-memory-aware-partitioning-enable-lifetime-analysis",
    llvm::cl::desc("Enable lifetime analysis for memory aware partitioning"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clMemoryAwarePartitioningReducedSearchPipeline(
    "iree-stream-memory-aware-partitioning-reduced-search-pipeline",
    llvm::cl::desc("Run only the passes that determine the region sizes, on "
                   "the functions with partitioned regions, in the feedback "
                   "loop's search iterations"),
    llvm::cl::init(true));

namespace {

static bool isFunctionLike(Operation *op) {
  return isa<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>(op);
}

struct MemoryAwarePartitioningFeedbackLoopPass
    : public IREE::Stream::impl::MemoryAwarePartitioningFeedbackLoopPassBase<
          MemoryAwarePartitioningFeedbackLoopPass> {
  using IREE::Stream::impl::MemoryAwarePartitioningFeedbackLoopPassBase<
      MemoryAwarePartitioningFeedbackLoopPass>::
      MemoryAwarePartitioningFeedbackLoopPassBase;

  // Adds the passes that form the execution regions to the function-level
  // |passManager|.
  static void addSchedulePasses(OpPassManager &passManager,
                                bool enableMemoryAwarePartitioning) {
    ScheduleExecutionPassOptions scheduleExecutionPassOptions;
    scheduleExecutionPassOptions.enableMemoryAwarePartitioning =
        enableMemoryAwarePartitioning;
    // Combine async work into execution regions.
    passManager.addPass(IREE::Stream::createScheduleExecutionPass(
        scheduleExecutionPassOptions));
    // Group concurrently executable work into waves.
    passManager.addPass(IREE::Stream::createScheduleConcurrencyPass());
  }

  // Adds the passes of the full pipeline after the scheduling passes to the
  // module-level |passManager|.
  void addPostSchedulePasses(OpPassManager &passManager, bool firstPass,
                             int passNo) {
    IREE::Stream::TransformOptions transformOptions;
    transformOptions.initializationMode = initializationMode;
    transformOptions.optimizeBindings = optimizeBindings;
    transformOptions.dumpStatisticsFormat = dumpStatisticsFormat;
    transformOptions.dumpStatisticsFile = dumpStatisticsFile;

    if (clMemoryAwarePartitioningEnableLifetimeAnalysis) {
      CalculateLifetimesPassOptions calculateLifetimesPassOptions;
      calculateLifetimesPassOptions.fileSuffixId = passNo;
      passManager.addPass(IREE::Stream::createCalculateLifetimesPass(
          calculateLifetimesPassOptions));
    }

    // Analysis must run on the initial partitions produced by
    // Reference partitioning, so only on the first pass where subsequent
    // partitioning (memory-aware partitioning) is turned off.
    if (firstPass) {
      passManager.addPass(IREE::Stream::createAnalyzeExecutionRegionsPass());
    }

    // When synchronous initialization is requested we need to separate any
    // work behind a timepoint in the initializer from the consumers of that
    // timepoint.
    if (transformOptions.initializationMode ==
        IREE::Stream::InitializationMode::Synchronous) {
      passManager.addPass(IREE::Stream::createSyncInitializersPass());
    }

    // Materialize timepoints across the entire module. This simplifies
    // scheduling of the timeline as we can shake the IR and see what
    // timepoints we still have left.
    passManager.addPass(IREE::Stream::createPropagateTimepointsPass());

    // Expand builtins to dispatches. This may introduce new executables.
    // We do this after scheduling so that we preserve the semantics of the
    // ops for partitioning/placement before turning them into opaque
    // dispatches.
    passManager.addPass(IREE::Stream::createMaterializeBuiltinsPass());

    // TODO(benvanik): outline streams (ala dispatch regions). Note that we
    // may want to do this earlier to enable better deduplication but that
    // makes the above passes trickier. Outlining may be more like "find
    // chunks of streams useful to move into secondary command buffers."

    buildStreamCleanupPassPipeline(passManager, transformOptions);

    // Everything must now be in stream.async.* form.
    passManager.addPass(IREE::Stream::createVerifyLoweringToAsyncPass());

    // Schedule fine-grained allocations and insert placeholders for
    // larger/longer lifetime allocations.
    passManager.addPass(IREE::Stream::createScheduleAllocationPass());

    FunctionLikeNest(passManager)
        // Allocate backing storage for fused constant resources.
        // This expands packed constants into explicit forms with partitioned
        // storage buffers and upload logic.
        .addPass(IREE::Stream::createPackConstantsPass)

        // Layout packed slices to emit the arithmetic required for all
        // resource offsets. This enables us to propagate the subviews across
        // the program below.
        .addPass(IREE::Stream::createLayoutSlicesPass)

        // Apply canonicalization patterns to clean up subview ops prior to
        // propagating subranges.
        .addPass(mlir::createCanonicalizerPass);

    // Check that all execution regions have transient slabs with size less
    // than the memory limit
    passManager.addPass(IREE::Stream::createCheckPartitionMemoryLimitPass());
  }

  // Runs the full pipeline on the whole module.
  LogicalResult runFullPipeline(ModuleOp moduleOp, bool firstPass, int passNo) {
    // For the first pass, memory-aware partitioning is turned off to see
    // worst case memory usage. Also, the first pass is
    // used to gather info about initial partitions.
    OpPassManager passManager;
    addSchedulePasses(passManager.nest<func::FuncOp>(), !firstPass);
    addSchedulePasses(passManager.nest<IREE::Util::InitializerOp>(),
                      !firstPass);
    addSchedulePasses(passManager.nest<IREE::Util::FuncOp>(), !firstPass);
    addPostSchedulePasses(passManager, firstPass, passNo);
    return runPipeline(passManager, moduleOp);
  }

  // Runs the passes that |addPasses| adds on each op of |ops|.
  LogicalResult runOnEach(ArrayRef<Operation *> ops,
                          function_ref<void(OpPassManager &)> addPasses) {
    for (Operation *op : ops) {
      OpPassManager passManager(op->getName());
      addPasses(passManager);
      if (failed(runPipeline(passManager, op)))
        return failure();
    }
    return success();
  }

  static void addMemoryAwareSchedulePasses(OpPassManager &passManager) {
    addSchedulePasses(passManager, /*enableMemoryAwarePartitioning=*/true);
  }
  static void addAllocationPasses(OpPassManager &passManager) {
    passManager.addPass(IREE::Stream::createScheduleAllocationPass());
  }
  static void addPackPasses(OpPassManager &passManager) {
    passManager.addPass(IREE::Stream::createPackConstantsPass());
    passManager.addPass(IREE::Stream::createLayoutSlicesPass());
  }
  static void addCheckPasses(OpPassManager &passManager) {
    passManager.addPass(IREE::Stream::createCheckPartitionMemoryLimitPass());
  }

  // Runs a search iteration on |searchOps|, the functions with partitioned
  // regions. If the region sizes fit, completes the full pipeline, whose
  // memory limit check decides, and returns its time in |finalSeconds|.
  LogicalResult runSearchIteration(ModuleOp moduleOp,
                                   ArrayRef<Operation *> searchOps, int passNo,
                                   std::optional<double> &finalSeconds) {
    // Partition the search functions; dagP and CP-SAT run here
    if (failed(runOnEach(searchOps, addMemoryAwareSchedulePasses)))
      return failure();
    OwningOpRef<ModuleOp> scheduledState = moduleOp.clone();

    // Measure the region sizes with only the passes that determine them
    Operation *module = moduleOp;
    if (failed(runOnEach(module, addAllocationPasses)) ||
        failed(runOnEach(searchOps, addPackPasses)) ||
        failed(runOnEach(module, addCheckPasses)))
      return failure();
    auto partitioningInfoAttr =
        moduleOp->getAttrOfType<StringAttr>("iree.stream.partitioning.info");
    if (!partitioningInfoAttr || partitioningInfoAttr.getValue() != "pass") {
      return success();
    }

    // The sizes fit: complete the full pipeline from the partitioned state
    auto finalStart = std::chrono::steady_clock::now();
    // Restore the partitioned state, which keeps the CP-SAT schedule, and the
    // partitioning info this iteration started with
    for (unsigned i = 0; i < moduleOp->getNumRegions(); ++i) {
      moduleOp->getRegion(i).takeBody(scheduledState.get()->getRegion(i));
    }
    moduleOp->setAttrs(scheduledState.get()->getAttrs());
    // Schedule the other functions, then run the rest of the full pipeline
    SmallVector<Operation *> otherOps;
    for (Operation &op : moduleOp.getOps()) {
      if (isFunctionLike(&op) && !llvm::is_contained(searchOps, &op))
        otherOps.push_back(&op);
    }
    if (failed(runOnEach(otherOps, addMemoryAwareSchedulePasses)) ||
        failed(runOnEach(module, [&](OpPassManager &passManager) {
          addPostSchedulePasses(passManager, /*firstPass=*/false, passNo);
        })))
      return failure();
    finalSeconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - finalStart)
                       .count();
    return success();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    auto loopStart = std::chrono::steady_clock::now();
    bool firstPass = true;
    OwningOpRef<ModuleOp> initialState = moduleOp.clone();

    // Build the dagP graph of each region once, in the first iteration that
    // partitions it, and reuse it in the later iterations
    beginMemoryAwarePartitioningGraphCache(moduleOp);
    auto endGraphCache = llvm::make_scope_exit(
        [&]() { endMemoryAwarePartitioningGraphCache(moduleOp); });

    // The functions with partitioned regions, which the search iterations
    // run on; if empty, every iteration runs the full pipeline.
    SmallVector<std::string> searchFunctions;

    int passNo = 1;

    while (1) {
      // Copy previous state and re-run the pipeline (not needed on the first
      // pass)
      auto restoreStart = std::chrono::steady_clock::now();
      if (!firstPass) {
        unsigned numRegions = moduleOp->getNumRegions();
        for (unsigned i = 0; i < numRegions; ++i) {
          moduleOp->getRegion(i).takeBody(initialState.get()->getRegion(i));
        }
        auto partitioningInfo =
            moduleOp->getAttr("iree.stream.partitioning.info");
        moduleOp->setAttrs(initialState.get()->getAttrs());
        moduleOp->setAttr("iree.stream.partitioning.info", partitioningInfo);
        initialState = moduleOp.clone();
      }
      std::chrono::duration<double> restoreSeconds =
          std::chrono::steady_clock::now() - restoreStart;

      LLVM_DEBUG({
        llvm::dbgs() << "PASS #" << passNo << "\n\n\n";
        llvm::dbgs() << "Initial state:\n\n";
        auto partitioningInfoAttr = moduleOp->getAttrOfType<StringAttr>(
            "iree.stream.partitioning.info");
        if (!partitioningInfoAttr) {
          llvm::dbgs() << "No partitioning info attribute found\n";
        } else {
          partitioningInfoAttr.dump();
        }
        llvm::dbgs()
            << "\n\n==========================================================="
               "===================================================\n\n\n";
      });

      // Look the search functions up again: the restore replaced all ops
      SmallVector<Operation *> searchOps;
      for (const std::string &name : searchFunctions) {
        searchOps.push_back(SymbolTable::lookupSymbolIn(moduleOp, name));
      }
      // Without search functions (iteration 1, or no restriction), run the
      // full pipeline on the module; otherwise, a search iteration on them
      std::optional<double> finalSeconds;
      auto pipelineStart = std::chrono::steady_clock::now();
      LogicalResult result =
          searchOps.empty()
              ? runFullPipeline(moduleOp, firstPass, passNo)
              : runSearchIteration(moduleOp, searchOps, passNo, finalSeconds);
      if (failed(result)) {
        return signalPassFailure();
      }
      std::chrono::duration<double> pipelineSeconds =
          std::chrono::steady_clock::now() - pipelineStart;
      // The full pipeline that completes a search iteration is printed apart
      llvm::errs() << "Feedback loop iteration " << passNo << ": restore "
                   << llvm::format("%.6f", restoreSeconds.count())
                   << " seconds, pipeline "
                   << llvm::format("%.6f", pipelineSeconds.count() -
                                               finalSeconds.value_or(0.0))
                   << " seconds\n";
      if (finalSeconds) {
        llvm::errs() << "Feedback loop final pipeline: "
                     << llvm::format("%.6f", *finalSeconds) << " seconds\n";
      }

      LLVM_DEBUG({
        llvm::dbgs()
            << "\n\n==========================================================="
               "===================================================\n\n\n";
        llvm::dbgs() << "PASS #" << passNo << "\n\n\n";
        llvm::dbgs() << "After pass:\n\n";
        auto partitioningInfoAttr = moduleOp->getAttrOfType<StringAttr>(
            "iree.stream.partitioning.info");
        if (!partitioningInfoAttr) {
          llvm::dbgs() << "No partitioning info attribute found\n";
        } else {
          partitioningInfoAttr.dump();
        }
        llvm::dbgs()
            << "\n\n==========================================================="
               "===================================================\n\n\n";
      });

      const char *env_var = std::getenv("NUM_PARTITIONS");

      // AnalyzeExecutionRegions records the functions with partitioned
      // regions; restrict the search iterations to them if all of them can be
      // looked up by name after a restore
      if (firstPass) {
        auto functionsAttr = moduleOp->getAttrOfType<ArrayAttr>(
            "iree.stream.partitioning.functions");
        moduleOp->removeAttr("iree.stream.partitioning.functions");
        if (functionsAttr && clMemoryAwarePartitioningReducedSearchPipeline &&
            !clMemoryAwarePartitioningEnableLifetimeAnalysis && !env_var) {
          for (auto name : functionsAttr.getAsValueRange<StringAttr>()) {
            Operation *op = initialState.get().lookupSymbol(name);
            if (!op || !isFunctionLike(op)) {
              searchFunctions.clear();
              break;
            }
            searchFunctions.push_back(name.str());
          }
        }
      }

      // If memory limit is not breached, break the loop
      if (auto partitioningInfoAttr = moduleOp->getAttrOfType<StringAttr>(
              "iree.stream.partitioning.info")) {
        if (partitioningInfoAttr.getValue() == "pass") {
          break;
        }
      }

      if (env_var && !firstPass) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Environment variable NUM_PARTITIONS is set to "
                   << env_var << ", stopping loop\n");
        break;
      }

      // Update flag
      if (firstPass) {
        firstPass = false;
      }

      passNo++;
    }

    std::chrono::duration<double> loopSeconds =
        std::chrono::steady_clock::now() - loopStart;
    llvm::errs() << "Feedback loop total: "
                 << llvm::format("%.6f", loopSeconds.count()) << " seconds, "
                 << passNo << " iterations\n";
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
