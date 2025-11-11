#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-stream-memory-aware-partitioning-feedback-loop"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MEMORYAWAREPARTITIONINGFEEDBACKLOOPPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

namespace {

struct MemoryAwarePartitioningFeedbackLoopPass
    : public IREE::Stream::impl::MemoryAwarePartitioningFeedbackLoopPassBase<
          MemoryAwarePartitioningFeedbackLoopPass> {
  using IREE::Stream::impl::MemoryAwarePartitioningFeedbackLoopPassBase<
      MemoryAwarePartitioningFeedbackLoopPass>::
      MemoryAwarePartitioningFeedbackLoopPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    bool firstPass = true;
    auto initialState = moduleOp.clone();

    int passNo = 1;

    while (1) {
      // Copy previous state and re-run the pipeline (not needed on the first
      // pass)
      if (!firstPass) {
        unsigned numRegions = moduleOp->getNumRegions();
        for (unsigned i = 0; i < numRegions; ++i) {
          moduleOp->getRegion(i).takeBody(initialState->getRegion(i));
        }
        auto partitioningInfo =
            moduleOp->getAttr("iree.stream.partitioning.info");
        moduleOp->setAttrs(initialState->getAttrs());
        moduleOp->setAttr("iree.stream.partitioning.info", partitioningInfo);
        initialState = moduleOp.clone();
      }

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

      OpPassManager passManager;

      IREE::Stream::TransformOptions transformOptions;
      transformOptions.initializationMode = initializationMode;
      transformOptions.optimizeBindings = optimizeBindings;
      transformOptions.dumpStatisticsFormat = dumpStatisticsFormat;
      transformOptions.dumpStatisticsFile = dumpStatisticsFile;

      ScheduleExecutionPassOptions scheduleExecutionPassOptions;

      // For the first pass, turn off memory-aware partitioning to see worst
      // case memory usage (max concurrency). Also, the first pass is used to
      // gather info about initial partitions.
      if (firstPass) {
        scheduleExecutionPassOptions.enableMemoryAwarePartitioning = false;
      } else {
        scheduleExecutionPassOptions.enableMemoryAwarePartitioning = true;
      }

      FunctionLikeNest(passManager)
          // Combine async work into execution regions.
          .addPass([&]() {
            return IREE::Stream::createScheduleExecutionPass(
                scheduleExecutionPassOptions);
          })
          // Group concurrently executable work into waves.
          .addPass(IREE::Stream::createScheduleConcurrencyPass);

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

      if (failed(runPipeline(passManager, moduleOp))) {
        return signalPassFailure();
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
        llvm::dbgs()
            << "\n\n==========================================================="
               "===================================================\n\n\n";
      });

      // Update flag
      if (firstPass) {
        firstPass = false;
      }

      // If memory limit is not breached, break the loop
      if (auto partitioningInfoAttr = moduleOp->getAttrOfType<StringAttr>(
              "iree.stream.partitioning.info")) {
        if (partitioningInfoAttr.getValue() == "pass") {
          break;
        }
      }

      passNo++;
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream