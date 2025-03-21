#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-stream-memory-aware-partitioning-feedback-loop"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MEMORYAWAREPARTITIONINGFEEDBACKLOOPPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

static llvm::cl::opt<bool> clEnableMemoryAwarePartitioning(
    "iree-stream-enable-memory-aware-partitioning",
    llvm::cl::desc("Enable memory aware partitioning"), llvm::cl::init(false));

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

    while (1) {
      OpPassManager passManager;

      IREE::Stream::TransformOptions transformOptions;
      transformOptions.initializationMode = initializationMode;
      transformOptions.optimizeBindings = optimizeBindings;
      transformOptions.dumpStatisticsFormat = dumpStatisticsFormat;
      transformOptions.dumpStatisticsFile = dumpStatisticsFile;

      ScheduleExecutionPassOptions scheduleExecutionPassOptions;

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

      buildStreamCleanupPassPipeline(passManager, transformOptions);

      // Everything must now be in stream.async.* form.
      passManager.addPass(IREE::Stream::createVerifyLoweringToAsyncPass());

      // Schedule fine-grained allocations and insert placeholders for
      // larger/longer lifetime allocations.
      passManager.addPass(IREE::Stream::createScheduleAllocationPass());
      FunctionLikeNest(passManager)
          // TODO(benvanik): passes to convert alloc to alloca and thread
          // through streams. Ideally all transient allocs become stream-ordered
          // allocas. createPropagateTransientsPass()

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

      if (failed(runPipeline(passManager, moduleOp))) {
        return signalPassFailure();
      }
      if (!clEnableMemoryAwarePartitioning) {
        break;
      }
      if (firstPass) {
        firstPass = false;
      }
      break;
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream