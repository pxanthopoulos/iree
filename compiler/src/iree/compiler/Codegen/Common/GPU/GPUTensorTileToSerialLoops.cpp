// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace {
struct GPUTensorTensorTileToSerialLoopsPass final
    : public GPUTensorTileToSerialLoopsBase<
          GPUTensorTensorTileToSerialLoopsPass> {
public:
  GPUTensorTensorTileToSerialLoopsPass(bool optionsCollapseLoops = false) {
    coalesceLoops = optionsCollapseLoops;
  }
  GPUTensorTensorTileToSerialLoopsPass(
      const GPUTensorTensorTileToSerialLoopsPass &pass) {
    coalesceLoops = pass.coalesceLoops;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    // Tile reductions based on the annotated tiling configuration.
    if (failed(tileReductionToSerialLoops(getOperation(),
                                          /*fuseInputProducer=*/true,
                                          coalesceLoops))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUTensorTileToSerialLoops(bool coalesceLoops) {
  return std::make_unique<GPUTensorTensorTileToSerialLoopsPass>(coalesceLoops);
}

} // namespace mlir::iree_compiler
