// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/StaticSlabPacking.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/IntegerSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-layout-slices"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_LAYOUTSLICESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

using Slice = IREE::Stream::ResourcePackOp::Slice;

// Lays out the static slices of |packOp| with packStaticSlab, the packer
// memory-aware partitioning also weighs partitions with. Fails (with an error
// on |packOp|) only if idealloc was asked for but is not built in.
static FailureOr<StaticSlabLayout>
layoutStaticSlices(IREE::Stream::ResourcePackOp packOp, ArrayRef<Slice> slices,
                   IREE::Stream::ResourceConfigAttr resourceConfig) {
  SmallVector<StaticSlabSlice> slabSlices;
  slabSlices.reserve(slices.size());
  for (auto &slice : slices) {
    slabSlices.push_back(
        {slice.lifetimeStart, slice.lifetimeEnd,
         cast<arith::ConstantIndexOp>(slice.dynamicSize.getDefiningOp())
             .value()});
  }
  FailureOr<StaticSlabLayout> layout = packStaticSlab(
      slabSlices, resourceConfig.getMinBufferOffsetAlignment(),
      resourceConfig.getMinBufferRangeAlignment(), packOp.getLoc());
  if (failed(layout)) {
    packOp.emitError()
        << "--iree-stream-layout-slices-static-packer=idealloc/best needs a "
           "compiler built with -DIREE_ENABLE_IDEALLOC=ON";
  }
  return layout;
}

// Updates the packed offset SSA values of |slices| to start at |baseOffset|.
// Returns |baseOffset| + the size of the static slab.
static Value emitStaticLayout(IREE::Stream::ResourcePackOp packOp,
                              Value baseOffset, ArrayRef<Slice> slices,
                              const StaticSlabLayout &layout,
                              IndexSet &indexSet, OpBuilder &builder) {
  for (auto [slice, offset] : llvm::zip_equal(slices, layout.offsets)) {
    Value packedOffset = slice.packedOffset;
    packedOffset.replaceAllUsesWith(builder.createOrFold<arith::AddIOp>(
        packOp.getLoc(), baseOffset, indexSet.get(offset)));
  }
  return builder.createOrFold<arith::AddIOp>(packOp.getLoc(), baseOffset,
                                             indexSet.get(layout.size));
}

// Packs a set of dynamically-sized slices based on the structural information
// in the IR. Only slices that have the exact same size will be allowed to
// alias.
//
// We can improve this if we know which sizes are larger/smaller relative to
// others as then we can do things like reuse an allocation bucket with
// non-overlapping lifetimes if the thing we are trying to pack in it is
// definitely <=. If we end up knowing that certain sizes are less than some
// absolute value then we could also alias with static allocations like if %sz
// is known less than 1000b it could reuse any static allocation >= 1000b.
//
// We could also emit code for efficient runtime bucketing by providing the
// sorted, compacted, delta-coded lifetime intervals and runtime-computed
// sizes if we wanted to produce the smallest buffers.
//
// Slice packed offset SSA values will be updated and start at the given
// |baseOffset|. Returns |baseOffset| + the total size of the allocation
// aligned to the requirements of |resourceConfig|.
static Value
packDynamicSlicesConservatively(IREE::Stream::ResourcePackOp packOp,
                                Value baseOffset, MutableArrayRef<Slice> slices,
                                IREE::Stream::ResourceConfigAttr resourceConfig,
                                IndexSet &indexSet, OpBuilder &builder) {
  auto loc = packOp.getLoc();
  int64_t offsetAlignment = resourceConfig.getMinBufferOffsetAlignment();
  int64_t rangeAlignment = resourceConfig.getMinBufferRangeAlignment();

  // Bucket all slices by their size SSA value. We rely on shapes being
  // lowered to computed byte sizes and CSE to then dedupe the values for us.
  llvm::MapVector<Value, SmallVector<Slice *>> slicesBySize;
  for (auto &slice : slices) {
    slicesBySize[slice.dynamicSize].push_back(&slice);
  }

  // Allocate each bucket while observing the lifetime of the slices within.
  // Two or more slices may overlap in lifetime and need their own unique
  // reservation.
  Value offset = baseOffset;
  for (auto &sizeBucket : slicesBySize) {
    auto sliceSize = builder.createOrFold<IREE::Util::AlignOp>(
        loc, sizeBucket.first, rangeAlignment);
    auto &slices = sizeBucket.second;
    std::stable_sort(slices.begin(), slices.end());

    // Bin the slices by those that do not overlap. All of the allocations in
    // each bin can alias.
    // NOTE: O(n^2) in the worst case but there's usually only a small number
    // of bins (<10) as we have already bucketed by size class. We could do
    // some sorting and make this O(nlogn) or O(logn) with some interval tree
    // magic.
    struct Bin {
      Value offset;
      SmallVector<const Slice *> slices;
      bool intersects(const Slice &slice) const {
        for (auto *binSlice : slices) {
          if (binSlice->intersects(slice))
            return true;
        }
        return false;
      }
    };
    SmallVector<Bin> bins;
    for (auto *slice : slices) {
      // Try to find a bin we can reuse (non-intersecting lifetime).
      Bin *targetBin = nullptr;
      for (auto &bin : bins) {
        if (!bin.intersects(*slice)) {
          targetBin = &bin;
          break;
        }
      }
      if (!targetBin) {
        // Allocate a new bin for this slice.
        bins.push_back({offset, {}});
        targetBin = &bins.back();
        auto binSize =
            builder.createOrFold<arith::AddIOp>(loc, offset, sliceSize);
        offset = builder.createOrFold<IREE::Util::AlignOp>(loc, binSize,
                                                           offsetAlignment);
      }
      targetBin->slices.push_back(slice);
      slice->packedOffset.replaceAllUsesWith(targetBin->offset);
    }
  }

  return builder.createOrFold<IREE::Util::AlignOp>(loc, offset, rangeAlignment);
}

//===----------------------------------------------------------------------===//
// --iree-stream-layout-slices
//===----------------------------------------------------------------------===//

struct LayoutSlicesPass
    : public IREE::Stream::impl::LayoutSlicesPassBase<LayoutSlicesPass> {
  void runOnOperation() override {
    mlir::CallableOpInterface parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    // NOTE: we could try several algorithms and compute which packs best. For
    // now we just pack greedily as it's fast and what most existing ML
    // frameworks do.
    auto walkResult = parentOp.walk([&](IREE::Stream::ResourcePackOp packOp) {
      // Derive resource constraints based on pack affinity.
      auto resourceConfig = IREE::Stream::ResourceConfigAttr::lookup(packOp);

      // Bucket into static and dynamic sizes. Static packing is a much more
      // constrained problem.
      auto allSlices = packOp.getSlices();
      SmallVector<Slice> staticSlices;
      SmallVector<Slice> dynamicSlices;
      staticSlices.reserve(allSlices.size());
      dynamicSlices.reserve(allSlices.size());
      for (auto &slice : allSlices) {
        if (isa_and_nonnull<arith::ConstantOp>(
                slice.dynamicSize.getDefiningOp())) {
          staticSlices.push_back(slice);
        } else {
          dynamicSlices.push_back(slice);
        }
      }

      OpBuilder builder(packOp);
      IndexSet indexSet(packOp.getLoc(), builder);

      // First pack all static slices as these are entirely knowable here at
      // compile time.
      auto offset = packOp.getOffset() ? packOp.getOffset() : indexSet.get(0);
      if (!staticSlices.empty()) {
        FailureOr<StaticSlabLayout> layout =
            layoutStaticSlices(packOp, staticSlices, resourceConfig);
        if (failed(layout)) {
          return WalkResult::interrupt();
        }
        offset = emitStaticLayout(packOp, offset, staticSlices, *layout,
                                  indexSet, builder);

        // TODO(benvanik): make this an option; it can be useful for debugging
        // this code.
        // offset = packSlicesWithNoAliasing(packOp, offset, staticSlices,
        //                                   resourceConfig, indexSet, builder);
      }

      // Next pack all dynamic slices. Depending on the analysis information
      // available we could reuse static slices with non-overlapping lifetimes
      // in some cases.
      if (!dynamicSlices.empty()) {
        offset = packDynamicSlicesConservatively(
            packOp, offset, dynamicSlices, resourceConfig, indexSet, builder);
      }

      for (auto op : packOp.getTotalLength().getUsers()) {
        if (llvm::dyn_cast<IREE::Stream::CmdExecuteOp>(op)) {
          auto value = offset.getDefiningOp()->getAttr("value");
          if (value) {
            int64_t valueI64 =
                llvm::dyn_cast_if_present<IntegerAttr>(value).getInt();
            auto integerAttr = builder.getI64IntegerAttr(valueI64);
            op->setAttr("iree.stream.partitioning.size", integerAttr);
            break;
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Size of transient slab is dynamic, "
                                       "setting dummy value of 40");
            auto dummyValue = builder.getI64IntegerAttr(40);
            op->setAttr("iree.stream.partitioning.size", dummyValue);
            break;
          }
        }
      }

      // Total packed length is the current offset after all slices are
      // allocated. This should be aligned to the range constraints.
      packOp.getTotalLength().replaceAllUsesWith(offset);

      packOp.erase();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
