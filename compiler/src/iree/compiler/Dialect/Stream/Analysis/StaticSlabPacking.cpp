// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/StaticSlabPacking.h"

#include <algorithm>
#include <list>
#include <numeric>
#include <tuple>

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Diagnostics.h"

#if defined(IREE_HAVE_IDEALLOC)
#include "idealloc.h"
#endif // IREE_HAVE_IDEALLOC

namespace mlir::iree_compiler::IREE::Stream {

namespace {

enum class StaticPacker { Greedy, Idealloc, Best };

llvm::cl::opt<StaticPacker> clStaticPacker(
    "iree-stream-layout-slices-static-packer",
    llvm::cl::desc("Packer for statically-sized transient slices, in "
                   "LayoutSlices and in the memory-aware partitioning "
                   "scheduler."),
    llvm::cl::values(
        clEnumValN(StaticPacker::Greedy, "greedy",
                   "Greedy best-fit strip packing (default)."),
        clEnumValN(StaticPacker::Idealloc, "idealloc",
                   "idealloc (third_party/idealloc; needs a compiler built "
                   "with -DIREE_ENABLE_IDEALLOC=ON)."),
        clEnumValN(StaticPacker::Best, "best",
                   "Runs greedy and idealloc and keeps the smaller slab "
                   "(greedy on ties).")),
    llvm::cl::init(StaticPacker::Greedy));

llvm::cl::opt<unsigned> clIdeallocMaxLives(
    "iree-stream-layout-slices-idealloc-max-lives",
    llvm::cl::desc("Number of tries idealloc gets to beat its bootstrap "
                   "heuristic."),
    llvm::cl::init(10));

llvm::cl::opt<double> clIdeallocMaxFrag(
    "iree-stream-layout-slices-idealloc-max-frag",
    llvm::cl::desc("idealloc stops once the slab is at most this many times "
                   "the peak of the live slice sizes (>= 1.0)."),
    llvm::cl::init(1.0));

llvm::cl::opt<uint64_t> clIdeallocSeed(
    "iree-stream-layout-slices-idealloc-seed",
    llvm::cl::desc("Seed for all of idealloc's random choices; the same seed "
                   "gives the same layout."),
    llvm::cl::init(0));

bool intersects(const StaticSlabSlice &a, const StaticSlabSlice &b) {
  return a.end >= b.start && b.end >= a.start;
}

// Packs a set of statically-sized slices by greedy strip packing.
//
// This is the same algorithm used in tflite here:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/simple_memory_arena.cc
// It's not fantastic and can end up with a significant amount of wastage.
// They do the packing at runtime and as such care more about performance than
// we do while doing the packing here offline. For this initial version I
// wanted to ensure we matched apples/apples their implementation.
//
// We should use a set of heuristics and pick the smallest one. There are also
// some really great papers that have approximations (as all of these are -
// 2D strip packing is NP-hard) such as
// https://www.sciencedirect.com/science/article/pii/S0925772113001016 that
// someone with a brain able to parse mathy papers can try implementing.
//
// Slices are placed in input order, so the result depends on it.
StaticSlabLayout packGreedily(ArrayRef<StaticSlabSlice> slices,
                              int64_t offsetAlignment, int64_t rangeAlignment) {
  struct Reservation {
    const StaticSlabSlice *slice = nullptr;
    int64_t staticOffset = 0;
    int64_t staticSize = 0;
  };
  static constexpr int64_t UNASSIGNED = INT64_MAX;

  std::list<Reservation> reservations;
  StaticSlabLayout layout;
  int64_t highwaterMark = 0;
  for (auto &slice : slices) {
    int64_t bestOffset = UNASSIGNED;
    int64_t bestOffsetFit = UNASSIGNED;
    int64_t alignedSize = IREE::Util::align(slice.size, rangeAlignment);

    // Iterate through reservations (sorted by ascending offset) and identify
    // gaps in which the slice will fit. To reduce wastage we want to find the
    // smallest gap.
    int64_t currentOffset = 0;
    for (auto &reservation : reservations) {
      if (!intersects(*reservation.slice, slice)) {
        // Non-overlapping - we can reuse the currentOffset (assuming we find
        // no better place).
        continue;
      }

      // If we found a gap >= the required size and smaller than
      // previous best fit take it.
      int64_t alignedOffset = IREE::Util::align(currentOffset, offsetAlignment);
      if (alignedOffset + alignedSize <= reservation.staticOffset &&
          reservation.staticOffset - alignedOffset < bestOffsetFit) {
        bestOffset = alignedOffset;
        bestOffsetFit = reservation.staticOffset - currentOffset;
      }
      currentOffset = std::max(currentOffset, reservation.staticOffset +
                                                  reservation.staticSize);
    }
    if (bestOffset == UNASSIGNED) {
      bestOffset = IREE::Util::align(currentOffset, offsetAlignment);
    }

    // Reserve the memory.
    Reservation reservation;
    reservation.slice = &slice;
    reservation.staticOffset = bestOffset;
    reservation.staticSize = alignedSize;
    auto insertionIt = reservations.begin();
    while (insertionIt != reservations.end() &&
           insertionIt->staticOffset < reservation.staticOffset) {
      ++insertionIt;
    }
    reservations.insert(insertionIt, reservation);
    layout.offsets.push_back(bestOffset);
    layout.sizes.push_back(alignedSize);

    // Update highwater mark indicating how much memory needs to be allocated
    // for the entire slab.
    highwaterMark = std::max(highwaterMark, bestOffset + alignedSize);
  }

  layout.size = IREE::Util::align(highwaterMark, rangeAlignment);
  return layout;
}

#if defined(IREE_HAVE_IDEALLOC)
// Packs a set of statically-sized slices with idealloc
// (https://arxiv.org/abs/2504.04874), with closed lifetimes. Sizes are
// rounded up to both alignments, so every offset (a sum of sizes) is aligned
// too; with equal alignments (the default) these are the greedy sizes.
//
// LayoutSlices and the partitioning scheduler can pass the same slices in a
// different order and with shifted times, which idealloc's layout may depend
// on. So idealloc gets them sorted by (start, end, size), with times starting
// at 0.
//
// If idealloc reports an error, warns at |loc| and fails.
FailureOr<StaticSlabLayout> packWithIdealloc(ArrayRef<StaticSlabSlice> slices,
                                             int64_t offsetAlignment,
                                             int64_t rangeAlignment,
                                             Location loc) {
  SmallVector<size_t> order(slices.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return std::make_tuple(slices[a].start, slices[a].end, slices[a].size) <
           std::make_tuple(slices[b].start, slices[b].end, slices[b].size);
  });
  int64_t origin = slices.empty() ? 0 : slices[order.front()].start;
  int64_t alignment = std::lcm(offsetAlignment, rangeAlignment);

  SmallVector<idealloc_buffer_t> buffers;
  buffers.reserve(slices.size());
  for (size_t i : order) {
    buffers.push_back(
        {static_cast<uint64_t>(slices[i].start - origin),
         static_cast<uint64_t>(slices[i].end - origin),
         static_cast<uint64_t>(IREE::Util::align(slices[i].size, alignment))});
  }

  idealloc_options_t options;
  idealloc_options_initialize(&options);
  options.liveness = IDEALLOC_LIVENESS_INCLUSIVE;
  options.max_lives = clIdeallocMaxLives;
  options.max_frag = clIdeallocMaxFrag;
  options.seed = clIdeallocSeed;
  options.alignment = alignment;

  SmallVector<uint64_t> offsets(buffers.size());
  uint64_t makespan = 0;
  idealloc_status_t status = idealloc_place(
      buffers.data(), buffers.size(), &options, offsets.data(), &makespan);
  if (status != IDEALLOC_STATUS_OK) {
    emitWarning(loc) << "idealloc failed (" << idealloc_status_string(status)
                     << "); using the greedy packer";
    return failure();
  }

  StaticSlabLayout layout;
  layout.offsets.resize(slices.size());
  layout.sizes.resize(slices.size());
  for (auto [k, i] : llvm::enumerate(order)) {
    layout.offsets[i] = offsets[k];
    layout.sizes[i] = buffers[k].size;
  }
  // A multiple of the alignment, as all sizes and offsets are
  layout.size = makespan;
  return layout;
}
#endif // IREE_HAVE_IDEALLOC

} // namespace

FailureOr<StaticSlabLayout> packStaticSlab(ArrayRef<StaticSlabSlice> slices,
                                           int64_t offsetAlignment,
                                           int64_t rangeAlignment,
                                           Location loc) {
  StaticPacker packer = clStaticPacker;
  if (packer == StaticPacker::Greedy) {
    return packGreedily(slices, offsetAlignment, rangeAlignment);
  }

#if defined(IREE_HAVE_IDEALLOC)
  if (packer == StaticPacker::Idealloc) {
    FailureOr<StaticSlabLayout> idealloc =
        packWithIdealloc(slices, offsetAlignment, rangeAlignment, loc);
    if (failed(idealloc)) {
      return packGreedily(slices, offsetAlignment, rangeAlignment);
    }
    return idealloc;
  }

  if (packer == StaticPacker::Best) {
    // The smaller slab, greedy on ties or if idealloc fails
    StaticSlabLayout greedy =
        packGreedily(slices, offsetAlignment, rangeAlignment);
    FailureOr<StaticSlabLayout> idealloc =
        packWithIdealloc(slices, offsetAlignment, rangeAlignment, loc);
    if (failed(idealloc) || greedy.size <= idealloc->size) {
      return greedy;
    }
    return idealloc;
  }
#endif // IREE_HAVE_IDEALLOC

  return failure();
}

} // namespace mlir::iree_compiler::IREE::Stream
