// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_ANALYSIS_STATICSLABPACKING_H_
#define IREE_COMPILER_DIALECT_STREAM_ANALYSIS_STATICSLABPACKING_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Stream {

// A statically-sized transient slice: live over the closed interval
// [start, end], |size| bytes before alignment.
struct StaticSlabSlice {
  int64_t start = 0;
  int64_t end = 0;
  int64_t size = 0;
};

// A layout of static slices into one slab.
struct StaticSlabLayout {
  // Offset of each slice, in input order, relative to the slab start.
  SmallVector<int64_t> offsets;
  // Size of each slice as packed (aligned to the range alignment).
  SmallVector<int64_t> sizes;
  // Slab size, aligned to the range alignment.
  int64_t size = 0;
};

// Packs |slices| into one slab with the packer chosen by
// --iree-stream-layout-slices-static-packer (greedy, idealloc or best). Used
// by LayoutSlices for the static slices of each stream.resource.pack and by
// memory-aware partitioning to weigh partitions, so that the scheduler sees
// the slabs LayoutSlices will produce.
//
// If idealloc reports an error, warns at |loc| and returns the greedy layout.
// Fails only if idealloc or best is asked for in a compiler built without
// idealloc (-DIREE_ENABLE_IDEALLOC=ON).
FailureOr<StaticSlabLayout> packStaticSlab(ArrayRef<StaticSlabSlice> slices,
                                           int64_t offsetAlignment,
                                           int64_t rangeAlignment,
                                           Location loc);

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_ANALYSIS_STATICSLABPACKING_H_
