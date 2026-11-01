// RUN: iree-opt --split-input-file --iree-stream-layout-slices-static-packer=idealloc --pass-pipeline='builtin.module(util.func(iree-stream-layout-slices, cse))' %s | FileCheck %s --check-prefixes=CHECK,IDEALLOC
// RUN: iree-opt --split-input-file --iree-stream-layout-slices-static-packer=best --pass-pipeline='builtin.module(util.func(iree-stream-layout-slices, cse))' %s | FileCheck %s --check-prefixes=CHECK,BEST

// Only built with -DIREE_ENABLE_IDEALLOC=ON. idealloc is seeded
// (--iree-stream-layout-slices-idealloc-seed, default 0), so its layouts are
// fixed.

#layoutStaticConfig = #stream.resource_config<{
  max_allocation_size = 1073741824,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// The input of @layoutStatic in layout_slices.mlir. Both packers need 432
// bytes, the peak of the live sizes (at time 1: 112 + 112 + 208); on the tie,
// best keeps the greedy layout.

// CHECK-LABEL: @layoutStatic
util.func public @layoutStatic() -> (index, index, index, index, index, index, index)
    attributes {stream.resources = #layoutStaticConfig} {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %t:7 = stream.resource.pack slices({
    [0, 1] = %c100,
    [1, 2] = %c100,
    [2, 3] = %c100,
    [0, 4] = %c200,
    [5, 6] = %c200,
    [5, 8] = %c100,
  }) : index
  // IDEALLOC: util.return %c432
  // IDEALLOC-SAME: %c208, %c320, %c208, %c0, %c0, %c208
  // BEST: util.return %c432
  // BEST-SAME: %c0, %c112, %c0, %c224, %c0, %c208
  util.return %t#0, %t#1, %t#2, %t#3, %t#4, %t#5, %t#6 : index, index, index, index, index, index, index
}

// -----

#layoutBetterConfig = #stream.resource_config<{
  max_allocation_size = 1073741824,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// A gpt2-xl slab (sizes divided by 51200). Greedy puts [4, 7] above [3, 4]
// at 128, as the gap below [3, 4] is too small: 320 bytes. idealloc moves
// [3, 4] up and fits the slab in the peak of the live sizes, 256 bytes; best
// takes it.

// CHECK-LABEL: @layoutIdeallocBetter
util.func public @layoutIdeallocBetter() -> (index, index, index, index, index)
    attributes {stream.resources = #layoutBetterConfig} {
  %c64 = arith.constant 64 : index
  %c192 = arith.constant 192 : index
  %c256 = arith.constant 256 : index
  %t:5 = stream.resource.pack slices({
    [0, 1] = %c256,
    [2, 3] = %c64,
    [3, 4] = %c64,
    [4, 7] = %c192,
  }) : index
  // CHECK: util.return %c256
  // CHECK-SAME: %c0, %c0, %c192, %c0
  util.return %t#0, %t#1, %t#2, %t#3, %t#4 : index, index, index, index, index
}

// -----

#layoutClosedConfig = #stream.resource_config<{
  max_allocation_size = 1073741824,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// Lifetimes are closed intervals: [0, 1] and [1, 2] are both live at 1, and
// the two [3, 3] slices at 3, so neither pair may alias.

// CHECK-LABEL: @layoutClosedIntervals
util.func public @layoutClosedIntervals() -> (index, index, index, index, index)
    attributes {stream.resources = #layoutClosedConfig} {
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %t:5 = stream.resource.pack slices({
    [0, 1] = %c64,
    [1, 2] = %c128,
    [3, 3] = %c64,
    [3, 3] = %c64,
  }) : index
  // IDEALLOC: util.return %c192
  // IDEALLOC-SAME: %c128, %c0, %c0, %c64
  // BEST: util.return %c192
  // BEST-SAME: %c0, %c64, %c0, %c64
  util.return %t#0, %t#1, %t#2, %t#3, %t#4 : index, index, index, index, index
}

// -----

#layoutMixedStaticDynamicConfig = #stream.resource_config<{
  max_allocation_size = 1073741824,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// Dynamic slices are still packed after the static slab, as in
// layout_slices.mlir.

// CHECK-LABEL: @layoutMixedStaticDynamic
// CHECK-SAME: (%[[SIZE_A:.+]]: index, %[[SIZE_B:.+]]: index)
util.func public @layoutMixedStaticDynamic(%size_a: index, %size_b: index) -> (index, index, index, index, index)
    attributes {stream.resources = #layoutMixedStaticDynamicConfig} {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %t:5 = stream.resource.pack slices({
    [0, 1] = %c100,
    [1, 2] = %size_a,
    [2, 3] = %size_b,
    [5, 6] = %c200,
  }) : index

  // CHECK-DAG: %c0 = arith.constant 0 : index
  // CHECK-DAG: %c208 = arith.constant 208 : index
  // CHECK-DAG: %c16 = arith.constant 16 : index
  // CHECK-DAG: %0 = util.align %[[SIZE_A]], %c16 : index
  // CHECK-DAG: %1 = arith.addi %0, %c208 : index
  // CHECK-DAG: %2 = util.align %[[SIZE_B]], %c16 : index
  // CHECK-DAG: %3 = arith.addi %1, %2 : index

  // CHECK: util.return %3, %c0, %c208, %1, %c0
  util.return %t#0, %t#1, %t#2, %t#3, %t#4 : index, index, index, index, index
}
