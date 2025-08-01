################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Dialect/HAL/Transforms/test/BUILD.bazel           #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "annotate_target_devices.mlir"
    "assign_legacy_target_devices.mlir"
    "assign_target_devices.mlir"
    "capture_executable_sources.mlir"
    "convert_to_hal.mlir"
    "dump_executable_benchmarks.mlir"
    "dump_executable_sources.mlir"
    "elide_redundant_commands.mlir"
    "hoist_executable_objects.mlir"
    "initialize_devices.mlir"
    "inline_memoize_regions.mlir"
    "materialize_dispatch_instrumentation.mlir"
    "materialize_interfaces.mlir"
    "materialize_resource_caches.mlir"
    "materialize_target_devices.mlir"
    "memoize_device_queries.mlir"
    "memoize_device_selection.mlir"
    "outline_memoize_regions.mlir"
    "preprocess_executables.mlir"
    "prune_executables.mlir"
    "repeat_dispatches.mlir"
    "resolve_device_aliases.mlir"
    "resolve_device_promises.mlir"
    "resolve_export_ordinals.mlir"
    "resolve_ranked_shaped_type.mlir"
    "resolve_topology_queries.mlir"
    "strip_executable_contents.mlir"
    "substitute_executables.mlir"
    "verify_devices.mlir"
  TOOLS
    FileCheck
    iree-opt
  DATA
    substitute_executables_replacement.mlir
    substitute_executables_replacement.obj
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
