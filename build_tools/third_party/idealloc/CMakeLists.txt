# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# idealloc (third_party/idealloc) is a Rust workspace. Its idealloc-ffi crate
# is built with cargo into a static library with a C header, exposed here as
# the `idealloc_ffi` target, for linking only. Users compile against
# IREE_IDEALLOC_INCLUDE_DIR and define IREE_HAVE_IDEALLOC=1 on their own
# sources only: IREE libraries pass their dependencies on publicly, so an
# interface define or include path would reach (and, when the option flips,
# recompile) every dependent object.
#
# With IREE_ENABLE_IDEALLOC off, `idealloc_ffi` is an empty interface library,
# so libraries can always depend on it.

if(NOT IREE_ENABLE_IDEALLOC)
  add_library(idealloc_ffi INTERFACE)
  return()
endif()

find_program(IREE_CARGO_EXECUTABLE cargo REQUIRED)
find_package(Threads REQUIRED)

set(_IDEALLOC_SOURCE_DIR "${IREE_SOURCE_DIR}/third_party/idealloc")
set(IREE_IDEALLOC_INCLUDE_DIR "${_IDEALLOC_SOURCE_DIR}/idealloc-ffi/include"
    CACHE INTERNAL "Directory of idealloc.h.")
set(_IDEALLOC_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(_IDEALLOC_LIBRARY
    "${_IDEALLOC_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}idealloc_ffi${CMAKE_STATIC_LIBRARY_SUFFIX}")

# Rebuild when any input of the two crates changes.
file(GLOB_RECURSE _IDEALLOC_INPUTS CONFIGURE_DEPENDS
  "${_IDEALLOC_SOURCE_DIR}/coreba/src/*.rs"
  "${_IDEALLOC_SOURCE_DIR}/idealloc-ffi/src/*.rs"
  "${_IDEALLOC_SOURCE_DIR}/idealloc-ffi/include/*.h"
)
list(APPEND _IDEALLOC_INPUTS
  "${_IDEALLOC_SOURCE_DIR}/Cargo.toml"
  "${_IDEALLOC_SOURCE_DIR}/Cargo.lock"
  "${_IDEALLOC_SOURCE_DIR}/coreba/Cargo.toml"
  "${_IDEALLOC_SOURCE_DIR}/idealloc-ffi/Cargo.toml"
)

add_custom_command(
  OUTPUT "${_IDEALLOC_LIBRARY}"
  COMMAND "${IREE_CARGO_EXECUTABLE}" build --release --locked --quiet
          --package idealloc-ffi
          --manifest-path "${_IDEALLOC_SOURCE_DIR}/Cargo.toml"
          --target-dir "${_IDEALLOC_TARGET_DIR}"
  DEPENDS ${_IDEALLOC_INPUTS}
  COMMENT "Building idealloc with cargo"
  USES_TERMINAL
  VERBATIM
)
add_custom_target(idealloc_ffi_cargo DEPENDS "${_IDEALLOC_LIBRARY}")

add_library(idealloc_ffi STATIC IMPORTED GLOBAL)
set_target_properties(idealloc_ffi PROPERTIES
  IMPORTED_LOCATION "${_IDEALLOC_LIBRARY}"
)
# The Rust standard library inside the archive needs these.
target_link_libraries(idealloc_ffi INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)
add_dependencies(idealloc_ffi idealloc_ffi_cargo)
