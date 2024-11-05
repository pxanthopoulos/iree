// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "dagP.h"
#include "option.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"

#include <cstring>
#include <fstream>
#include <string>

#define DEBUG_TYPE "iree-stream-mem-partitioning"

static llvm::cl::opt<bool> clEnableMemAwarePartitioning(
    "iree-stream-mem-aware-partitioning",
    llvm::cl::desc("Enable Memory Aware Partitioning."), llvm::cl::init(false));

static llvm::cl::opt<std::string> clEnableMemAwarePartitioningIOFile(
    "iree-stream-mem-aware-partitioning-io-file",
    llvm::cl::desc("Input/Output File Name For Memory Aware Partitioning."),
    llvm::cl::init("comp-graph.dot"));

namespace mlir::iree_compiler::IREE::Stream {

PartitionSet memoryAwarePartition(PartitionSet initialPartitions) {
  if (!clEnableMemAwarePartitioning) {
    return initialPartitions;
  }
  for (auto &partition : initialPartitions.partitions) {
    bool noDispatches = true;
    bool onlyCuda = true;
    for (const auto &op : partition.ops) {
      if (isa<IREE::Stream::AsyncDispatchOp>(op)) {
        noDispatches = false;
      }
      auto deviceAffinityAttr =
          llvm::dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(
              op->getAttr("affinity"));
      if (deviceAffinityAttr) {
        auto deviceSymbolAttr =
            deviceAffinityAttr.getDevice().getRootReference();
        auto moduleOp = op->getParentOp();
        while (moduleOp->getParentOp() && !isa<ModuleOp>(moduleOp)) {
          moduleOp = moduleOp->getParentOp();
        }
        SymbolTable symbolTable(moduleOp);
        auto rootAttrDef = symbolTable.lookup(deviceSymbolAttr);
        auto initialValue = rootAttrDef->getAttr("initial_value");
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        initialValue.print(os);
        std::string result = os.str();
        std::string device;
        if (result.find("cuda") == std::string::npos) {
          onlyCuda = false;
          LLVM_DEBUG({
            initialValue.dump();
            op->dump();
            llvm::dbgs() << "\n\n";
          });
          break;
        }
      }
    }
    if (!noDispatches && onlyCuda) {
      LLVM_DEBUG({
        llvm::dbgs() << "Partition with dispatches\n\n";
        if (partition.affinity) {
          llvm::dbgs() << "AFFINITY:\n";
          partition.affinity.dump();
          llvm::dbgs() << "\n\n";
        }
        llvm::dbgs() << "INS:\n\n";
        for (auto &in : partition.ins) {
          in.dump();
        }
        llvm::dbgs() << "\n\nOUTS:\n\n";
        for (auto &out : partition.outs) {
          out.dump();
        }
        llvm::dbgs() << "\n\nOPS:\n\n";
        for (auto *op : llvm::reverse(partition.ops)) {
          op->dump();
        }
        llvm::dbgs() << "\n\n";
      });

      // TODO: create graph from set of operations
      // std::ofstream outFile("partition-graph.dot");
      // if (!outFile) {
      //   LLVM_DEBUG(llvm::dbgs()
      //              << "Failed to open file: partition-graph.dot\nReturning "
      //                 "initial partitions.\n");
      //   return initialPartitions;
      // }

      // outFile << "digraph cfg {";
      // outFile.close();

      // TODO: setup dagP
      MLGP_option opt;
      dgraph G;

      dagP_init_parameters(&opt,
                           std::min((unsigned long)3, partition.ops.size()));

      char *filename = new char[clEnableMemAwarePartitioningIOFile.size() + 1];
      std::strcpy(filename, clEnableMemAwarePartitioningIOFile.c_str());
      dagP_init_filename(&opt, filename);

      /* part_ub and part_lb (upper/lower bounds for partition sizes) are left
       * to default -> tot_weight/nbpart) {*,/} 1.03 */
      opt.seed = 0;             // initialized with time
      opt.ratio = 1.03;         // max imbalance ratio
      opt.debug = 5;            // debug verbosity
      opt.toggle = 0;           // toggle plot generation
      opt.print = 3;            // verbosity
      opt.runs = 1;             // # of runs
      opt.use_binary_input = 0; // work with binary input
      opt.write_parts = 0;      // do not output partitions in file

      opt.undir_alg = UNDIR_SCOTCH; // undirected partitioning algo

      // opt.ccr = 0;             // comm/comp ratio
      opt.live_traversal = 0;  // traverse nodes in natural ordering
      opt.obj = 0;             // objective -> minimize edge-cut
      opt.ignore_livesize = 0; // ?
      opt.nbProc = 1;          // # of processors

      LLVM_DEBUG({
        const char c = '\n';
        printOptions(&opt, c);
      });

      // TODO: do partitioning (ratio is size available mem over min transient
      // size -> worst case the min transient causes overflow and is placed
      // alone)

      dagP_read_graph(filename, &G, &opt);

      idxType *parts = (idxType *)calloc((G.nVrtx + 1), sizeof(idxType));
      if (parts == NULL) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to allocate space for the partitions.\nReturning "
                      "initial partitions.\n");
        // TODO: cleanup
        delete[] filename;
        dagP_free_option(&opt);
        dagP_free_graph(&G);
        return initialPartitions;
      }

      ecType x = dagP_partition_from_dgraph(&G, &opt, parts);

      LLVM_DEBUG({
        printf("edge cut: %d\n", (int)x);
        // node id's are 1-indexed
        for (idxType i = 1; i <= G.nVrtx; ++i) {
          printf("part[node:%d] = %d\n", i, parts[i]);
        }
      });

      // TODO: create new partition set and return it

      // TODO: cleanup
      free(parts);
      delete[] filename;
      dagP_free_option(&opt);
      dagP_free_graph(&G);
    }
  }
  return initialPartitions;
}

} // namespace mlir::iree_compiler::IREE::Stream