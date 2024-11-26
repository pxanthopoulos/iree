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
// #include <string>

#define DEBUG_TYPE "iree-stream-mem-partitioning"

static llvm::cl::opt<std::string> clEnableMemAwarePartitioningIOFile(
    "iree-stream-mem-aware-partitioning-io-file",
    llvm::cl::desc("Input/Output File Name For Memory Aware Partitioning. Will "
                   "create files like: {name}-{partition number}.dot"),
    llvm::cl::init("comp-graph.dot"));

namespace mlir::iree_compiler::IREE::Stream {

static std::unique_ptr<AsmState> getRootAsmState(Block *block) {
  LLVM_DEBUG({
    auto *rootOp = block->getParentOp();
    while (auto parentOp = rootOp->getParentOp()) {
      if (!isa<IREE::Stream::TimelineOpInterface>(parentOp) &&
          parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
        rootOp = parentOp;
        break;
      }
      rootOp = parentOp;
    }
    return std::make_unique<AsmState>(rootOp);
  });
  return nullptr;
}

bool checkSomeDispatches(Partition partition) {
  for (const auto &op : partition.ops) {
    auto dispatchOp = llvm::dyn_cast<IREE::Stream::AsyncDispatchOp>(op);
    if (dispatchOp)
      return true;
  }
  return false;
}

PartitionSet memoryAwarePartition(PartitionSet initialPartitions,
                                  Block *block) {
  auto asmState = getRootAsmState(block);
  for (const auto &[partitionIndex, partition] :
       llvm::enumerate(initialPartitions.partitions)) {
    if (!checkSomeDispatches(partition)) {
      continue;
    }

    // TODO: create graph from set of operations

    std::ofstream outFile("partition-graph-" + std::to_string(partitionIndex) +
                              ".dot",
                          std::ios::trunc);
    if (!outFile) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to open file: partition-graph-" << partitionIndex
                 << ".dot\nReturning "
                    "initial partitions.\n");
      return initialPartitions;
    }
    outFile << "digraph cfg {\n";

    llvm::DenseMap<unsigned, Operation *> opMap;
    llvm::DenseMap<Operation *, unsigned> inverseOpMap;
    int64_t totalWeight = 0;
    for (const auto [opIndex, op] :
         llvm::enumerate(llvm::reverse(partition.ops))) {
      opMap[opIndex] = op;
      inverseOpMap[op] = opIndex;
      int64_t opWeight = 0;

      auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(*op);
      auto sizeAwareOp = dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(op);

      for (const auto &[resultIndex, result] :
           llvm::enumerate(op->getResults())) {
        if (!llvm::isa<IREE::Stream::ResourceType>(result.getType())) {
          continue;
        }
        if (tiedOp) {
          auto tiedOperand = tiedOp.getTiedResultOperand(result);
          if (tiedOperand) {
            continue;
          }
        }

        // dummy default value, partitioning only works for static sizes
        int64_t actualSize = 50;
        if (sizeAwareOp) {
          auto resultSize = sizeAwareOp.getResultSizeFromValue(result);
          auto valueTypedAttr =
              resultSize.getDefiningOp()->getAttrOfType<TypedAttr>("value");

          if (resultSize.getType().isIntOrIndex() && valueTypedAttr) {
            auto integerAttr =
                llvm::dyn_cast<mlir::IntegerAttr>(valueTypedAttr);
            if (integerAttr) {
              actualSize = integerAttr.getInt();
            }
          }
        }

        opWeight += actualSize;
      }
      outFile << opIndex << "[weight=" << opWeight << "];\n";
      totalWeight += opWeight;
    }

    for (const auto op : llvm::reverse(partition.ops)) {
      for (const Value &value : op->getOperands()) {
        Operation *definingOp = value.getDefiningOp();

        if (partition.ops.contains(definingOp)) {
          int64_t actualSize = 50;

          auto sizeAwareOp =
              dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(definingOp);
          if (sizeAwareOp) {
            auto resultSize = sizeAwareOp.getResultSizeFromValue(value);
            auto valueTypedAttr =
                resultSize.getDefiningOp()->getAttrOfType<TypedAttr>("value");

            if (resultSize.getType().isIntOrIndex() && valueTypedAttr) {
              actualSize =
                  llvm::dyn_cast<mlir::IntegerAttr>(valueTypedAttr).getInt();
            }
          }
          unsigned parentOpIndex = inverseOpMap[definingOp];
          unsigned opIndex = inverseOpMap[op];
          outFile << parentOpIndex << "->" << opIndex
                  << "[weight=" << actualSize << "];\n";
        }
      }
    }

    outFile << "}\n";
    outFile.close();

    // TODO: setup dagP
    if (1) {
      MLGP_option opt;
      dgraph G;

      totalWeight = 72;
      int64_t memoryLimit = 11;
      dagP_init_parameters(&opt,
                           static_cast<unsigned long>(
                               (totalWeight + memoryLimit - 1) / memoryLimit));

      char *filename = new char[clEnableMemAwarePartitioningIOFile.size() + 1];
      std::strcpy(filename, clEnableMemAwarePartitioningIOFile.c_str());
      dagP_init_filename(&opt, filename);

      /* part_ub and part_lb (upper/lower bounds for partition sizes) are
       * left to default -> tot_weight/nbpart) {*,/} 1.03
       *
       */
      //      for (int i = 0; i < opt.nbPart; ++i) {
      //        opt.ub[i] = 10;
      //        opt.lb[i] = 1;
      //      }
      opt.seed = 0;             // initialized with time
      opt.ratio = 1;            // max imbalance ratio
      opt.debug = 0;            // debug verbosity
      opt.toggle = 0;           // toggle plot generation
      opt.print = 0;            // verbosity
      opt.runs = 5;             // # of runs
      opt.use_binary_input = 0; // work with binary input
      opt.write_parts = 0;      // do not output partitions in file

      opt.undir_alg = UNDIR_SCOTCH; // undirected partitioning algo

      // opt.ccr = 0;             // comm/comp ratio
      opt.live_traversal = 0;  // traverse nodes in natural ordering
      opt.obj = 0;             // objective -> minimize edge-cut
      opt.ignore_livesize = 0; // ?
      opt.nbProc = 1;          // # of processors

      LLVM_DEBUG({
        llvm::dbgs() << "{\nBEFORE\n";
        const char c = '\n';
        printOptions(&opt, c);
        fflush(stdout);
        llvm::dbgs() << "}\n\n\n";
      });

      // TODO: do partitioning (ratio is size available mem over min
      // transient size -> worst case the min transient causes overflow and
      // is placed alone)

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
        printf("{\nedge cut: %d\n", (int)x);
        // node id's are 1-indexed
        for (idxType i = 1; i <= G.nVrtx; ++i) {
          printf("part[node:%d] = %d\n", i, parts[i]);
        }
        printf("}\n\n\n");
        fflush(stdout);
      });

      LLVM_DEBUG({
        llvm::dbgs() << "{\nAFTER\n";
        for (int i = 0; i < opt.nbPart; i++) {
          printf("For partition %d, LB: %lf, UB: %lf\n", i, opt.lb[i],
                 opt.ub[i]);
          fflush(stdout);
        }
        llvm::dbgs() << "}\n\n\n";
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