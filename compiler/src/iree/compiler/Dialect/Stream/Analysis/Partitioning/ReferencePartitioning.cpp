// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/Analysis/ResourceHazards.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-stream-partitioning"

namespace mlir::iree_compiler::IREE::Stream {

// Returns an AsmState at the ancestor to |block| that is isolated from above.
// Returns nullptr if debug dumps of partitioning is disabled.
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

// This is terrible. See Stream/Analysis/Partition.h for a description of what
// a real implementation would do. We want cost modeling for tie breakers when
// an op could be in multiple partitions, cloning for ops that are not worth
// spanning partitions (like splats), etc.
PartitionSet
partitionStreamableOpsReference(IREE::Stream::PartitioningConfigAttr config,
                                Block *block) {
  PartitionSet partitionSet;

  struct OpInfo {
    // Which partitions the op is contained within.
    llvm::BitVector membership;
    // Which partitions transitively depend on this operation.
    llvm::BitVector hazards;
  };
  DenseMap<Operation *, OpInfo> opInfos;

  struct PartitionBuilder {
    unsigned ordinal;
    // Affinity of the partition.
    IREE::Stream::AffinityAttr affinity;
    // Ops present in the partition; ops may be present in multiple partitions.
    SetVector<Operation *> ops;
    // Ops that were cloned and are known not to have their values escape.
    DenseSet<Operation *> clonedOps;
    // Which partitions transitively depend on this partition.
    llvm::BitVector hazards;
    void insert(Operation *op, OpInfo &opInfo) {
      if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
        affinity = affinity ? affinity.joinAND(affinityOp.getAffinityAttr())
                            : affinityOp.getAffinityAttr();
      }
      opInfo.membership.set(ordinal);
      if (opInfo.hazards.size() > ordinal)
        opInfo.hazards.reset(ordinal);
      ops.insert(op);
      hazards |= opInfo.hazards;
    }
  };
  SmallVector<std::unique_ptr<PartitionBuilder>> builders;
  llvm::BitVector usableBuilders;

  auto willCreateCircularDependencyBetweenPartitions =
      [&](unsigned sourceOrdinal, unsigned targetOrdinal) -> bool {
    // Returns:
    // If we are to make partition with ordinal targetOrdinal to
    // depend on partition with ordinal sourceOrdinal,
    // will this create a circular dependency.
    if (sourceOrdinal == targetOrdinal)
      return false;
    return builders[sourceOrdinal]->hazards.size() > targetOrdinal &&
           builders[sourceOrdinal]->hazards[targetOrdinal];
  };

  auto canAddOpToPartition = [&](Operation &op, OpInfo &opInfo,
                                 unsigned partitionOrdinal) {
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);
    if (!streamableOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot add to partition: Op is not streamable\n");
      return false;
    }
    IREE::Stream::AffinityAttr affinityAttr;
    if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op))
      affinityAttr = affinityOp.getAffinityAttr();
    if (!IREE::Stream::AffinityAttr::canExecuteTogether(
            affinityAttr, builders[partitionOrdinal]->affinity)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot add to partition: Op has incompatible affinity\n");
      return false;
    }

    bool preferCloneToConsumers = streamableOp.preferCloneToConsumers();
    llvm::BitVector *opHazards = nullptr;
    llvm::BitVector opHazardsInCandidatePartition;
    if (preferCloneToConsumers) {
      // If we are cloning we care only about users that are a part of the
      // candidate partition.
      // Here we would need to walk further down the users if a user is also
      // cloned into the partition. This will be useful if we have a block of
      // cloneable ops. If left like that, other than the inefficiency,
      // it should not produce invalid partitioning.
      opHazards = &opHazardsInCandidatePartition;
      for (auto user : op.getUsers()) {
        if (builders[partitionOrdinal]->ops.contains(user))
          opHazardsInCandidatePartition |= opInfos[user].hazards;
      }
    } else
      opHazards = &opInfo.hazards;

    for (auto opHazardOrdinal : opHazards->set_bits()) {
      // Smaller because we iterate over the ops in reverse order so low ordinal
      // means later in the original IR
      if (partitionOrdinal < opHazardOrdinal) {
        // Reject partition ordering that would require partition sorting.
        // TODO: It is probably more optimal to reorder the partitions after
        // their formation based on their dependency graph instead of rejecting
        // here. Since this is considered not a good partitioning algorithm
        // and will probably get removed, we leave it like that.
        LLVM_DEBUG(llvm::dbgs()
                   << "Cannot add to partition: Op has hazard with earlier "
                      "partition (no reordering of partitions)\n");
        return false;
      }
      // Check for formation of circular dependency between partitions.
      if (willCreateCircularDependencyBetweenPartitions(opHazardOrdinal,
                                                        partitionOrdinal)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cannot add to partition: Op has hazard another "
                      "partition which will lead to circular dependency with "
                      "the candidate\n");
        return false;
      }
    }
    return true;
  };

  auto asmState = getRootAsmState(block);

  LLVM_DEBUG(
      llvm::dbgs() << "~~~ Iterating over ops in the block in reverse:\n\n");
  for (auto &op : llvm::reverse(*block)) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n~~~ ~~~ On op:\n";
      op.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });
    // Skip constants; they just add noise (and since they are heavily CSE'd
    // they have lots of users to test).
    if (op.hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "(ignoring constant, continuing...)\n\n");
      continue;
    } else if (isa<IREE::Util::GlobalStoreOpInterface>(op)) {
      // We ignore global stores as they are unobservable within an execution
      // region - we must still block on loads though.
      LLVM_DEBUG(llvm::dbgs() << "(ignoring global store, continuing...)\n\n");
      continue;
    } else if (!isa<IREE::Stream::StreamableOpInterface>(op)) {
      // Not a streamable op. If it has side-effects then we force a hazard on
      // all builders so that we don't move ops across it.
      if (!mlir::wouldOpBeTriviallyDead(&op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "(non-streamable, side-effecting op forcing flush "
                          "and freeze, cannot move ops across it, resetting "
                          "usable builders)\n";
        });
        usableBuilders.reset();
      }
      // Even though not a streamable op we still want to track it below.
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Initialize op info for this op - whether streamable or not. We track
    // transitive hazards on each op. Note that thanks to the ordering of ops
    // in SSA form (_reversed here!_) we know that once we visit this op no
    // partition created after it can ever depend on it if it doesn't here. This
    // lets us keep the bitvectors small.
    auto &opInfo = opInfos[&op];
    opInfo.hazards.reserve(builders.size() + 1);
    opInfo.hazards.resize(builders.size(), /*t=*/false);

    LLVM_DEBUG({
      llvm::dbgs() << "~~~ ~~~ ~~~ Partitioning op:\n";
      op.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n\n";
    });

    LLVM_DEBUG({
      size_t counter = 0;
      for ([[maybe_unused]] auto user : op.getUsers()) {
        counter++;
      }
      llvm::dbgs() << "~~~ ~~~ ~~~ Iterating over the users of current op:\n";
      if (counter != 0)
        llvm::dbgs() << "\n";
      else
        llvm::dbgs() << "Current op has no users\n\n";
    });

    // Set bits for each partition this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a partition.
    llvm::BitVector consumers(builders.size(), /*t=*/false);
    for (auto user : op.getUsers()) {
      LLVM_DEBUG({
        llvm::dbgs() << "~~~ ~~~ ~~~ ~~~ On user:\n";
        user->print(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
      });
      auto userInfoIt = opInfos.find(user);
      if (userInfoIt == opInfos.end()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "User is untracked (was skipped or is in another "
                      "region/block), continuing...\n\n");
        continue;
      }
      auto &userInfo = userInfoIt->second;
      LLVM_DEBUG({
        llvm::dbgs() << "User info:\n";
        for (auto membershipOrdinal : userInfo.membership.set_bits()) {
          llvm::dbgs() << "  member of partition " << membershipOrdinal << "\n";
        }
        for (auto hazardOrdinal : userInfo.hazards.set_bits()) {
          llvm::dbgs() << "  hazard w/ partition " << hazardOrdinal << "\n";
        }
      });
      LLVM_DEBUG({
        llvm::dbgs()
            << "\n~~~ ~~~ ~~~ ~~~ Adding partitions that user is a part of in "
               "the list of consumer partitions for the current op\n";
        llvm::dbgs()
            << "~~~ ~~~ ~~~ ~~~ Adding partitions that user is a part of in "
               "the list of hazard (dependent) partitions for the current op\n";
        llvm::dbgs()
            << "~~~ ~~~ ~~~ ~~~ Adding hazard partitions of the user "
               "to the list of hazard partitions for the current op\n\n";
      });
      consumers |= userInfo.membership;
      opInfo.hazards |= userInfo.membership;
      opInfo.hazards |= userInfo.hazards;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "~~~ ~~~ ~~~ Removing hazard partitions from the "
                      "candidate partitions of the current op\n";
      llvm::dbgs() << "~~~ ~~~ ~~~ Adding consumer partitions to the "
                      "candidate partitions of the current op\n";
      llvm::dbgs() << "~~~ ~~~ ~~~ Filters remaining candidates based on "
                      "usable builders (created partitions so far)\n\n";
    });
    llvm::BitVector candidates(builders.size(), /*t=*/true);
    candidates ^= opInfo.hazards;
    candidates |= consumers;
    candidates &= usableBuilders;

    LLVM_DEBUG(llvm::dbgs()
               << "~~~ ~~~ ~~~ Pruning candidate partitions that are "
                  "incompatible:\n");
    // Prune candidates that do not have a compatible affinity.
    for (auto ordinal : candidates.set_bits()) {
      if (!canAddOpToPartition(op, opInfo, ordinal)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Candidate partition " << ordinal << " incompatible\n");
        candidates.reset(ordinal);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // If this op is not streamable then bail here; we've still setup the hazard
    // map for following iteration.
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);
    if (!streamableOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "~~~ ~~~ ~~~  Op is not streamable, skipping now...\n\n");
      continue;
    }

    // First see which partitions are consuming this that we can also safely
    // move in to.
    consumers &= candidates;

    opInfo.membership.reserve(builders.size() + 1);
    opInfo.membership.resize(builders.size(), /*t=*/false);

    // If we have one or more consumers we should go into those first.
    if (consumers.any()) {
      LLVM_DEBUG(llvm::dbgs() << "~~~ ~~~ ~~~ Op has consumer partitions so it "
                                 "will be cloned in them (one or more)\n\n");
      // If we are a clonable op (like splat) clone us into every partition.
      // Otherwise we just pick the first we find (probably a bad heuristic).
      if (streamableOp.preferCloneToConsumers() && consumers.count() > 1) {
        LLVM_DEBUG(llvm::dbgs() << "~~~ ~~~ ~~~ ~~~ Op is clonable and has "
                                   "more than 1 consumer partitions. It will "
                                   "be cloned into all of them\n");
        for (auto consumerOrdinal : consumers.set_bits()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "~~~ ~~~ ~~~ ~~~ ~~~ Cloning into consumer partition "
                     << consumerOrdinal << "\n");
          auto &consumerBuilder = builders[consumerOrdinal];
          consumerBuilder->insert(&op, opInfo);
          consumerBuilder->clonedOps.insert(&op);
        }
        LLVM_DEBUG(llvm::dbgs() << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "~~~ ~~~ ~~~ ~~~ Op has only 1 consumer partition\n");
        int consumerOrdinal = consumers.find_last();
        LLVM_DEBUG(llvm::dbgs()
                   << "~~~ ~~~ ~~~ ~~~ Moving into consumer partition "
                   << consumerOrdinal << "\n\n");
        auto &consumerBuilder = builders[consumerOrdinal];
        consumerBuilder->insert(&op, opInfo);
      }
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "~~~ ~~~ ~~~ Op has no consumer partitions\n\n");
    // No consumers - if there's any candidate then we'll go into that.
    int firstCandidateOrdinal = candidates.find_first();
    if (firstCandidateOrdinal != -1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "~~~ ~~~ ~~~ Moving to first candidate partition "
                 << firstCandidateOrdinal << "\n\n");
      builders[firstCandidateOrdinal]->insert(&op, opInfo);
      continue;
    } else
      LLVM_DEBUG(llvm::dbgs()
                 << "~~~ ~~~ ~~~ Op has no candidate partitions at all\n\n");

    // Mark the op as having hazards against all other partitions.
    // It is better to be safe than incorrect, especially with our current
    // minimal test coverage. It's not always safe to reorder things - if
    // anything we are unlikely to be conservative enough here - for example,
    // if there's a stream.resource.load of a resource or a global we can't
    // move anything that may affect that resource or global. This partitioning
    // was designed to be conservative because debugging such issues is really
    // difficult.
    if (!builders.empty()) {
      opInfo.hazards.set(0, builders.size() - 1);
    }

    // Create a new partition just for this op.
    opInfo.membership.resize(opInfo.membership.size() + 1, /*t=*/true);
    auto builder = std::make_unique<PartitionBuilder>();
    builder->ordinal = builders.size();
    builder->insert(&op, opInfo);
    LLVM_DEBUG(llvm::dbgs() << "~~~ ~~~ ~~~ Created new partition "
                            << builder->ordinal << " for the op\n\n");
    builders.push_back(std::move(builder));
    usableBuilders.resize(builders.size(), /*t=*/true);
  }

  // Ops cloned into multiple partitions may still escape if there are
  // non-streamable consumers. We need to make sure we only let one result
  // escape.
  DenseSet<Operation *> clonedEscapingOps;

  LLVM_DEBUG(
      llvm::dbgs() << "~~~ Emitting partitions in forward order (iterate over "
                      "created builders and ops inside them in reverse)\n");
  LLVM_DEBUG(
      llvm::dbgs()
      << "~~~ Ops may be cloned into multiple partitions.\nOf those, "
         "some may have escaping uses (uses outside partitions)\nWe need to "
         "ensure only one partition produces each escaping use/result\n\n");
  // Emit partitions in forward order (as they are topologically sorted in
  // reverse order from our bottom-up walk).
  for (auto &builder : llvm::reverse(builders)) {
    Partition partition;

    SetVector<Value> consumedValues;
    SetVector<Value> producedValues;
    SetVector<Value> escapingValues;
    for (auto *op : llvm::reverse(builder->ops)) {
      bool didCloneEscape = false;
      for (auto operand : op->getOperands()) {
        consumedValues.insert(operand);
      }
      for (auto result : op->getResults()) {
        producedValues.insert(result);

        // Cloned ops default to local usage but may still have users outside
        // of any partition and need to escape.
        if (builder->clonedOps.contains(op)) {
          // We only want to have one partition produce the value and track ones
          // we've already produced via clonedEscapingOps.
          if (!clonedEscapingOps.contains(op)) {
            for (auto user : result.getUsers()) {
              if (!isa<IREE::Stream::StreamableOpInterface>(user)) {
                escapingValues.insert(result);
                didCloneEscape = true;
                break;
              }
            }
          }
        } else {
          // TODO(benvanik): optimize this - creates n^2/nlogn behavior.
          for (auto user : result.getUsers()) {
            if (!builder->ops.contains(user)) {
              escapingValues.insert(result);
            }
          }
        }
      }
      if (didCloneEscape) {
        clonedEscapingOps.insert(op);
      }
    }
    consumedValues.set_subtract(producedValues);
    partition.affinity = builder->affinity;
    partition.ins = consumedValues;
    partition.outs = escapingValues;

    partition.ops = std::move(builder->ops);
    partitionSet.partitions.push_back(std::move(partition));
  }

  LLVM_DEBUG({
    partitionSet.dump(*asmState);
    llvm::dbgs() << "\n\n";
  });

  return partitionSet;
}

// This looks to extract a single level of concurrency; we should be recursively
// dividing the block to identify both serial and concurrent regions.
PartitionSet
partitionRegionConcurrencyReference(IREE::Stream::PartitioningConfigAttr config,
                                    Block *block) {
  PartitionSet waveSet;

  auto favor = config.getFavor().getValue();
  if (favor == IREE::Stream::Favor::Debug) {
    // Disable partitioning when favoring debuggability.
    return waveSet;
  }

  struct PartitionBuilder {
    unsigned ordinal;
    // Ops present in the wave; ops may be present in multiple waves.
    SetVector<Operation *> ops;
  };
  SmallVector<std::unique_ptr<PartitionBuilder>> builders;

  struct OpInfo {
    // Which waves the op is contained within.
    llvm::BitVector membership;
    // Which waves transitively depend on this operation.
    llvm::BitVector hazards;
  };
  DenseMap<Operation *, OpInfo> opInfos;

  auto asmState = getRootAsmState(block);

  // Run analysis - if it fails then we'll just be conservative.
  IREE::Stream::ResourceHazardAnalysis hazardAnalysis(block->getParentOp());
  if (failed(hazardAnalysis.run())) {
    LLVM_DEBUG(llvm::dbgs() << "WARNING: resource hazard analysis failed; "
                               "conservatively scheduling\n");
  }

  LLVM_DEBUG(
      llvm::dbgs() << "~~~ Iterating over ops in the block in reverse:\n\n");
  for (auto &op : llvm::reverse(*block)) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n~~~ ~~~ On op:\n";
      op.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });

    // Skip constants; they just add noise (and since they are heavily CSE'd
    // they have lots of users to test).
    if (op.hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "(ignoring constant, continuing...)\n\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // NOTE: it's ok if this op is not streamable as we still need to track the
    // hazards for other ops that it may use/may use it.
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);

    // Initialize op info for this op - whether streamable or not. We track
    // transitive hazards on each op. Note that thanks to the ordering of ops
    // in SSA form (_reversed here!_) we know that once we visit this op no
    // wave created after it can ever depend on it if it doesn't here. This
    // lets us keep the bitvectors small.
    auto &opInfo = opInfos[&op];
    opInfo.hazards.reserve(builders.size() + 1);
    opInfo.hazards.resize(builders.size(), /*t=*/false);

    LLVM_DEBUG({
      llvm::dbgs() << "~~~ ~~~ ~~~ Partitioning op:\n";
      op.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n\n";
    });

    LLVM_DEBUG({
      size_t counter = 0;
      for ([[maybe_unused]] auto user : op.getUsers()) {
        counter++;
      }
      llvm::dbgs() << "~~~ ~~~ ~~~ Iterating over the users of current op:\n";
      if (counter != 0)
        llvm::dbgs() << "\n";
      else
        llvm::dbgs() << "Current op has no users\n\n";
    });

    // Set bits for each wave this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a wave.
    for (auto user : op.getUsers()) {
      LLVM_DEBUG({
        llvm::dbgs() << "~~~ ~~~ ~~~ ~~~ On user:\n";
        user->print(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
      });
      auto userInfoIt = opInfos.find(user);
      if (userInfoIt == opInfos.end()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "User is untracked (was skipped or is in another "
                      "region/block), continuing...\n\n");
        continue;
      }
      auto &userInfo = userInfoIt->second;
      LLVM_DEBUG({
        llvm::dbgs() << "User info:\n";
        for (auto membershipOrdinal : userInfo.membership.set_bits()) {
          llvm::dbgs() << "  member of wave " << membershipOrdinal << "\n";
        }
        int lastHazardOrdinal = userInfo.hazards.find_last();
        if (lastHazardOrdinal != -1) {
          llvm::dbgs() << "  hazard w/ waves 0-" << lastHazardOrdinal << "\n";
        }
      });
      LLVM_DEBUG({
        llvm::dbgs()
            << "\n~~~ ~~~ ~~~ ~~~ Checking if user has hazard with current op "
               "(also inherit users hazards)\n\n";
      });
      bool hazardPresent = hazardAnalysis.hasHazard(streamableOp, user);
      if (hazardPresent) {
        // Hazard with existing op usage - prevent concurrent scheduling.
        opInfo.hazards |= userInfo.membership;
        LLVM_DEBUG(llvm::dbgs()
                   << "~~~ ~~~ ~~~ ~~~ ~~~ Hazard analysis says NOT ok to "
                      "schedule, adding "
                      "users' membership to current op hazards\n\n");
      } else {
        LLVM_DEBUG(
            llvm::dbgs()
            << "~~~ ~~~ ~~~ ~~~ ~~~ Hazard analysis says ok to schedule\n\n");
      }
      // Always inherit hazards whether merging or not.
      opInfo.hazards |= userInfo.hazards;
    }

    LLVM_DEBUG(
        { llvm::dbgs() << "~~~ ~~~ ~~~ Iterating over the operands:\n\n"; });
    // Additional exhaustive testing for users of tied operands.
    // For each resource operand of this op we scan back through previously
    // created waves to see if there are any partitioned ops that have a hazard.
    for (auto operand : op.getOperands()) {
      LLVM_DEBUG({
        llvm::dbgs() << "~~~ ~~~ ~~~ ~~~ On operand: \n";
        operand.print(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
      });
      if (!isa<IREE::Stream::ResourceType>(operand.getType())) {
        LLVM_DEBUG({
          llvm::dbgs() << "Operand is not of resource type, skipping...\n\n";
        });
        continue;
      }
      LLVM_DEBUG({
        llvm::dbgs()
            << "\n~~~ ~~~ ~~~ ~~~ Iterating over the users of the operand:\n\n";
      });
      for (auto user : operand.getUsers()) {
        LLVM_DEBUG({
          llvm::dbgs() << "~~~ ~~~ ~~~ ~~~ ~~~ On user of operand:\n";
          user->print(llvm::dbgs(), *asmState);
          llvm::dbgs() << "\n";
        });
        if (user == &op || user->getBlock() != block ||
            user->isBeforeInBlock(&op)) {
          LLVM_DEBUG(
              {
                llvm::dbgs()
                    << "User is either the current op, is in another block or "
                       "is before the current op in the block, skipping...\n\n";
              });
          continue;
        }
        auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(user);
        if (!tiedOp || !tiedOp.hasAnyTiedUses(operand)) {
          LLVM_DEBUG({
            llvm::dbgs()
                << "User either does not support the tied op "
                   "interface or the operand has no tied uses, skipping...\n\n";
          });
          continue;
        }
        auto userInfoIt = opInfos.find(user);
        if (userInfoIt == opInfos.end()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "User is untracked (was skipped or is in another "
                        "region/block), continuing...\n\n");
          continue;
        }
        auto &userInfo = userInfoIt->second;
        LLVM_DEBUG({
          llvm::dbgs() << "\nUser is tied. User info:\n";
          for (auto membershipOrdinal : userInfo.membership.set_bits()) {
            llvm::dbgs() << "  member of wave " << membershipOrdinal << "\n";
          }
          int lastHazardOrdinal = userInfo.hazards.find_last();
          if (lastHazardOrdinal != -1) {
            llvm::dbgs() << "  hazard w/ waves 0-" << lastHazardOrdinal << "\n";
          }
        });
        LLVM_DEBUG({
          llvm::dbgs() << "\n~~~ ~~~ ~~~ ~~~ Checking if user has hazard with "
                          "current op "
                          "(also inherit users hazards)\n\n";
        });
        bool hazardPresent = hazardAnalysis.hasHazard(streamableOp, user);
        if (hazardPresent) {
          // Hazard with existing op usage - prevent concurrent scheduling.
          opInfo.hazards |= userInfo.membership;
          LLVM_DEBUG(
              llvm::dbgs()
              << "~~~ ~~~ ~~~ ~~~ ~~~ ~~~ Hazard analysis says NOT ok to "
                 "schedule, adding "
                 "users' membership to current op hazards\n\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "~~~ ~~~ ~~~ ~~~ ~~~ ~~~ Hazard analysis "
                                     "says ok to schedule\n\n");
        }
        // Always inherit hazards whether merging or not.
        opInfo.hazards |= userInfo.hazards;
      }
    }
    LLVM_DEBUG({
      llvm::dbgs()
          << "~~~ ~~~ ~~~ Candidate wave list is all minus hazards\n\n";
    });
    llvm::BitVector candidates(builders.size(), /*t=*/true);
    candidates ^= opInfo.hazards;

    // If this op is not streamable then bail here; we've still setup the hazard
    // map for following iteration.
    if (!streamableOp || streamableOp.isMetadata()) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "~~~ ~~~ ~~~ Op is not streamable/is subview, skipping...\n\n";
      });
      continue;
    }

    opInfo.membership.reserve(builders.size() + 1);
    opInfo.membership.resize(builders.size(), /*t=*/false);

    // No consumers - if there's any candidate then we'll go into that.
    int firstCandidateOrdinal = favor == IREE::Stream::Favor::MaxConcurrency
                                    ? candidates.find_first()
                                    : candidates.find_last();
    if (firstCandidateOrdinal != -1) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "~~~ ~~~ ~~~ If favoring max concurrency, go to the first "
               "candidate wave\nIf favoring min memory, go to the last\n";
        if (favor == IREE::Stream::Favor::MaxConcurrency) {
          llvm::dbgs() << "Here we are favoring max concurrency, so we merge "
                          "with the first candidate wave\n";
        } else {
          llvm::dbgs() << "Here we are favoring min memory, so we merge "
                          "with the last candidate wave (better spreaded ops "
                          "-> lower peak memory for waves)\n";
        }
      });
      LLVM_DEBUG(llvm::dbgs() << "Moving to candidate wave "
                              << firstCandidateOrdinal << " (continue)\n\n");
      builders[firstCandidateOrdinal]->ops.insert(&op);
      opInfo.membership.set(firstCandidateOrdinal);
      opInfo.hazards.set(0, firstCandidateOrdinal);
      opInfo.hazards.reset(firstCandidateOrdinal);
      continue;
    }

    LLVM_DEBUG({
      llvm::dbgs()
          << "~~~ ~~~ ~~~ Op has no candiate waves, creating a new one\n";
    });
    // Mark the op as having hazards against all other waves.
    opInfo.hazards.set(0, builders.size());

    // Create a new wave just for this op.
    opInfo.membership.resize(opInfo.membership.size() + 1, /*t=*/true);
    auto builder = std::make_unique<PartitionBuilder>();
    builder->ordinal = builders.size();
    builder->ops.insert(&op);
    LLVM_DEBUG(llvm::dbgs() << "Created wave " << builder->ordinal << "\n\n");
    builders.push_back(std::move(builder));
  }

  LLVM_DEBUG({
    if (!builders.empty())
      llvm::dbgs() << "~~~ Emitting waves in forward order\n\n";
  });

  // Emit waves in forward order (as they are topologically sorted in
  // reverse order from our bottom-up walk).
  for (auto &builder : llvm::reverse(builders)) {
    Partition wave;

    SetVector<Value> consumedValues;
    SetVector<Value> producedValues;
    SetVector<Value> escapingValues;
    for (auto *op : llvm::reverse(builder->ops)) {
      for (auto operand : op->getOperands()) {
        consumedValues.insert(operand);
      }
      for (auto result : op->getResults()) {
        producedValues.insert(result);
        // TODO(benvanik): optimize this - creates n^2/nlogn behavior.
        for (auto user : result.getUsers()) {
          if (!builder->ops.contains(user)) {
            escapingValues.insert(result);
          }
        }
      }
    }
    consumedValues.set_subtract(producedValues);
    wave.ins = consumedValues;
    wave.outs = escapingValues;

    wave.ops = std::move(builder->ops);
    waveSet.partitions.push_back(std::move(wave));
  }

  LLVM_DEBUG({
    waveSet.dump(*asmState);
    llvm::dbgs() << "\n\n";
  });

  return waveSet;
}

} // namespace mlir::iree_compiler::IREE::Stream
