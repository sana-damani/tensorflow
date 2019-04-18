#include "tensorflow/compiler/xla/service/gpu/fusion_library.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"

namespace xla {
namespace gpu {

NewFusion::NodeType NewFusion::getRoot(bool RPO)
{
  auto post_order = computation->MakeInstructionPostOrder();
  if (!RPO)
    return computation->root_instruction();
  else
    return post_order.front();
}

OpPatternKind NewFusion::getPatternKind(NewFusion::NodeType instruction)
{
  if (!IsFusible(*instruction) || ImplementedAsLibraryCall(*instruction) || instruction->opcode() == HloOpcode::kConstant)
    return kOpaque;
  switch(instruction->opcode()) {
    case HloOpcode::kBroadcast:
      return kBroadcast;
    case HloOpcode::kConcatenate:
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
    case HloOpcode::kPad:
      return kInjective;
    case HloOpcode::kReduce:
      return kCommReduce;
    case HloOpcode::kConvolution:
      return kOutEWiseFusable;
  }
  if (instruction->IsElementwise())
    return kElemWise;
  return kOpaque;
}

bool NewFusion::IsLegalToFuse(HloInstruction* inst1, HloInstruction* inst2, bool MultiOutput)
{
  if (MultiOutput)
    return ShapesCompatibleForMultiOutputFusion(*inst1, *inst2);
  else
    return true;
}

int NewFusion::GetFusionCost(HloInstruction* inst1, HloInstruction* inst2)
{
  return 0;
}

NewFusion::NodeType NewFusion::MergeIntoConsumers(NewFusion::NodeType instruction)
{
  // aim: to prevent GTE limitation

  if (instruction->user_count() == 0 || !instruction->IsFusible())
    return instruction;

  // first fuse all consumers together
  HloInstruction* merged = NULL;
  for (auto it : instruction->users()) {
    if (merged == NULL) {
      merged = it;
    } else {
      merged = Fuse(it, merged);
    }
  }
  
  // fuse producer into fused consumer
  return MergeIntoConsumer(instruction, merged, false);
}

NewFusion::NodeType NewFusion::Merge(NewFusion::NodeType inst1, NewFusion::NodeType inst2, bool Duplicate, bool ProducerConsumer)
{
  if (ProducerConsumer) {
    return MergeIntoConsumer(inst1, inst2, Duplicate);
  } else {
    // no relation between nodes being fused: perform multioutput fusion
    return FuseIntoMultiOutput(inst1, inst2);
  }
  return NULL;
}

HloInstruction* NewFusion::MergeIntoConsumer(HloInstruction* producer, HloInstruction* consumer, bool Duplicate)
{
  if (consumer->opcode() != HloOpcode::kFusion) {
    auto fKind = ChooseKind(producer, consumer);

    HloInstruction* fusion =
        computation->AddInstruction(HloInstruction::CreateFusion(
            consumer->shape(), fKind, consumer));
    VLOG(2) << "Fuse producer " << producer->name() << " and its consumer "
            << consumer->name() << " into " << fusion->name();
    TF_CHECK_OK(computation->ReplaceInstruction(consumer, fusion));
    if (producer->opcode() == HloOpcode::kFusion) {
      if (Duplicate) {
        fusion->MergeFusionInstruction(producer);
      } else {
        fusion->MergeFusionInstructionIntoMultiOutput(producer);
      }
    } else {
      if (Duplicate) {
        fusion->FuseInstruction(producer);
      } else {
        fusion->FuseInstructionIntoMultiOutput(producer);
      }
    }
    if (producer->user_count() == 0)
      computation->RemoveInstruction(producer);
    return fusion;
  } else {
    VLOG(2) << "Fuse producer " << producer->name() << " into its consumer "
            << consumer->name();
    if (producer->opcode() == HloOpcode::kFusion) {
      if (Duplicate) {
        consumer->MergeFusionInstruction(producer);
      } else {
        consumer->MergeFusionInstructionIntoMultiOutput(producer);
      }
    } else {
      if (Duplicate) {
        consumer->FuseInstruction(producer);
      } else {
        consumer->FuseInstructionIntoMultiOutput(producer);
      }
    }
    if (producer->user_count() == 0)
      computation->RemoveInstruction(producer);
    return consumer;
  }
}

StatusOr<bool> NewFusion::Run(HloModule* module) {
  VLOG(2) << "Before new fusion:";
  XLA_VLOG_LINES(2, module->ToString());

  for (auto* computation : module->computations()) {
    NewFusion fusion;
    fusion.computation = computation;
    fusion.runFusion();
    fusion.doMerge();
  }
  return false;
}
}
}
