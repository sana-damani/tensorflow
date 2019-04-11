#include "tensorflow/compiler/xla/service/gpu/fusion_library.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"

namespace xla {
namespace gpu {

NewFusion::NodeType NewFusion::getRoot(bool RPO)
{
  absl::Span<HloInstruction* const> post_order = computation->MakeInstructionPostOrder();

  if (RPO) {
    return post_order[post_order.size() - 1];
  } else {
    return post_order[0];
  }
}

OpPatternKind NewFusion::getPatternKind(NewFusion::NodeType instruction)
{
  if (instruction->IsElementwise())
    return kElemWise;
  if (!IsFusible(*instruction) || ImplementedAsLibraryCall(*instruction))
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
      // injective
      return kInjective;
    case HloOpcode::kReduce:
      return kCommReduce;
    case HloOpcode::kConvolution:
      return kOutEWiseFusable;
  }
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

HloInstruction* NewFusion::MergeIntoConsumer(HloInstruction* inst1, HloInstruction* inst2, bool Duplicate)
{
  if (GetNumConsumers(inst1) == 1) {
      return Fuse(inst1, inst2);
  } else {
    if (Duplicate) {
      return Fuse(inst1, inst2);
    } else {
      return FuseIntoMultiOutput(inst1, inst2);
    }
  }
  return NULL;
}

StatusOr<bool> NewFusion::Run(HloModule* module) {
  VLOG(2) << "Before new fusion:";
  XLA_VLOG_LINES(2, module->ToString());

  for (auto* computation : module->computations()) {
    NewFusion fusion;
    fusion.computation = computation;
//    fusion.runFusion();
//    fusion.doMerge();
  }
  return Status::OK();
}
}
}
