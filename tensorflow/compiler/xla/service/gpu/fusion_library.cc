#include "tensorflow/compiler/xla/service/gpu/fusion_library.h"

HloInstruction& NewFusion::GetRoot(bool RPO=false)
{
  absl::Span<HloInstruction* const> post_order = computation->MakeInstructionPostOrder();

  if (RPO) {
    return post_order[post_order.size() - 1];
  } else {
    return post_order[0];
  }
}

int NewFusion::GetMapping(const HloInstruction& instruction)
{
  if (instruction.IsElementwise())
    return 0;
  if (!IsFusible(instruction) || ImplementedAsLibraryCall(instruction))
    return 5;
  switch(instruction.opcode) {
    case HloOpcode::kBroadcast:
      return 1;
    case HloOpcode::kConcatenate:
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
    case HloOpcode::kPad:
      // injective
      return 2;
    case HloOpcode::kReduce:
      return 3;
    case HloOpcode::kConvolution:
      return 4;
  }
  return 5;
}

bool NewFusion::IsLegalToFuse(const HloInstruction& inst1, const HloInstruction& inst2, bool MultiOutput)
{
  if (MultiOutput)
    return ShapesCompatibleForMultiOutputFusion(inst1, inst2);
  else
    return true;
}

int NewFusion::GetFusionCost(const HloInstruction& inst1, const HloInstruction& inst2)
{
  return 0;
}

HloInstruction& NewFusion::Merge(const HloInstruction& inst1, const HloInstruction& inst2, bool Duplicate, bool ProducerConsumer)
{
  HloInstruction &fusion_instruction;

  if (ProducerConsumer) {
    MergeIntoConsumer(inst1, inst2, Duplicate);
  } else {
    // no relation between nodes being fused: perform multioutput fusion
    fusion_instruction = FuseIntoMultiOutput(inst1, inst2);
  }
}

HloInstruction& NewFusion::MergeIntoConsumer(const HloInstruction& inst1, const HloInstruction& inst2, bool Duplicate)
{
  HloInstruction &fusion_instruction;

  if (GetNumConsumers(inst1) == 1) {
      fusion_instruction = Fuse(inst1, inst2);
  } else {
    if (Duplicate) {
      fusion_instruction = Fuse(inst1, inst2);
    } else {
      fusion_instruction = FuseIntoMultiOutput(inst1, inst2);
    }
  }

  return fusion_instruction;
}
}

void RunNewFusion(HloModule* module) {
  VLOG(2) << "Before new fusion:";
  XLA_VLOG_LINES(2, module->ToString());

  for (int idx = 0; idx < module->computations.size(); idx++) {
    NewFusion<HloInstruction> fusion(module->computations[idx].get());
    fusion.Run();
  }
}

