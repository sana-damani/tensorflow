#include "tensorflow/compiler/xla/service/fusion_library.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"

HloInstruction& GetInit(bool RPO=false)
{
  if (RPO) {
  } else {
  }
}

int GetNumConsumers(const HloInstruction& instruction)
{
  return instruction.user_count();
}

HloInstruction& GetConsumer(const HloInstruction& instruction, int idx)
{
  return instruction.users(idx);
}

int GetNumProducers(const HloInstruction& instruction)
{
  return instruction.operand_count();
}

HloInstruction& GetProducer(const HloInstruction& instruction, int idx)
{
  return instruction.operand(idx);
}

int GetLayoutKind(const HloInstruction& instruction)
{
}

bool IsLegalToFuse(const HloInstruction& instruction)
{
  return instruction.IsFusible();
}

int GetFusionCost(const HloInstruction& inst1, const HloInstruction& inst2)
{
  return 0;
}

HloInstruction& Merge(const HloInstruction& inst1, const HloInstruction& inst2, bool Duplicate)
{
  HloInstruction &fusion_instruction;

  if (Duplicate) {
    fusion_instruction = Fuse(inst1, inst2);
  } else {
    fusion_instruction = FuseIntoMultiOutput(inst1, inst2);
  }
}

HloInstruction& MergeIntoConsumer(const HloInstruction& inst1, const HloInstruction& inst2, bool Duplicate)
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
