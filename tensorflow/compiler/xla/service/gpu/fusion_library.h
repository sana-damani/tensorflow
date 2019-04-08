#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"

class NewFusion:LibraryFusion {

HloComputation computation;

public:

NewFusion(HloComputation& computation, int arch) {this.computation = computation;}

HloInstruction& GetRoot(bool RPO=false);

bool SupportsDuplication() {return true;}

bool SupportsMultiOutputFusion() {return true;}

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

int GetMapping(const HloInstruction& instruction);

bool IsLegalToFuse(const HloInstruction& inst1, const HloInstruction& inst2, bool MultiOutput = true);

int GetFusionCost(const HloInstruction& inst1, const HloInstruction& inst2);

HloInstruction& Merge(const HloInstruction& inst1, const HloInstruction& inst2, bool Duplicate, bool ProducerConsumer);

HloInstruction& MergeIntoConsumer(const HloInstruction& inst1, const HloInstruction& inst2, bool Duplicate);

}
