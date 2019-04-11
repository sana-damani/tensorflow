#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_LIBRARY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_LIBRARY_H_

#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/FusionHeader.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {

class NewFusion : public FusionPass<HloInstruction>, public GpuInstructionFusion {

public:

HloComputation *computation;

NewFusion() : GpuInstructionFusion(true) {}

absl::string_view name() const override { return "new fusion"; }

StatusOr<bool> Run(HloModule* module) override;

NodeType getRoot(bool RPO=false);

OpPatternKind getPatternKind(NodeType);

bool SupportsDuplication() {return true;}

bool SupportsMultiOutputFusion() {return true;}

int GetNumConsumers(HloInstruction* instruction)
{
  return instruction->user_count();
}

HloInstruction* GetConsumer(HloInstruction* instruction, int idx)
{
  return instruction->users()[idx];
}

int GetNumProducers(HloInstruction* instruction)
{
  return instruction->operand_count();
}

HloInstruction* GetProducer(HloInstruction* instruction, int idx)
{
  return instruction->operands()[idx];
}

bool IsLegalToFuse(HloInstruction* inst1, HloInstruction* inst2, bool MultiOutput = true);

int GetFusionCost(HloInstruction* inst1, HloInstruction* inst2);

NodeType Merge(NodeType inst1, NodeType inst2, bool Duplicate, bool ProducerConsumer);

HloInstruction* MergeIntoConsumer(HloInstruction* inst1, HloInstruction* inst2, bool Duplicate);

};
}
}
#endif
