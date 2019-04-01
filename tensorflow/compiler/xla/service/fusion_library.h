
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_LIBRARY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_LIBRARY_H_

#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/platform/macros.h"

HloInstruction& GetInit();

int GetNumSucc(const HloInstruction& instruction);

HloInstruction& GetSucc(const HloInstruction& instruction, int idx);

bool IsContractible(const HloInstruction& inst1, const HloInstruction& inst2);

int GetFusionCost(const HloInstruction& inst1, const HloInstruction& inst2);

HloInstruction& Merge(const HloInstruction& inst1, const HloInstruction& inst2, bool Duplicate=false);

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_LIBRARY_H_
