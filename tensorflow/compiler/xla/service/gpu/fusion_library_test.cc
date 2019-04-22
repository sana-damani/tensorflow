/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_library.h"

#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

std::ofstream debugDumpfile;

namespace op = xla::testing::opcode_matchers;

using NewFusionTest = HloTestBase ;

// Tests that broadcasts fused into a fusion with a reduce root.
TEST_F(NewFusionTest, BroadcastIntoReduce) {
  auto module = ParseHloString(R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY BroadcastIntoReduce {
      constant = f32[] constant(1)
      broadcast = f32[16,16,16,16]{3,2,1,0} broadcast(constant), dimensions={}
      constant.1 = f32[] constant(0)
      ROOT reduce = f32[] reduce(broadcast, constant.1), dimensions={0,1,2,3},
                                                         to_apply=add
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
/*  EXPECT_TRUE(GpuInstructionFusion(true)
                  .Run(module.get())
                  .ValueOrDie());*/
  EXPECT_TRUE(NewFusion()
                  .Run(module.get())
                  .ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
