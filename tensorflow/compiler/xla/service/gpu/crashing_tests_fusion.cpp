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

TEST_F(NewFusionTest, T2_Injective2Opaque_ReshapeIntoDot) {

  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY Injective2Opaque_ReshapeIntoDot {
      arg0 = s32[1,2,1]{2,1,0} parameter(0)
      reshape.rhs = s32[2,1]{1,0} reshape(arg0)
      ROOT inj3 = s32[1,2]{1,0} reshape(reshape.rhs )
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
 // EXPECT_TRUE(GpuInstructionFusion(true)
 //                 .Run(module.get())
 //                 .ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion()
                  .Run(module.get())
                  .ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  HloInstruction* root = module->entry_computation()->root_instruction();
  // EXPECT_THAT(root, op::Dot());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
