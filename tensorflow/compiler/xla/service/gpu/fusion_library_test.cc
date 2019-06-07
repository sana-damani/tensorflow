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
// Test 0, elementwise to broadcast to reduce, all fusible
//                        
//                         
//          elementwise     elementwise 
//                 \           |
//                  \        broadcast 
//                   \       /
//                    reduce
//
// Tests that broadcasts fused into a fusion with a reduce root.
TEST_F(NewFusionTest, T0_BroadcastIntoReduce) {
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
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  HloInstruction* root = module->entry_computation()->root_instruction();
  //EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}
           
// Tests Opaque Consumer with 2 injective Inputs, cannot be fused 
//
//                        
//                         
//          Injective1      Injective2
//                 \      /
//                  Opaque
//
TEST_F(NewFusionTest, T1_Injective2Opaque_ReshapeIntoDot) {

  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY Injective2Opaque_ReshapeIntoDot {
      arg0 = s32[1,2,1]{2,1,0} parameter(0)
      reshape.lhs = s32[2,1]{1,0} reshape(arg0)
      arg1 = s32[1,2,1]{2,1,0} parameter(1)
      reshape.rhs = s32[2,1]{1,0} reshape(arg1)
      ROOT dot = s32[1,1]{1,0} dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  HloInstruction* root = module->entry_computation()->root_instruction();
  // EXPECT_THAT(root, op::Dot());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}

// Tests 1 Opaque Consumer and 1 Elemwise with 2 injective Inputs, duplication required
//                        
//                         
//          Injective1 Injective2
//               /  \  /  \
//              /    \/    \
//             /     /\     \
//            elemwise Opaque
//
TEST_F(NewFusionTest, T2_duplication_Injective2Opaque) {

  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY Injective2Opaque_ReshapeIntoDot {
      arg0 = s32[1,2,1]{2,1,0} parameter(0)
      reshape.lhs = s32[2,1]{1,0} reshape(arg0)
      arg1 = s32[1,2,1]{2,1,0} parameter(1)
      reshape.rhs = s32[2,1]{1,0} reshape(arg1)
      mul = s32[2,1]{1,0} multiply(reshape.lhs,reshape.rhs)
      ROOT dot = s32[1,1]{1,0} dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  HloInstruction* root = module->entry_computation()->root_instruction();
  // EXPECT_THAT(root, op::Dot());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}
// Tests Elemwise Consumer with 2 injective Inputs,  everything can be fused
//
//                        
//                         
//          Injective1      Injective2
//            |    \      /
//            |    Elemwise
//            |        |
//       Injective3  Injective4
//             \       |
//              Elemwise

//              Sample output from GpuInstructionFusion

//%fused_computation (param_0.1: s32[2,1], param_1.3: s32[2,1]) -> s32[1,2] {
//  %param_0.1 = s32[2,1]{1,0} parameter(0)
//  %inj3.1 = s32[1,2]{1,0} reshape(s32[2,1]{1,0} %param_0.1)
//  %param_1.3 = s32[2,1]{1,0} parameter(1)
//  %multX.1 = s32[2,1]{1,0} multiply(s32[2,1]{1,0} %param_0.1, s32[2,1]{1,0} %param_1.3)
//  %inj4.1 = s32[1,2]{1,0} reshape(s32[2,1]{1,0} %multX.1)
//  ROOT %finalM.1 = s32[1,2]{1,0} multiply(s32[1,2]{1,0} %inj3.1, s32[1,2]{1,0} %inj4.1)
//}
//
//ENTRY %Injective2Opaque_ReshapeIntoDot (arg0: s32[1,2,1], arg1: s32[1,2,1]) -> s32[1,2] {
//  %arg0 = s32[1,2,1]{2,1,0} parameter(0)
//  %reshape.lhs = s32[2,1]{1,0} reshape(s32[1,2,1]{2,1,0} %arg0)
//  %arg1 = s32[1,2,1]{2,1,0} parameter(1)
//  %reshape.rhs = s32[2,1]{1,0} reshape(s32[1,2,1]{2,1,0} %arg1)
//  ROOT %fusion = s32[1,2]{1,0} fusion(s32[2,1]{1,0} %reshape.lhs, s32[2,1]{1,0} %reshape.rhs), kind=kLoop, calls=%fused_computation
//}
TEST_F(NewFusionTest, T3_Injective2Elemwise) {

  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY Injective2Opaque_ReshapeIntoDot {
      arg0 = s32[1,2,1]{2,1,0} parameter(0)
      reshape.lhs = s32[2,1]{1,0} reshape(arg0)
      arg1 = s32[1,2,1]{2,1,0} parameter(1)
      reshape.rhs = s32[2,1]{1,0} reshape(arg1)
      multX = s32[2,1]{1,0} multiply(reshape.lhs, reshape.rhs)
      inj4 = s32[1,2]{1,0} reshape(multX)
      inj3 = s32[1,2]{1,0} reshape(reshape.lhs )
      ROOT finalM = s32[1,2]{1,0} multiply(inj3, inj4)
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  // TODO: Enable this after the crash is fixed
  //EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  HloInstruction* root = module->entry_computation()->root_instruction();
  // EXPECT_THAT(root, op::Dot());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}
// Tests ElemWise producers and Elemwise consumers , 3 nodes Fusible ?
//
//        Opaque Opaque  Opaque Opaque              
//            \  /           \  /
//          Elemwise        Elemwise 
//                 \      /
//                Elemwise
//

TEST_F(NewFusionTest, T4_ElemWiseIntoElemwise_MultiplyIntoAdd) {

  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY ElemWiseIntoElemwise_MultiplyIntoAdd {
      p0 = s32[8] parameter(0)
      p1 = s32[8] parameter(1)
      p2 = s32[8] parameter(2)
      x = s32[8] multiply(p0, p2)
      y = s32[8] multiply(p1, p2)
      ROOT sum = s32[8] add(x, y)

    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Add());
}

// Tests 2 level Fusion, Elemwise into Broadcast and Broadcast into ElemwiseA, 3 levels, no dupilcation
//
//                       ElemWise              
//                           |
//          Opaque       Broadcast 
//                 \      /
//                  Elemwise
//
//%fused_computation (param_0: s32[4], param_1.1: s32[]) -> s32[4] {
//  %param_0 = s32[4]{0} parameter(0)
//  %param_1.1 = s32[] parameter(1)
//  %b.1 = s32[4]{0} broadcast(s32[] %param_1.1), dimensions={}
//  ROOT %d.1 = s32[4]{0} divide(s32[4]{0} %param_0, s32[4]{0} %b.1)
//}
//
//%fused_computation.1 () -> (s32[4], s32[]) {
//  %c.1 = s32[] constant(8)
//  %b.2 = s32[4]{0} broadcast(s32[] %c.1), dimensions={}
//  ROOT %tuple = (s32[4]{0}, s32[]) tuple(s32[4]{0} %b.2, s32[] %c.1)
//}
//
//ENTRY %BroadcastIntoDivide (p: s32[4]) -> s32[4] {
//  %c = s32[] constant(8)
//  %p = s32[4]{0} parameter(0)
//  %fusion.1 = (s32[4]{0}, s32[]) fusion(), kind=kLoop, calls=%fused_computation.1
//  %get-tuple-element.1 = s32[] get-tuple-element((s32[4]{0}, s32[]) %fusion.1), index=1
//  ROOT %fusion = s32[4]{0} fusion(s32[4]{0} %p, s32[] %get-tuple-element.1), kind=kLoop, calls=%fused_computation
//  %get-tuple-element = s32[4]{0} get-tuple-element((s32[4]{0}, s32[]) %fusion.1), index=0
//}
//
TEST_F(NewFusionTest, T5_Elem2Broad2Elem_BroadcastIntoDivide) {

  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY Elem2Broad2Elem_BroadcastIntoDivide {
      p = s32[4] parameter(0)
      c = s32[] constant(8)
      b = s32[4] broadcast(c), dimensions={}
      ROOT d = s32[4] divide(p, b)
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  //HloInstruction* root = module->entry_computation()->root_instruction();
  //EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}

// Except opaque all fusble, No duplication !
//
//                       ElemWise              
//                           |
//          Opaque       Broadcast 
//                 \      /    |
//                  Elemwise   |
//                   \         /
//                     Elemwise
//

TEST_F(NewFusionTest, T6_Elem2Broad2Elem_BroadcastIntoDivide) {

  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY Elem2Broad2Elem_BroadcastIntoDivide {
      p = s32[4] parameter(0)
      c = s32[] constant(8)
      b = s32[4] broadcast(c), dimensions={}
      d = s32[4] divide(p, b)
      ROOT s = s32[4] add(p, d)
    })")
                    .ValueOrDie();

      //ROOT x = s32[4] add(d, s)
  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  //HloInstruction* root = module->entry_computation()->root_instruction();
  //EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}
// Tests 5, elemwise into reduce , with multuple producers,  
//                            
//       Opaque  Opaque elemwise elemwise 
//             \   \      /       /
//                  Reduce
//%fused_computation (param_0: f32[0,10,10], param_1: f32[0,10,10]) -> (f32[10,10], f32[10,10]) {
//  %param_0 = f32[0,10,10]{2,1,0} parameter(0)
//  %param_1 = f32[0,10,10]{2,1,0} parameter(1)
//  %constant.2 = f32[] constant(0)
//  %constant.3 = f32[] constant(1)
//  ROOT %reduce.1 = (f32[10,10]{1,0}, f32[10,10]{1,0}) reduce(f32[0,10,10]{2,1,0} %param_0, f32[0,10,10]{2,1,0} %param_1, f32[] %constant.2, f32[] %constant.3), dimensions={0}, to_apply=%reducer
//}
//
//ENTRY %GTEintoReduce (parameter.6: (f32[0,10,10], f32[0,10,10])) -> (f32[10,10], f32[10,10]) {
//  %constant.0 = f32[] constant(0)
//  %constant.1 = f32[] constant(1)
//  %parameter.6 = (f32[0,10,10]{2,1,0}, f32[0,10,10]{2,1,0}) parameter(0)
//  %get-tuple-element.10 = f32[0,10,10]{2,1,0} get-tuple-element((f32[0,10,10]{2,1,0}, f32[0,10,10]{2,1,0}) %parameter.6), index=0
//  %get-tuple-element.11 = f32[0,10,10]{2,1,0} get-tuple-element((f32[0,10,10]{2,1,0}, f32[0,10,10]{2,1,0}) %parameter.6), index=1
//  ROOT %fusion = (f32[10,10]{1,0}, f32[10,10]{1,0}) fusion(f32[0,10,10]{2,1,0} %get-tuple-element.10, f32[0,10,10]{2,1,0} %get-tuple-element.11), kind=kInput, calls=%fused_computation
//}
TEST_F(NewFusionTest, T7_multipleProds_elemwise2reduce_GTEintoReduce) {

  auto module = ParseHloString(R"(
    HloModule test_module

    reducer {
      parameter.1 = f32[] parameter(0)
      parameter.3 = f32[] parameter(2)
      mul.2 = f32[] add(parameter.1, parameter.3)
      parameter.0 = f32[] parameter(1)
      parameter.2 = f32[] parameter(3)
      add.3 = f32[] add(parameter.0, parameter.2)
      ROOT tuple.4 = (f32[], f32[]) tuple(mul.2, add.3)
    }

    ENTRY multipleProds_elemwise2reduce_GTEintoReduce {
      parameter.6 = (f32[0, 10, 10], f32[0, 10, 10]) parameter(0)
      get-tuple-element.10 = f32[0, 10, 10] get-tuple-element(parameter.6), index=0
      get-tuple-element.11 = f32[0, 10, 10] get-tuple-element(parameter.6), index=1
      constant.0 = f32[] constant(0)
      constant.1 = f32[] constant(1)
      ROOT reduce = (f32[10, 10], f32[10, 10]) reduce(get-tuple-element.10, get-tuple-element.11, constant.0, constant.1), dimensions={0}, to_apply=reducer

    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  //HloInstruction* root = module->entry_computation()->root_instruction();
  //EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}

// Tests Except opaque all fusible, no duplication ! 
//                            
//       Opaque   elemwise elemwise 
//         \         |        |
//          \     Broadcast Broadcast
//           \       |    /
//             \     |  /
//                \  | /
//                  Reduce
TEST_F(NewFusionTest, T8_multipleProds_elemwise2reduce_GTEintoReduce) {

  auto module = ParseHloString(R"(
    HloModule test_module

    reducer {
      parameter.1 = f32[] parameter(0)
      parameter.3 = f32[] parameter(2)
      mul.2 = f32[] add(parameter.1, parameter.3)
      parameter.0 = f32[] parameter(1)
      parameter.2 = f32[] parameter(3)
      add.3 = f32[] add(parameter.0, parameter.2)
      ROOT tuple.4 = (f32[], f32[]) tuple(mul.2, add.3)
    }

    ENTRY multipleProds_elemwise2reduce_GTEintoReduce {
      parameter.6 = (f32[0, 10, 10], f32[0, 10, 10]) parameter(0)
      constant.0 = f32[] constant(0)
      constant.1 = f32[] constant(1)
      consBr1 = f32[10,10] broadcast(constant.1), dimensions={}
      consBr2 = f32[10,10] broadcast(constant.0), dimensions={}
      ROOT reduce = (f32[10, 10], f32[10, 10]) reduce(consBr1, consBr2, constant.0, constant.1), dimensions={0}, to_apply=reducer
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  //TODO: This Crashes
  //EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  //HloInstruction* root = module->entry_computation()->root_instruction();
  //EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}
// Tests 5, Duplication and Multioutput, elemwise-> duplicated, elemwise fused to outelem fused to elem
//
//                                             
//                            
//         Opaque            elemwise 
//            |              |    |                           
//          elemwise   Broadcast  Broadcast 
//             \       /            /
//         outElemWiseFusible      /
//                      \         /
//                       elemwise 
//
//%fused_computation (param_0.1: f32[1,1,2], param_1.1: f32[1,2,2]) -> f32[1,2,2] {
//  %param_1.1 = f32[1,2,2]{2,1,0} parameter(1)
//  %copy.1 = f32[1,2,2]{2,0,1} copy(f32[1,2,2]{2,1,0} %param_1.1)
//  %param_0.1 = f32[1,1,2]{2,1,0} parameter(0)
//  ROOT %convolution.1 = f32[1,2,2]{2,0,1} convolution(f32[1,2,2]{2,0,1} %copy.1, f32[1,1,2]{2,1,0} %param_0.1), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
//}
//
//ENTRY %Convolve1D1Window_0_module (input: f32[1,2,2], filter: f32[1,1,2]) -> f32[1,2,2] {
//  %input = f32[1,2,2]{2,1,0} parameter(0)
//  %copy = f32[1,2,2]{2,0,1} copy(f32[1,2,2]{2,1,0} %input)
//  %filter = f32[1,1,2]{2,1,0} parameter(1)
//  ROOT %fusion = f32[1,2,2]{2,0,1} fusion(f32[1,1,2]{2,1,0} %filter, f32[1,2,2]{2,1,0} %input), kind=kLoop, calls=%fused_computation
//}
TEST_F(NewFusionTest, T9_Elem2Conv_Convolve1D1Window_0_module) {

  auto module = ParseHloString(R"(
    HloModule test_module
    ENTRY Convolve1D1Window_0_module (input: f32[1,2,2]) -> f32[1,2,2] {
      %input = f32[1,2,2]{2,1,0} parameter(0)
      %copy1 = f32[1,2,2]{2,0,1} copy(f32[1,2,2]{2,1,0} %input)
      %cons1 = f32[] constant(1)
      %filter = f32[1,1,2]{2,1,0} broadcast(cons1), dimensions={}
      %convol = f32[1,2,2]{2,0,1} convolution(f32[1,2,2]{2,0,1} %copy1, f32[1,1,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
      %const2 = f32[1,2,2]{2,0,1} broadcast(cons1), dimensions={}
      ROOT %relu = f32[1,2,2] maximum(%const2, %convol)
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  //HloInstruction* root = module->entry_computation()->root_instruction();
  //EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}

// Tests 5, Duplication and Multioutput  
//
//                                             
//                            
//         Opaque            elemwise 
//            |               /     |                         
//          elemwise   Broadcast    Broadcast 
//             \       /           /
//         outElemWiseFusible     /
//              |       \        /
//              |        elemwise 
//              |          |
//           injective  injective
//                 \     /
//                 elemwise
TEST_F(NewFusionTest, T10_Elem2Conv_Convolve1D1Window_0_module) {

  auto module = ParseHloString(R"(
    HloModule test_module
    ENTRY Convolve1D1Window_0_module  {
      %input = f32[1,2,2]{2,1,0} parameter(0)
      %copy1 = f32[1,2,2]{2,0,1} copy(f32[1,2,2]{2,1,0} %input)
      %cons1 = f32[] constant(1)
      %filter = f32[1,1,2]{2,1,0} broadcast(cons1), dimensions={}
      %convol = f32[1,2,2]{2,0,1} convolution(f32[1,2,2]{2,0,1} %copy1, f32[1,1,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
      %const2 = f32[1,2,2]{2,0,1} broadcast(cons1), dimensions={}
      %relu = f32[1,2,2] maximum(%const2, %convol)
      %image2 = f32[2,2,1]{2,0,1} reshape(%relu)
      %filter2 = f32[2,1,1]{2,1,0} reshape(%filter)
      ROOT %convol2 = f32[2,2,1]{2,0,1} convolution( f32[2,2,1]{2,0,1} %image2, f32[2,1,1]{2,1,0} %filter2), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
    })")
                    .ValueOrDie();

  debugDumpfile.open ("fusion_library_test_log.txt", std::ios_base::app);
  string before = module->ToString();
  DBGPRINT(before);
  auto msg = "GpuInstructionFusion output:";
  DBGPRINT(msg);
  //EXPECT_TRUE(GpuInstructionFusion(true).Run(module.get()).ValueOrDie());
  msg = "Our Fusion Output:";
  DBGPRINT(msg);
  //TODO: This Crashes
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  //HloInstruction* root = module->entry_computation()->root_instruction();
  //EXPECT_THAT(root, op::Fusion());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}

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
  EXPECT_TRUE(NewFusion().Run(module.get()).ValueOrDie());

  string after = module->ToString();
  DBGPRINT(after);
  debugDumpfile.close();
  // HloInstruction* root = module->entry_computation()->root_instruction();
  // EXPECT_THAT(root, op::Dot());
  //EXPECT_THAT(root->fused_expression_root(),
  //            op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
