#include <iostream>
#include <ctime>
#include <fstream>
#include <set>
#include <string>
#include <queue>
#include <map>

std::ofstream debugDumpfile;
using namespace std;
#define PRINT_TIME (asctime(localtime(&getTime)))
#define DBGPRINT( var ) \
  (debugDumpfile) << "DBG: "<< __FUNCTION__ << "(" << __LINE__ << ") "\
       << #var << " = [" << (var) << "]" << std::endl
#define DBGPRINTSET( setVars ) \
  for (auto child : setVars ) {\
    DBGPRINT( child ); \
  }
   
enum OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // Broadcasting operator, can always map output axis to the input in order.
  // for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
  // Note that the axis need to be in order so transpose is not a bcast operator.
  kBroadcast = 1,
  // Injective operator, can always injectively map output axis to a single input axis.
  // All injective operator can still be safely fused to injective and reduction.
  kInjective = 2,
  // Communicative reduction operator.
  kCommReduce = 3,
  // Complex operation, can still fuse elemwise operations into its output.
  // but cannot chain another complex op
  kOutEWiseFusable = 4,
  // Opaque operation, cannot fuse anything.
  kOpaque = 8
};

time_t getTime;
template <class T>
class FusionPass {
public:
  using NodeType = T*;//This is the type of IR nodes, and We donot want to modify the IR
  std::map<NodeType, NodeType> ParentOfNode;  //A map of node to its parent, naive implementation
  using SetOfNodes = std::set<NodeType>;
  using VectorOfNodes = std::vector<NodeType>;
  using QofNodes = std::queue<NodeType>;
  FusionPass(){
      //debugDumpfile.open ("NewFusionHeaderDump.txt", std::ios_base::app);
      time (&getTime);
  }
  ~FusionPass(){
      //debugDumpfile.close();
  }

  virtual string getString(NodeType inst)=0; //IR specific, get instruction as string
  virtual NodeType getRoot(bool RPO=false)=0; //IR specific, get the root/exit node, Add a boolean flag
  virtual NodeType Merge(NodeType inst1, NodeType inst2, bool Duplicate=false, bool ProducerConsumer=true) = 0;
  virtual NodeType MergeIntoConsumers(NodeType inst) = 0;
  virtual OpPatternKind getPatternKind(NodeType inst) = 0;
  virtual void GetConsumers(NodeType instruction, SetOfNodes& consumers) = 0;
  virtual void GetProducers(NodeType instruction, SetOfNodes& producers) = 0;

  NodeType combinePattern(
      NodeType &X, NodeType &Y) {
    OpPatternKind XKind = getPatternKind(X);
    OpPatternKind YKind = getPatternKind(Y);
    if (XKind > YKind ) return X;
    return Y;
  }
  OpPatternKind combinePattern(
      OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > rhs) return lhs;
    return rhs;
  }

  bool isOpPatternKindsFusible(OpPatternKind Op1, OpPatternKind Op2){
    if (Op1 == kElemWise && 
        Op2 <= kCommReduce) 
      return true;
    if (Op1 == kBroadcast && 
        Op2 <= kCommReduce) 
      return true;
    if (Op1 == kInjective &&
        Op2 <= kInjective )
      return true;
    if (Op1 == kOutEWiseFusable &&
        Op2 == kElemWise )
      return true;
    return false;
  }

  bool cannotFuse(NodeType &X, NodeType &Y) {
    auto Op1 = getPatternKind(X);
    auto Op2 = getPatternKind(Y);
    DBGPRINT(Op1);
    DBGPRINT(Op2);
    auto CanFuseFlag = isOpPatternKindsFusible(Op1,Op2) || isOpPatternKindsFusible(Op2,Op1);
    DBGPRINT(CanFuseFlag);
    if (CanFuseFlag)
      return false;
    return true;
  }

  //The naive implementation of Find for union-find algorithm, recursively track the parent
  NodeType findGroupParent(NodeType N)
  {
    //if the node does not exist then it has no parent
    if (ParentOfNode.find(N) == ParentOfNode.end()) 
      return N; 
    return findGroupParent(ParentOfNode[N]); 
  }

  //A utility function to do union of two subsets  
  void unionGroups(NodeType X, NodeType Y) 
  { 
    NodeType XSet = findGroupParent(X); 
    NodeType YSet = findGroupParent(Y); 
    if(XSet!=YSet){ 
      if (XSet == combinePattern(XSet, YSet)) 
        ParentOfNode[YSet] = XSet; 
      else 
        ParentOfNode[XSet] = YSet; 
    }
  }
  std::map<NodeType, SetOfNodes> FusedNodesMap;

  //Fuse N with the set of Nodes
  void fuseAllNodes(NodeType N, SetOfNodes RestOfNodes){
    auto FuseParent = N;
    string FuseParentWithConsumersStr = getString(FuseParent);
    DBGPRINT(FuseParentWithConsumersStr);
    FusedNodesMap[N] = RestOfNodes;
    for (auto FuseChild : RestOfNodes) {
      string FuseChildWithParentStr = getString(FuseChild);
      DBGPRINT(FuseChildWithParentStr);
      unionGroups(N,FuseChild);
    }
  }

  bool runFusion() {
    //get the root node
    debugDumpfile.open ("NewFusionHeaderDump.txt", std::ios_base::app);
    NodeType RootNode = getRoot();
    string RootStr = getString(RootNode);
    DBGPRINT(RootStr);
    SetOfNodes AlreadyVisitedSet;
    QofNodes Qnodes; Qnodes.push(RootNode);
    bool DidFusion = false;

    while (!Qnodes.empty()) {
      auto Node = Qnodes.front(); Qnodes.pop();
      string NodeRBFSVisitStr = getString(Node);
      DBGPRINT(NodeRBFSVisitStr);
      if (Node == NULL) break;
      auto ParentNode = findGroupParent(Node);
      // if the Node already has a parent
      // Means its already fused
      //if (Node != ParentNode) continue;
      DBGPRINT("Try to fuse");
      string ParentNodefromUnionGroup = getString(ParentNode);
      DBGPRINT(ParentNodefromUnionGroup);
      SetOfNodes Consumers;
      GetConsumers(Node, Consumers);
      bool canFuse = true ;
      //check if Node can be fused with all its consumers
      for (auto ConsNode : Consumers) {
        string ConsNodeCheckifFusibleStr = getString(ConsNode);
        DBGPRINT(ConsNodeCheckifFusibleStr);
        auto ParentConsNode = findGroupParent(ConsNode);
        if (ParentConsNode == ParentNode) continue; //already in same group
        if (cannotFuse(ParentConsNode, ParentNode)) {
          auto Cannot_Fuse = ParentConsNode;
          DBGPRINT(Cannot_Fuse);
          canFuse = false;
          break;
        }
      }
      if (canFuse) //Fuse Node with all its consumers
      {
        DidFusion = true;
        fuseAllNodes(Node, Consumers);
      }
      //Now push all producers of this node into the Q
      SetOfNodes Producers;
      GetProducers(Node, Producers);
      for (auto P : Producers) 
        Qnodes.push(P);
    }
    return DidFusion;
    debugDumpfile.close();
  }
  void doMerge(){
    debugDumpfile.open ("NewFusionHeaderDump.txt", std::ios_base::app);
    std::map<NodeType, NodeType> OldNodes_MergedNodeMap;
    for (auto Iter : FusedNodesMap){
      auto ParentNode = Iter.first;
      if (OldNodes_MergedNodeMap.find(ParentNode) != OldNodes_MergedNodeMap.end()){
        // If this Node was already fused to its parent
        //continue;
        // Then consider the Fused Node
        ParentNode = OldNodes_MergedNodeMap[ParentNode];
      }
      string ParentnodeStr = getString(ParentNode);
      DBGPRINT(ParentnodeStr);
      // fuse into all consumers
      auto MergedNode = MergeIntoConsumers(ParentNode);
      string MergednodeStr = getString(MergedNode);
      DBGPRINT(MergednodeStr);
      // update map
      OldNodes_MergedNodeMap[ParentNode] = MergedNode;
      for (auto ConsNode: Iter.second){
        OldNodes_MergedNodeMap[ConsNode] = MergedNode;
      }
    }
    debugDumpfile.close();
  }
};

