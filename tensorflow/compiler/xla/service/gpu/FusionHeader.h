#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include <queue>
#include <map>

std::ofstream debugDumpfile;
using namespace std;

#define DBGPRINT( var ) \
  (debugDumpfile) << "DBG: " << __FUNCTION__ << "(" << __LINE__ << ") "\
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

template <class T>
class FusionPass {
public:
  using NodeType = T*;//This is the type of IR nodes, and We donot want to modify the IR
  std::map<NodeType, NodeType> ParentOfNode;  //A map of node to its parent, naive implementation
  using SetOfNodes = std::set<NodeType>;
  using VectorOfNodes = std::vector<NodeType>;
  using QofNodes = std::queue<NodeType>;
  FusionPass(){
      debugDumpfile.open ("FusionHeaderDump.txt", std::ios_base::app);
  }
  ~FusionPass(){
      debugDumpfile.close();
  }

  virtual string getString(NodeType inst)=0; //IR specific, get instruction as string
  virtual NodeType getRoot(bool RPO=true)=0; //IR specific, get the root/exit node, Add a boolean flag
  virtual NodeType Merge(NodeType inst1, NodeType inst2, bool Duplicate=false, bool ProducerConsumer=true) = 0;
  virtual NodeType MergeIntoConsumers(NodeType inst) = 0;
  virtual OpPatternKind getPatternKind(NodeType inst) = 0;
  virtual void GetConsumers(NodeType instruction, SetOfNodes& consumers) = 0;

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
    DBGPRINT(Op1);
    auto CanFuseFlag = isOpPatternKindsFusible(Op1,Op2);
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
    string fuseparent = getString(FuseParent);
    DBGPRINT(fuseparent);
    //DBGPRINTSET(RestOfNodes);
    FusedNodesMap[N] = RestOfNodes;
    for (auto FuseChild : RestOfNodes) {
      string fusechild = getString(FuseChild);
      DBGPRINT(fusechild);
      unionGroups(N,FuseChild);
    }
  }

  void runFusion() {
    //get the root node
    NodeType RootNode = getRoot();
    SetOfNodes AlreadyVisitedSet;
    QofNodes Qnodes; Qnodes.push(RootNode);

    while (!Qnodes.empty()) {
      auto Node = Qnodes.front(); Qnodes.pop();
      string node = getString(Node);
      DBGPRINT(node);
      if (Node == NULL) break;
      auto ParentNode = findGroupParent(Node);
      DBGPRINT("Try to fuse");
      string parentnode = getString(ParentNode);
      DBGPRINT(parentnode);
      SetOfNodes Consumers;
      GetConsumers(Node, Consumers);
      //DBGPRINTSET(Consumers);
      bool canFuse = true ;
      //check if Node can be fused with all its consumers
      for (auto ConsNode : Consumers) {
        Qnodes.push(ConsNode);
        string cons = getString(ConsNode);
        DBGPRINT(cons);
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
        fuseAllNodes(Node, Consumers);
    }
  }
  void doMerge(){
    std::map<NodeType, NodeType> OldNodes_MergedNodeMap;
    for (auto Iter : FusedNodesMap){
      auto ParentNode = Iter.first;
      if (OldNodes_MergedNodeMap.find(ParentNode) != OldNodes_MergedNodeMap.end()) {
        continue;
        ParentNode = OldNodes_MergedNodeMap[ParentNode];
      }
      string parentnode = getString(ParentNode);
      DBGPRINT(parentnode);
      for (auto ConsNode: Iter.second){
        string consnode = getString(ConsNode);
        DBGPRINT(consnode);
        if (OldNodes_MergedNodeMap.find(ConsNode) != OldNodes_MergedNodeMap.end()) {
          ConsNode = OldNodes_MergedNodeMap[ConsNode];
        }
      }
      // fuse into all consumers
      auto MergedNode = MergeIntoConsumers(ParentNode);
      string mergednode = getString(MergedNode);
      DBGPRINT(mergednode);
      // update map
      OldNodes_MergedNodeMap[ParentNode] = MergedNode;
      for (auto ConsNode: Iter.second){
        OldNodes_MergedNodeMap[ConsNode] = MergedNode;
      }
    }
  }
};

