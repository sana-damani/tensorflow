#include <iostream>

#include <set>
#include <queue>
#include <map>

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

  virtual NodeType getRoot(bool RPO=false)=0; //IR specific, get the root/exit node, Add a boolean flag
  virtual NodeType Merge(NodeType inst1, NodeType inst2, bool Duplicate=false, bool ProducerConsumer=true) = 0;
  virtual OpPatternKind getPatternKind(NodeType inst) = 0;
  virtual int GetNumConsumers(NodeType instruction) = 0;
  virtual NodeType GetConsumer(NodeType instruction, int idx) = 0;

  SetOfNodes getConsumers(NodeType X ) { //IR specific, Get the successors of a node, 
    SetOfNodes Consumers ; 
    unsigned num = GetNumConsumers(X );
    for (unsigned i = 0 ; i < num ; i++){
      Consumers.insert(GetConsumer(X, i));
    }
    return Consumers;
  }

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
    if (isOpPatternKindsFusible(Op1,Op2))
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
    FusedNodesMap[N] = RestOfNodes;
    for (auto F : RestOfNodes) {
      unionGroups(N,F);
    }
  }

  void runFusion() {
    //get the root node
    NodeType RootNode = getRoot();
    SetOfNodes AlreadyVisitedSet;
    QofNodes Qnodes; Qnodes.push(RootNode);

    while (!Qnodes.empty()) {
      auto Node = Qnodes.front(); Qnodes.pop();
      auto ParentNode = findGroupParent(Node);
      SetOfNodes Consumers = getConsumers(Node);
      bool canFuse = true ;
      //check if Node can be fused with all its consumers
      for (auto ConsNode : Consumers) {
        Qnodes.push(ConsNode);
        auto ParentConsNode = findGroupParent(ConsNode);
        if (ParentConsNode == ParentNode) continue; //already in same group
        if (cannotFuse(ParentConsNode, ParentNode)) {
          canFuse = false;
          break;
        }
      }
      if (canFuse) //Fuse Node with all its consumers
        fuseAllNodes(Node, Consumers);
    }
  }
  void doMerge(){
    for (auto Iter : FusedNodesMap){
      auto ParentNode = Iter.first;
      for (auto ConsNode: Iter.second){
        Merge(ParentNode, ConsNode, true);
      }
    }
  }
};
