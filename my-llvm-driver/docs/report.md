## 必做部分
#### 1.思考题
> 在runSemiNCA函数Step 2计算NCA时，如何证明找到的 WIDomCandidate一定是Node[i].Semi和Node[i].Parent的最近共同祖先？  
> ```c++
> # Step 2 采用NCA方法，计算各个节点的直接支配节点
> 1  for(i : [2,NextDFSNum-1] ){
> 2		SDomNum = Node[i].Semi
> 3       WIDomCandidate = Node[i].IDom	#初始为当前节点的父亲
> 4       while (Node[WIDomCandidate].DFSNum > SDomNum)
> 5       	WIDomCandidate = Node[WIDomCandidate].IDom       
> 6       Node[i].IDom = WIDomCandidate	
> 7   }  
>```

##### 预备引理

###### 引理1 
> Node[i].Semi.DFSNum < i.DFSNum

证明：`Node[i].Semi`是`Node[i]`的一个半必经路径的首点，由半必经路径定义，`Node[i].Semi.DFSNum < i.DFSNum`
###### 引理2 
> Node[i].iDom.DFSNum < Node[i].DFSNum
 
证明：由支配节点定义，从流图入口到达`Node[i]`的每条路径都会经过`Node[i].iDom`，因此在DFS先序遍历时，第一次到达`Node[i]`并编号前一定已经到达过`Node[i].iDom`并编号，因此`Node[i].iDom.DFSNum < Node[i].DFSNum`  
类似地，我们也可以用如下方式表述：
支配树上的任意一条边$x\rightarrow y$，有`x.DFSNum<y.DFSNum`
###### 引理3 
> Node[i].Parent.DFSNum $\geq$ Node[i].Semi.DFSNum
 
证明：由于在CFG图上，`i.Parent->i`也是一条半必经路径，其起点为`i.Parnet`，因此`i.semi`作为半必经路径DFSNum最小的首点，一定有`Node[i].Parent.DFSNum $\geq$ Node[i].Semi.DFSNum`。

##### 证明  
> TODO:好像有关于semi和sdom符号指代问题可能需要修改  

记`Node[i].Semi = u,Node[i].Parent=v`, 约定`u>v`表示`u.DFSNum`>`v.DFSNum`, 记$f(x)=x.iDom,f^{(n)}=f*f^{(n-1)}$证明如下:

###### 1. 若`WIDomCandidate==u`  

①当$u=v$时，显然有`WIDomCandidate`是两者在支配树上的最近共同祖先。  

②当$u\neq v$时，由`iDom`定义，有`Node[i].iDom`是`Node[i]`在支配树上的`parent`，此时`Node[i].iDom`<`Node[i]`的节点已经在支配树上构建。由于`v `> `u`，必然会至少进入line4的while循环一次，由于`WIDomCandidate`的初始值为`v`，则$u = f^{(n)}(v),(n\geq 1)$,对应支配树上一条路径$u= w_0 \rightarrow \cdots \rightarrow w_n = v$,因此u是v在支配树上的祖先，自然有u是u和v在支配树上的最近共同祖先。    

###### 2. 若`WIDomCandidate!=u`  

由情况1，此时在支配树上不存在从u到v的路径，由于支配树上任意非根节点都有根节点作为共同祖先，因此在支配树上必定存在u和v的共同祖先，记最近共同祖先为x，即支配树上存在路径1从x到u，路径2从x到v。  

下面用反证法证明路径2中不存在非x的小于u的节点。

记路径2为$x = w_0 \rightarrow \cdots \rightarrow w_n = v$,假设存在$w_i$, 有$w_i$<`u`，不妨设该节点为$w_1$，特别地，假设$w_1<u<w_2$。由于u和$w_1$在支配树上以x为根节点的子树的不同分支上，此时在CFG上存在不经过$w_1$的从x到u的路径3，同样地，CFG上存在不经过u的从$w_1$到$w_2$的路径4。

若路径4是DFS时先搜索到的路径，由定义有$w_2$<`u`，与假设矛盾，因此在实际DFS次序中，发现$w_1$后，先搜索的是其他路径，在这种情况下，$w_1$到$w_2$路径上的中间节点序号均大于$w_2$,因此$w_1$是$w_2$的一个半支配节点，所以$w_2$.semi < $w_1$。 同时由上面的分析，为了保证$w_1<u<w_2$至少还存在一条经过u的从$w_1$到$w_2$的路径5是DFS时第一次发现$w_2$时所走的路径。因此$w_2$.parent的直接支配节点一定为u或u在支配树中的祖先。  

由于`WIDomCandidate!=u`，因此`WIDomCandidate<u`因此在前一步构建支配树中的$w_2$时，由于$w_2$.semi  $\leq w_1$, $w_2$.parent $\leq$ `u`，由假设，存在直接路径$x\rightarrow u,x\rightarrow w_1$,因此`w2.WIDomCandidate` $\leq$ `WIDomCandidate(u,w1)`=x。因此$w_2$在支配树上的parent$\leq$x，与假设中$w_2$在支配树上从x到v的一条路径中不符，假设不成立。  

因此支配树上x到v的路径中不存在小于v的中间节点，因此最终退出while循环的`WIDomCandidate`一定为x，即u和v在支配树上的最近公共祖先。 


综上，计算NCA时找到的`WIDomCandidate`一定是`Node[i].Semi`和`Node[i].Parent`的最近共同祖先，证毕。