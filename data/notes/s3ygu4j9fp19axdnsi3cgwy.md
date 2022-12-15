
### Graph Kernels

Graph kernels are used to compute the similarity between two graphs. $\phi$ is used to map $G$ to a different space then similarity is computed in that space as the inner product $\phi({G})^{\top}\phi({G^{\prime}})$. Graph kernels are one of the traditional methods used for learning on graphs.

### Weisfeiler-Lehman(WL) Kernel [^1]

The Weisfeiler-Lehman (WL) Kernel is a popular kernel for its time complexity and because of its expressiveness which is the ability to distinguish a large number of different types of graphs. WL Kernel computes computes $\phi$ via the color refinement algorithm.

[^1]: [Stanford CS224W Lecture 2.3](https://www.youtube.com/watch?v=buzsHTa4Hgs)

> **Color Refinement Algorithm**:
>
> - Given a graph $G$ with a set of nodes $V$
>     - Assign an initial color $c^{(0)}(v)$ to each node $v$
>     - Iteratively refine node colors by
>       - $C^{(k+1)}(v)=\operatorname{HASH}\left(\left\{c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u \in N(v)}\right\}\right)$
>
> - $\operatorname{HASH}$ maps different inputs to different **colors**

After $K$ steps of color refinement, $c^{K}(v)$ summarizes the structure of the $K$-hope neighborhood.

The $\phi$ from the WL kernel counts the number of nodes of a given color. $\phi$ represents a graph with a bag of colors. In more detail the kernel uses a generalized version of "bag of node degrees" since the node degrees are one-hop neighborhood information.

The WL kernel provides a strong benchmark for the development of GNNs since it is both computationally efficient and expressive. The cost to compute the WL kernel is linear in the number of edges since at each step information is aggregated at each node from its neighbors. Counting colors is linear w.r.t. the number of nodes, but since there are typically an order magnitude or more edges than nodes in real world graphs, the WL kernel time complexity is linear in the number of edges. Since we are discussing the kernel this is number of edges in both graphs.

<!-- CHECK above "but since there are typically an order magnitude or more edges than nodes in real world graphs" -->

### WL Graph Isomorphism Test

<!-- CHECK -->
The WL Graph Isomorphism Test (WL test) only guarantees that two are graphs not isomorophic if $\phi{(G)}^{\top}\phi{(G)} \neq \phi{(G)}^{\top}\phi{(G^{\prime})}$, at any stage during color refinement. In general the WL test can give a measure of similarity or closeness to isomorphism between two graphs.


### WL Test Toy Example

![](./assets/drawio/WL-test.drawio.png)

As we can see in the example different colors capture different $k$-hop neighborhood structure. In this case $\phi({G})^{\top}\phi({G^{\prime}})=36$. We know from $\phi({G})^{\top}\phi({G})$ and $\phi({G^{\prime}})^{\top}\phi({G^{\prime}})$, that the two graphs are not isomorphic. In fact, we could have seen that these graphs were different after the first stage of color refinement.

<!-- TODO fix this since the definition of the test is at any point discontinue -->
### Graph Isomorphism Network

In How Powerful are Graph Neural Networks, the GIN is proven to be at least as expressive as the WL test, whereas the GCN is shown to worse than the WL test in some cases[^2].

[^2]:[How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826.pdf)
