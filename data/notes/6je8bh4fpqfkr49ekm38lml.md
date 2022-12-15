### [Message Passing Framework](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)
Traditionally machine learning over graphs has relied on graph statistics and kernel methods, the idea being to convert the graph in to a representation that can capture high level information about node neighbors and graph structure. These representations could then be used for a prediction task on the graph like a node or graph classification. Graph statistics and kernel methods are limited because they rely on hand engineered features and they don't adapt through the learning process. Additionally the handcrafted engineering of these features can be time consuming on reliant specialist domain knowledge. Instead the representation can be learned.

![](/assets/images/1-introduction.md.node-embedding-learning-WL-Hamilton.png)

The idea is to learn a mapping from nodes $u$ and $v$ to $z_u$ and $z_v$ (Image[^1]). A decoder $DEC$ then decodes the embeddings vectors according to a supervised or unsupervised objective. A loss can then be back propagated through the $DEC$ and then the $ENC$ to update model weights. One way to do this is through the message passing framework.

One of the major motivations for the message passing framework is the fact that graph data is non-Euclidean, which precludes graphs from being input to a Euclidean convolutional neural network. The message passing framework generalized the convolution to non-Euclidean data. Furthermore it can be motivated by the color refinement algorithm that is presented in the Weisfeiler-Lehman(WL) graph isomorphism test.

[^1]: [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/)

Here we present the message passing framework as presented by the popular [PyTorch Geometric Library](https://pytorch-geometric.readthedocs.io/en/latest/index.html). This definition of message passing will be used to define the different types of message passing neural networks.

> $$
> \mathbf{x}_i^{(k)}=\gamma^{(k)}\left(\mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)}, \mathbf{e}_{j, i}\right)\right)
> $$
>
> $\square_{j \in \mathcal{N}(i)}$ **:** Differentiable, permutation invariant function, *e.g.* sum, mean, or max.
>
> $\phi^{(k)}$ **:** Differentiable functions such as Multi-Layer Perceptrons (MLPs).
>
> $\gamma^{(k)}$ **:** Differentiable functions such MLPs.

### Toy Example

To gain some intuition of a message passing framework we will take a small graph as example.

![](./assets/drawio/Message-Passing-Framework.drawio.png)

We will consider messages passed to node $D$ in a 2-layer message passing neural network. The computational graph can be traced out from node $D$ first to a 1-hop neighborhood and then to a 2-hop neighborhood. The 1-hop neighborhood contains all nodes except node $A$. Including node $D$ is often considered optional and is can be interpreted as adding self-loops to the graph. Layer 0 looks shows connections from the 2-hop neighborhood of node $D$.

### Popular Message Passing Neural Networks

#### Graph Convolutional Neural Network ([GCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv))

**Message Passing View**
> $$
> \mathbf{x}_i^{\prime}=\boldsymbol{\Theta}^{\top} \sum_{j \in \mathcal{N}(v) \cup\{i\}} \frac{e_{j, i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
> $$
>
> $e_{i,j}$ **:** edge weight from node $j$ to node $i$ (i.e. with no edge weight, $e_{i,j} = \mathbf{1}$).
>
> $\hat{d}_{i} = 1 + \sum_{j \in \mathcal{N}(i)}$ **:** weighted degree of node $i$, plus 1 to avoid divide by 0.
>
> $\Theta^{\top}$ **:** Weight matrix

**Matrix View**
> $X^l = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}X^{(k-1)}\boldsymbol{\Theta})$
>
> $\hat{A} = A + I$ **:** Adjacency matrix with inserted self loops.
>
> $\hat{D}_{ij} = \sum_{j=0}\hat{A}_{ij}$ **:** Diagonal degree matrix.

The Graph Convolutional Neural (GCN) network was first published by Thomas Kipf and Max Welling at the international conference for learning representations (ICLR) in 2017 [^2]. This paper has had a tremendous impact on the development of graph neural networks and has sparked the design of more complicated and more expressive GNNs. To get an idea of the magnitude of impact, this paper currently has 14,095 citations whereas the famous transformer paper "Attention is all you need" has 59,941 citations.

[^2]: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

![](./assets/drawio/GCN.drawio.png)

<!-- TODO update to OG Conv -->

We can look at the first layer of the GCN to see how it passes messages. Each node vector is multiplied by a weight matrix $\boldsymbol{\Theta}^{\top}$ and then normalized by the product of degrees. This normalizing will prevent message vectors from exploding and it will remove degree bias.

####  Graph Attention Network ([GAT](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv))

**Message Passing View**
> $$
> \mathbf{x}_i^{\prime}=\alpha_{i, i} \boldsymbol{\Theta} \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \alpha_{i, j} \boldsymbol{\Theta} \mathbf{x}_j
> $$
>
> $$
> \alpha_{i, j}=\frac{\exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}^{\top}\left[\boldsymbol{\Theta} \mathbf{x}_i \| \boldsymbol{\Theta} \mathbf{x}_j\right]\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}^{\top}\left[\boldsymbol{\Theta} \mathbf{x}_i \| \boldsymbol{\Theta} \mathbf{x}_k\right]\right)\right)}
> $$
>
> $||$ - Concatenation
>
> $\Theta$ - Weight matrix. This is the same matrix in both $\mathbf{x}^{\prime}_i$ and $\alpha_{i,j}$ equations.
>
> $\mathbf{a}$ - Attention weights
>
> $\alpha$ - Attention weight matrix

**Matrix View**
> $X^l = \sigma(\alpha AX^{(k-1)}\boldsymbol{\Theta})$

<!-- CHECK is it theta transpose? check size. -->

The Graph Attention Network (GAT) was first published later in 2017 by Petar Veličković. The GAT looks just like the GCN but it replaces the normalizing factor with an attention mechanism. Intuitively the attention weights will amplify important edges and suppress unimportant edges for the given prediction task[^3].

![](./assets/drawio/GAT.drawio.png)

The attention mechanism was made famous in the transformer paper that used a pair wise attention across the entire input vector[^4]. Sometimes the attention mechanism applied to GAT is called masked attention, because it only considers edges within the underlying graph, "masking" or zeroing out all other attention coefficients[^3].

[^3]: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

[^4]: [Attention is All You Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)


Attention weights are computed as a softmax of learned attention coefficients giving a value ranging from 0 to 1 for each attention weight in the attention matrix $\alpha$. The sum over any given row or column will be 0. The attention matrix $\alpha$ can be seen as weighted adjacency matrix where each of the non-zero elements is an attention weight. This maps on nicely to the matrix view of GAT.

For an in depth explanation see [Understanding Graph Attention Networks](https://www.youtube.com/watch?v=A-yKQamf2Fc) on YouTube or check out the original [Graph Attention Networks](https://arxiv.org/abs/1710.10903) Paper.

<!-- - It was later revised in 2018 -->

#### Graph Isomorphism Network ([GIN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv))

**Message Passing View**
> $$
> \mathbf{x}_i^{\prime}=h_{\Theta}\left((1+\epsilon) \cdot \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\right)
> $$
>
> $\epsilon$ **:** Hyperparameter that varies the impact of the self-loop.
>
> $h_{\Theta}$ **:** A neural network, .i.e. an MLP.


**Matrix View**
>$X^l = h_{\theta}(\sigma(X^{(k-1)}))$

The Graph Isomorphism Network (GIN) is a more complex version of the GCN that was published in 2019 by Keyulu Xu et al[^5]. This idea can be rationalized by the universal approximation theory of neural networks that shows nearly any function can be approximated by a two layer neural network [^6]$^,$[^7]. By passing the node representations through multiple layers of a Multi-Layer Perceptron (MLP) the GIN is more complex in number of parameters but more expressive in in the data distributions that can be learned.

[^5]:[How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826.pdf)
[^6]:[Multilayer Feedforward Networks are Universal Approximators](https://www.sciencedirect.com/science/article/abs/pii/0893608089900208)
[^7]:[Approximation Capabilities of Multilayer Feedforward Networks](https://www.sciencedirect.com/science/article/pii/089360809190009T)

![](./assets/drawio/GIN.drawio.png)

