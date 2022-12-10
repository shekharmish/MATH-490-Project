---
id: 6je8bh4fpqfkr49ekm38lml
title: Introduction
desc: ''
updated: 1670649475620
created: 1670641852737
enableGiscus: true
---
### Message Passing Framework
$$
\mathbf{x}_i^{(k)}=\gamma^{(k)}\left(\mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)}, \mathbf{e}_{j, i}\right)\right)
$$

$\square_{j \in \mathcal{N}(i)}$ - Differentiable, permutation invariant function, *e.g.* sum, mean, or max.

$\phi^{(k)}$ - Differentiable functions such as Multi-Layer Perceptrons (MLPs)

$\gamma^{(k)}$ - Differentiable functions such MLPs


### Toy Example

To gain some intuition of a message passing framework we will take a small graph as example.

![](./assets/drawio/Message-Passing-Framework.drawio.png)

We will consider messages passed to node $D$ in a 2-layer message passing neural network. The computational graph can be traced out from node $D$ first to a 1-hop neighborhood and then to a 2-hop neighborhood. The 1-hop neighborhood contains all nodes except node $A$. Including node $D$ is often considered optional and is can be interpreted as adding self-loops to the graph. Layer 0 looks shows connections from the 2-hop neighborhood of node $D$.

### Popular Message Passing Neural Networks

#### GCN [^1]

[^1]: [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)

$$
\mathbf{x}_i^{(k)}=\sum_{j \in \mathcal{N}(i) \cup\{i\}} \frac{1}{\sqrt{\operatorname{deg}(i)} \cdot \sqrt{\operatorname{deg}(j)}} \cdot\left(\boldsymbol{\Theta}^{\top} \cdot \mathbf{x}_j^{(k-1)}\right)
$$

#### GAT [^2]

[^2]: [GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv)

$$
\mathbf{x}_i^{\prime}=\alpha_{i, i} \boldsymbol{\Theta} \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \alpha_{i, j} \boldsymbol{\Theta} \mathbf{x}_j
$$

$$
\alpha_{i, j}=\frac{\exp \left(\mathbf{a}^{\top} \operatorname{LeakyReLU}\left(\boldsymbol{\Theta}\left[\mathbf{x}_i \| \mathbf{x}_j\right]\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left(\mathbf{a}^{\top} \operatorname{LeakyReLU}\left(\mathbf{\Theta}\left[\mathbf{x}_i \| \mathbf{x}_k\right]\right)\right)}
$$

#### GIN [^3]

[^3]: [PyG-GINConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv)

$$
\mathbf{x}_i^{\prime}=h_{\Theta}\left((1+\epsilon) \cdot \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\right)
$$

Where $h_{\Theta}$ denotes a neural network, .i.e. an MLP.
