### Models
To test the hypothesis that the GIN is more expressive we trained a GCN, GAT, and GIN model on a benchmark dataset provided by [OBG](https://ogb.stanford.edu/). The models are trained and then nodes representations are pooled to get a global representation of the graph for graph classification. All source code for the models can be found in the `pyg_gym` directory. Specifically the following files are used to train models:
- [main.py](https://github.com/Mjvolk3/MATH-490-Project/blob/main/pyg_gym/main.py)
- [metrics.py](https://github.com/Mjvolk3/MATH-490-Project/blob/main/pyg_gym/metrics.py)
- [models.py](https://github.com/Mjvolk3/MATH-490-Project/blob/main/pyg_gym/models.py)
- [mp.py](https://github.com/Mjvolk3/MATH-490-Project/blob/main/pyg_gym/mp.py)
- [pool.py](https://github.com/Mjvolk3/MATH-490-Project/blob/main/pyg_gym/pool.py)
- [runner.py](https://github.com/Mjvolk3/MATH-490-Project/blob/main/pyg_gym/runner.py)
- [config_override.py](https://github.com/Mjvolk3/MATH-490-Project/blob/main/pyg_gym/config_override.py)

### OGB Data
To familiarize yourself with the dataset we recommend reading the description of [obgb-ppa](https://ogb.stanford.edu/docs/graphprop/#ogbg-ppa). I have copied the relevant row from the data table provided. In brief, this is a graph classification tasks over 37 different classes.

| Scale  | Name     | Package | Num Graphs | Num Nodes per graph | Num lEdges per graph | Num Tasks | Split Type | Task Type                  | Metric   |
|--------|----------|---------|------------|---------------------|----------------------|-----------|------------|----------------------------|----------|
| Medium | ogbg-ppa | >=1.1.1 | 158,100    | 243.4               | 2,266.1              | 1         | Species    | Multi-class classification | Accuracy |

### Wandb

All experiments can be viewed on [wandb](https://wandb.ai/mjvolk3/MATH-490-Project?workspace=user-mjvolk3). This workspace should be public access so comment below if it is not.

<!-- TODO update. -->
At the time of last commit, models were still in the process of training so I did not pull in their key results.