2019ZJU_SummerResearch
====
The task is to restate a thesis and redo the experiment following the instructions in the paper.

Description
----
Worked under the supervision of Professor [Ling Chen](https://scholar.google.com/citations?user=Vxi9eakAAAAJ&hl=zh-CN) in [Pervasive Computing Lab](http://percom.zju.edu.cn/index-en.html) to implement a spatio temporal-multi-graphconvolution
network using Tensorflow for ride-hailing demands forecasting. <br> 
Packages and methods used: Tensorflow(1.14), LSTM(one-way, 3 layers), GCN(Chevb-net) 

Paper:
----
[Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting （Xu Geng, Yaguang Li, etc)](http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf) <br>

### Literature review
This paper proposes a novel deep learning model named the spatiotemporal multi-graph convolution network (ST-MGCN) for better region-level ride-hailing demand forecasting. And the network focus more on non-Euclidean pair-wise correlations among distant regions rather than Euclidean correlations as many previous researches. <br>
<br>
The challenges in this research are from complex spatial and temporal correlations. On the one hand, complicated dependencies are observed among different regions. On the other hand, non-linear dependencies also exist among different temporal observations. <br>
<br>
To address these challenges, the team firstly encodes the non-Euclidean correlations like the neighborhood, functional similarity and transportation connectivity among regions into multiple graphs. And then, for each graph, the researchers use contextual gated recurrent neural network which augments recurrent neural network with a contextual-aware gating mechanism to re-weights different historical observations. After that, the researchers further leverage the multi-graph convolution to explicitly model these correlations. And Finally, a fully connected neural network is used to transform features into the prediction. <br>
<br>
For the evaluation, the team evaluates the proposed model on two real-world large scale ride-hailing demand datasets and observes consistent improvement of more than 10% over state- of-the-art baselines. <br>
<br>
For the contribution: 
* Identified its unique spatiotemporal correlation 
* A novel deep learning based model which encoded the non-Euclidean correlations among regions using multiple graphs and explicitly captured them using multi-graph convolution 
* Further augmented the recurrent neural network with contextual gating mechanism to incorporate global contextual information in the temporal modeling procedure 


Tools:
---
* PC:
MacBook Air (13-inch, Early 2014)
* Python:
Python 2.7
* PyCharm:
PyCharm 2019.1.3, build PC-191.7479.30. Copyright JetBrains s.r.o., (c) 2000-2019
* Python library:
Tensorflow 1.14, keras 2.2.4, sklearn 0.20.3, Numpy 1.16.4, Pandas 0.24.2, Matplotlib 2.2.4


Results:
---
1.Use the damand of next timestep t+1 as the predicted demand of t, RMSE: 35.804382<br>
2.One-way LSTM, best RMSE:29.434<br>
3.ST-MGCN (no attention), best RMSE:20.3907909393<br>
4.ST-MGCN (attention), best RMSE:23.5178871155


Ref:
----
### Paper Ref:
1.[Semi-supervised classification with graph convolutional networks (Thomas N. Kipf, etc)](https://arxiv.org/pdf/1609.02907.pdf)<br> 
2.[Spaio-temporal graph convolutional networks: A deep learning Framework for traffic forecasting (Bing Yu, etc)](https://arxiv.org/pdf/1709.04875.pdf)<br> 
3.[Diffusion convolutional recurrent neural network: data-driven traffic forecasting (Yaguang Li, etc)](https://arxiv.org/pdf/1707.01926.pdf)<br> 
4.[Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting （Xu Geng, Yaguang Li, etc)](http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf)<br> 
5.[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf)

### Code Ref:
1.[Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/)<br>
2.[DCRNN](https://github.com/liyaguang/DCRNN)
