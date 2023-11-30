# $\Delta$-UQ

Scripts and demo for [`Single Model Uncertainty Estimation via Stochastic Data Centering`](https://arxiv.org/abs/2207.07235v1), NeurIPS 2022.

**TLDR**: $\Delta$-UQ is a simple strategy that can be used to train a _single_ DNN model to produce meaningful uncertainties.

<img src="/demo/files/duq.png" alt="delta-UQ" title="delta-UQ">

## Train your own $\Delta$-UQ model (CNNs or MLPs)
We provide a easy-to-use simple wrapper that can be used around most CNNs (and can be modified to work with any network) which works as follows:

Anchoring requires double the input channels (in CNNs) or dimensions (in MLPs) -- so first modify your network accordingly:

```
base_net = your_fav_CNN(in_channels=6) ## NOTE: 6 channels instead of 3
```

Next, use the provided utility to wrap your network to produce a $\Delta$-UQ model -- thats it! 

```
from lib.deltaUQ import deltaUQ_CNN as uq_wrapper
net_with_uq = uq_wrapper(base_net)
```

You can treat this model as a drop-in replacement for your old model, same loss function and hparams, its that easy!

```
inputs = torch.randn(64,3,32,32)

pred = net_with_uq(inputs) ## Drop in replacement for training

loss = criterion(targets,pred)
...
```
During inference, you can simply pass an additional flag to produce uncertainties:

```

test_inputs = torch.randn(64,3,32,32)
## (IMPORTANT: *** num test samples > n_anchors > 1 ***)
pred,unc = net_with_uq(test_inputs,n_anchors=5,return_std=True) 
```


We provide wrappers for CNNs and MLPs that modify the base class provided in [`deltaUQ.py`](./lib/deltaUQ.py). 

## Dependencies
This codebase was developed and tested using
+ botorch `0.6.0`
+ gpytorch `1.6.0`
+ matplotlib `3.4.3`
+ numpy `1.20.3`
+ scikit_learn `1.1.3`
+ PyTorch `1.13.1`

## Citation

If you find this repository useful, please consider citing our work as:

```
@inproceedings{thiagarajan2022single,
title={Single Model Uncertainty Estimation via Stochastic Data Centering},
author={Jayaraman J. Thiagarajan and Rushil Anirudh and Vivek Narayanaswamy and Peer-timo Bremer},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=j0J9upqN5va}
}
```
## Related Works
For other applications that use neural network anchoring:

* _Out of Distribution Detection via Neural Network Anchoring_, ACML 2022. [`github`](https://github.com/LLNL/AMP), [`paper`](https://arxiv.org/abs/2207.04125)
* _Predicting the Generalization Gap in Deep Models using Anchoring_, ICASSP 2022. [`github`](https://github.com/vivsivaraman/deltauq_pred_gen), [`paper`](https://ieeexplore.ieee.org/abstract/document/9747136/)
<!-- ## Robustness

## Predicting Generalization Gap in Deep Models
To run our experiments on using DeltaUQ for predicting domain generalization, follow the instructions provided in `./predicting_gen/` -->
## License
This code is distributed under the terms of the MIT license. All new contributions must be made under this license. LLNL-CODE-844746 SPDX-License-Identifier: MIT

