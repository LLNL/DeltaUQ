# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Authors: Rushil Anirudh, Jayaraman J. Thiagarajan. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
import torch

def testfun(n=20):
    sep = 1
    x = torch.zeros(n // 2, 1).uniform_(0, 0.5)
    x = torch.cat((x, torch.zeros(n // 2, 1).uniform_(0.5 + sep, 1 + sep)), 0)
    m = torch.distributions.Exponential(torch.tensor([3.0]))
    noise = m.rsample((n,))
    y = (2 * 3.1416 * x).sin() + 0.*noise
    x_test = torch.linspace(-0.5, 2.5, 2000).view(-1, 1)
    y_test = (2 * 3.1416 * x_test).sin()
    return x.detach().cpu().numpy(), y.detach().cpu().numpy(), x_test.squeeze(1).detach().cpu().numpy(), y_test.squeeze(1).detach().cpu().numpy()

def multi_optima(X):
    # X is a n x 1 tensor
    # outputs a n x 1 tensor
    # Reasonable bounds is (-1, 2)
    return (torch.sin(X) * torch.cos(5 * X) * torch.cos(22 * X)).squeeze(1)
    
    

def ackley(x):
    dim = x.shape[1]
    a = 20
    b = 0.2
    c = 2*np.pi

    x = x.detach().cpu().numpy()

    part1 = -a*np.exp(-b*np.sqrt(np.sum(x**2, axis=1)/dim))
    part2 = -np.exp(np.sum(np.cos(c*x),axis=1)/dim)
    part3 = a
    part4 = np.exp(1)
    return -1*torch.from_numpy(part1 + part2 + part3 + part4)


functions = {'multi_optima': multi_optima,
             'ackley': ackley}

bounds = {'multi_optima': (-1, 2),
          'ackley': (-5, 5)}

gopt = {'multi_optima': 0.951,
        'ackley': 0}
