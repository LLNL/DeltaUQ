# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Authors: Rushil Anirudh, Jayaraman J. Thiagarajan. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from torch.utils.data import TensorDataset, random_split
import torch
import torch.nn as nn
from copy import deepcopy
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.optim import optimize_acqf
from torch.optim import Adam, SGD
from .modeldefs import MLP
from .base import BaseModel
from .deltaUQ import deltaUQ_MLP

import numpy as np

qval = 1

def delta_enc(query_x,anchor_x,query_y):
    residual = query_x - anchor_x
    inp = torch.cat([residual,anchor_x],axis=1)
    out = query_y
    return inp, out

class _DUQv0(BaseModel):
    #old API
    def __init__(self,net, opt, X, y):
        super().__init__()
        self.net = net
        self.opt = opt
        self.h = self.net.input_mapping(X)
        self.y = y.view(-1,1)
        self.loss = nn.MSELoss()
        self.n_epochs = 300
        self.n_samp = X.shape[0]
        self.input_dim = X.shape[1]
        self.output_dim = 1
        
    def fit(self):
        x_denc, y_denc = [], []
        for i in range(self.n_samp):
            for j in range(self.n_samp):
                xd,yd = delta_enc(self.h[j].unsqueeze(0),self.h[i].unsqueeze(0),self.y[j])
                x_denc.append(xd)
                y_denc.append(yd)
                
        x_denc = torch.cat(x_denc)
        y_denc = torch.cat(y_denc).view(-1,1)
        
        for epoch in range(self.n_epochs):
            yhat = self.net(x_denc)
            self.opt.zero_grad()
            f_loss = self.loss(yhat, y_denc)
            f_loss.backward()
            self.opt.step()
            
    def _map_delta_model(self,anchor,query):
        query = self.net.input_mapping(query)
        residual = query-anchor
        denc = torch.cat([residual, anchor],1)
        pred = self.net(denc)
        return pred

    def get_prediction_with_uncertainty(self, q):
        out = super().get_prediction_with_uncertainty(q)
        if out is None:
            if len(q.shape) > 2:
                q = q.squeeze(1)
            nref=np.minimum(30, self.n_samp)
            self.net.eval()

            all_preds = []
            n_test = q.shape[0]
            inds = np.arange(self.n_samp)
            np.random.shuffle(inds)
            inds = inds[:nref]
            for i in inds:
                anchor = self.h[i]
                val = self._map_delta_model(anchor.expand([n_test,anchor.shape[0]]),q.float())
                all_preds.append(val)

            all_preds = torch.stack(all_preds).squeeze(2) #[nref, n_test]
            mu = torch.mean(all_preds,axis=0)
            var = torch.var(all_preds,axis=0)

            return mu,var
        return out

class DUQ(BaseModel):
    def __init__(self,net, opt, X, y):
        super().__init__()
        self.h = net.input_mapping(X)
        self.net = deltaUQ_MLP(net)
        self.opt = opt
        self.y = y.view(-1,1)
        self.loss = nn.MSELoss()
        self.n_epochs = 400
        self.n_samp = X.shape[0]
        self.input_dim = X.shape[1]
        self.output_dim = 1
        
    def fit(self):    
        for epoch in range(self.n_epochs):
            yhat = self.net(self.h)
            self.opt.zero_grad()
            f_loss = self.loss(yhat, self.y)
            f_loss.backward()
            self.opt.step()
            
    def get_prediction_with_uncertainty(self, query):
        out = super().get_prediction_with_uncertainty(query)
        q = self.net.net.input_mapping(query)

        if out is None:
            if len(q.shape) > 2:
                q = q.squeeze(1)
            nref=np.minimum(30, self.n_samp)
            self.net.eval()
            mu, std = self.net(q.float(),anchors=self.h,n_anchors=nref,return_std=True)
            return mu, std**2
        return out

def optimize(f, b, indim, outdim, X_init, Y_init, n_steps, scale = 2**-8, drp = -1, bn = False):
    full_train_X = X_init
    full_train_Y = Y_init

    state_dict = None
    buffer = None

    max_value_per_step = [full_train_Y.max().item()]
    
    for step in range(n_steps):
        full_train_X, full_train_Y = one_step_acquisition(f, full_train_X, full_train_Y, b, indim, outdim, scale,  drp, bn)
        max_value_per_step.append(full_train_Y.max().item())
        if (step+1)%5 == 0:
            print(f'Finished Step {step+1}/{n_steps}: Best value so far = {max_value_per_step[-1]}')
    buffer = (full_train_X, full_train_Y)
    
    return max_value_per_step, buffer

def one_step_acquisition(f, full_train_X, full_train_Y, b, indim, outdim, scale = 2**-6, drp = -1, bn = False):
    net = MLP(indim, outdim, ff = True, mapsize = 32, scale = scale, drp = drp, bn = bn)
    opt = Adam(net.parameters(), lr=0.01)
    duq = DUQ(net, opt, full_train_X, full_train_Y)
    duq.fit()
    candidate, EI = get_candidate(duq, full_train_Y, b, indim)
    candidate_image = f(candidate)
    full_train_X = torch.cat([full_train_X, candidate])
    full_train_Y = torch.cat([full_train_Y, candidate_image])
    return full_train_X, full_train_Y

def get_candidate(duq, full_train_Y, b, indim):
    EI = qExpectedImprovement(duq, full_train_Y.max().item())
    bounds_t = torch.FloatTensor([[b[0]] * indim, [b[1]] * indim])
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds_t, q=qval, num_restarts=15, raw_samples=5000
    )
    return candidate, EI

