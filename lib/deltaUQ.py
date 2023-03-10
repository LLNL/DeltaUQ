# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Authors: Rushil Anirudh, Jayaraman J. Thiagarajan. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np
import random
from abc import ABC, abstractmethod


class deltaUQ(torch.nn.Module):
    def __init__(self,base_network):
            super(deltaUQ, self).__init__()
            '''

            base_network (default: None):
                network used to perform anchor training (takes 6 input channels)

            '''
            if base_network is not None:
                self.net = base_network
            else:
                raise Exception('base network needs to be defined')

    def create_anchored_batch(self,x,anchors=None,n_anchors=1,corrupt=False):
        '''
        anchors (default=None):
            if passed, will use the same set of anchors for all batches during training.
            if  None, we will use a shuffled input minibatch to forward( ) as anchors for that batch (random)

            During *inference* it is recommended to keep the anchors fixed for all samples.

            n_anchors is chosen as min(n_batch,n_anchors)
        '''
        n_img = x.shape[0]
        if anchors is None:
            anchors = x[torch.randperm(n_img),:]
        
        ## make anchors (n_anchors) --> n_img*n_anchors
        if self.training:
            A = anchors[torch.randint(anchors.shape[0],(n_img*n_anchors,)),:]
        else:
            A = torch.repeat_interleave(anchors[torch.randperm(n_anchors),:],n_img,dim=0)    

        if corrupt:
            refs = self.corruption(A)
        else:
            refs = A

        ## before computing residual, make minibatch (n_img) --> n_img* n_anchors
        if len(x.shape)<=2:

            diff = x.tile((n_anchors,1))
            assert diff.shape[1]==A.shape[1], f"Tensor sizes for `diff`({diff.shape}) and `anchors` ({A.shape}) don't match!"
            diff -= A
        else:
            diff = x.tile((n_anchors,1,1,1)) - A

        batch = torch.cat([refs,diff],axis=1)

        return batch

    
    def corruption(self,samples):
        #base case does not use corruption in anchoring
        return samples

    @abstractmethod
    def calibrate(self):
        pass

    @abstractmethod
    def forward(self):
        pass

class deltaUQ_Encoder(deltaUQ):
    def __init__(self,base_network):
            super(deltaUQ, self).__init__()
            '''

            base_network (default: None):
                network used to perform anchor training (takes 6 input channels)

            '''
            if base_network is not None:
                self.net = base_network        

    def forward(self,x,anchors=None,n_anchors=1,return_std=False,calibrate=False):
        if n_anchors==1 and return_std:
            raise Warning('Use n_anchor>1, std. dev cannot be computed!')

        a_batch = self.create_anchored_batch(x,anchors=anchors,n_anchors=n_anchors)
        p = self.net(a_batch)
        
        p = p.reshape(n_anchors,x.shape[0],p.shape[1])
        mu = p.mean(0)

        if return_std:
            std = p.std(0)
            return mu, std
        else:
            return mu       
        


class deltaUQ_CNN(deltaUQ):
    def __init__(self,base_network):
            super(deltaUQ, self).__init__()
            '''

            base_network (default: None):
                network used to perform anchor training (takes 6 input channels)

            '''
            if base_network is not None:
                self.net = base_network
                if self.net.conv1.weight.shape[1]!=6:
                    raise ValueError('Base Network has incorrect number of input channels (must be 6 for RGB datasets)')

    def corruption(self,samples):
        self.txs = transforms.Compose([
                transforms.RandomResizedCrop(size=32,scale=(0.6,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.5),
                                        transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5))], p=0.8),
                                        transforms.RandomGrayscale(p=0.2)])
        return self.txs(samples)
        
    def calibrate(self,mu,sig):
        '''
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        '''
        c = torch.mean(sig,1)
        c = c.unsqueeze(1).expand(mu.shape)
        return torch.div(mu,1+torch.exp(c))
        # return torch.div(mu,c)

    def forward(self,x,anchors=None,corrupt=False,n_anchors=1,return_std=False,calibrate=False):
        if n_anchors==1 and return_std:
            raise Warning('Use n_anchor>1, std. dev cannot be computed!')

        a_batch = self.create_anchored_batch(x,anchors=anchors,n_anchors=n_anchors,corrupt=corrupt)
        p = self.net(a_batch)
        
        p = p.reshape(n_anchors,x.shape[0],p.shape[1])
        mu = p.mean(0)

        if return_std:
            std = p.sigmoid().std(0)
            if calibrate:
                return self.calibrate(mu,std), std
            else:
                return mu, std
        else:
            return mu


class deltaUQ_MLP(deltaUQ):
    def forward(self,x,anchors=None,corrupt=False,n_anchors=1,return_std=False):
        # no calibration
        if n_anchors==1 and return_std:
            raise Warning('Use n_anchor>1, std. dev cannot be computed!')

        a_batch = self.create_anchored_batch(x,anchors=anchors,n_anchors=n_anchors,corrupt=corrupt)
        p = self.net(a_batch)
        
        p = p.reshape(n_anchors,x.shape[0],p.shape[1])
        mu = p.mean(0)

        if return_std:
            std = p.std(0)
            return mu, std
        else:
            return mu


if __name__=='__main__':
    '''
    EXAMPLE USAGE
    '''
    from models.resnetv2 import resnet20
    base_net = resnet20(nc=6,n_classes=10)
    model = deltaUQ_CNN(base_net)

    inputs = torch.randn(64,3,32,32)
    anchors = torch.randn(32,3,32,32)

    ## acts like a vanilla model
    pred = model(inputs)

    ## returns std dev of predictions
    pred1,unc1 = model(inputs,n_anchors=5,return_std=True)

    ## returns std dev of predictions for specified anchors
    pred2,unc2 = model(inputs,anchors=anchors,n_anchors=5,return_std=True)
