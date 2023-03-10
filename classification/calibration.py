# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Authors: Rushil Anirudh, Jayaraman J. Thiagarajan. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

import numpy as np
import glob
import os
import sys
sys.path.insert(0,'..')

from lib.deltaUQ import deltaUQ
import pickle as pkl

from utils import CIFAR10C
from models import *
from models.resnetv2 import resnet20


def run_DUQ(corruption='brightness',clevel=4,nref=20,modeltype='resnet20',seed=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CIFAR_PATH = '../pytorch/classification/CIFAR/pytorch-cifar/data/'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=CIFAR_PATH, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    nclass=10
    if modeltype=='resnet20':
        net = resnet20(nc=6,num_classes=nclass)
        modelname = 'DUQ_repulsive_2e-2_ResNet20'
    elif modeltype=='resnet18':
        net = ResNet18(nc=6,num_classes=nclass)
        modelname = 'DUQv2_ResNet18'

    elif modeltype=='resnet50':
        net = ResNet50(nc=6,num_classes=nclass)
        modelname = 'DUQv2_ResNet50'

    else:
        print('modeltype not understood, training resnet20')
        net = resnet20(nc=6,num_classes=nclass)
        modelname = 'DUQv2_ResNet20'

    modelname = modelname + f'_seed_{seed}'

    modelpath = f'chkpts/cifar10/{modelname}/'
    dirs = glob.glob(f'chkpts/cifar10/{modelname}/*')
    modelpath = f'{dirs[0]}/ckpt-199.pth'

    savepath = f'./calib-results/{modeltype}/'
    filename = savepath+f'seed_{seed}_{corruption}_{clevel}_nref{nref}_predictions.pkl'
    
    if os.path.exists(filename):
        return

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    net = deltaUQ(base_network=net)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(f'---best accuracy---{best_acc:.4f}')

    net.to(device)
    net.eval()
    net.cuda()
    print(torch.cuda.memory_allocated()/1024**2)


    ood_dataset = CIFAR10C(numpy_path=f'../pytorch/classification/CIFAR/pytorch-cifar/data/CIFAR-10-c/{corruption}.npy',level=clevel,transform=transform_test)
    ood_dataloader = DataLoader(ood_dataset,shuffle=False,batch_size=1000,num_workers=2)

    uncs = []
    pred_vals = []
    gt = []
    softmax = []
    count = 0
    correct = 0
    total = 0

    pred_dict = {}
    criterion = torch.nn.BCELoss(reduction='none')

    for (ood_batch, ood_targets),(refs,_) in zip(ood_dataloader,trainloader):


        print(f'######## Batch :{count} ##########')
        count +=1
        ood_batch,ood_targets = ood_batch.to(device),ood_targets.to(device)
        refs = refs.to(device)
        with torch.no_grad():
            calib_mu, std = net(ood_batch,anchors=refs,n_anchors=nref,return_std=True)
            _, predicted = calib_mu.max(1)

        total += ood_targets.size(0)
        correct += predicted.eq(ood_targets).sum().item()

        uncs.append(std.detach().cpu().numpy())
        pred_vals.append(torch.argmax(calib_mu,1).detach().cpu().numpy())
        gt.append(ood_targets.detach().cpu().numpy())
        softmax.append(calib_mu.detach().cpu().numpy())

    acc = 100.*correct/total
    print(f'********** Corruption:{corruption}, Level: {clevel}, CIFAR-10-C Test Acc: {acc:.3f} **********')

    pred_dict['uq'] = np.array(uncs)
    pred_dict['preds'] = np.array(pred_vals)
    pred_dict['gt'] = np.array(gt)
    pred_dict['softmax'] = np.array(softmax)
    
    with open(filename,'wb') as f:
        pkl.dump(pred_dict,f)

    return
    
if __name__=='__main__':
    corruptions = ['shot_noise','gaussian_blur','spatter','pixelate','gaussian_noise',
                   'defocus_blur','brightness','fog','zoom_blur','frost','glass_blur',
                   'impulse_noise','contrast','speckle_noise','elastic_transform','saturate']

    seed = int(sys.argv[1])
    cl = int(sys.argv[2])
    print(f'Running Seed {seed}, C-Level: {cl}')
    for c in corruptions:
        run_DUQ(corruption=c,clevel=cl,nref=20,modeltype='resnet20',seed=seed)
