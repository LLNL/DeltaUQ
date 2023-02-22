''' Modify basic ResNet & Train ∆UQ on CIFAR10 with PyTorch.'''
import sys
sys.path.insert(0,'..')

from models import *
from models.resnetv2 import resnet20
from lib.deltaUQ import deltaUQCNN as deltaUQ

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import numpy as np

import datetime
import os
import argparse

from shutil import copyfile
import random

parser = argparse.ArgumentParser(description='Pytorch script for training a ∆UQ model')
parser.add_argument('--nn', default="resnet20", type=str,
                    help='neural network name and training set')
parser.add_argument('--seed', default=1, type=int,
                    help='model training seed')
parser.add_argument('--nref', default=5, type=int,
                    help='number of anchors at test time')
parser.add_argument('--data_path', default='/p/lustre1/anirudh1/delta/pytorch/classification/CIFAR/pytorch-cifar/data/', type=str,help='path to cifar10')


def run_training(**kwargs):
    datasettype = 'cifar10'
    args = kwargs['args']
    CIFAR_PATH = args.data_path
    modeltype = args.nn
    seed = args.seed

    ''' set random seed '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_epochs = 1


    logname = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    if datasettype=='cifar10':
        nclass = 10
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)

        trainset = torchvision.datasets.CIFAR10(
            root=CIFAR_PATH, train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=False, num_workers=2,generator=g_cpu)

        testset = torchvision.datasets.CIFAR10(
            root=CIFAR_PATH, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2,generator=g_cpu)
    elif datasettype =='cifar100':
        nclass = 100
        from utils import get_test_dataloader, get_training_dataloader
        print('==> Preparing data..')
        trainloader = get_training_dataloader(batch_size=128)
        testloader = get_test_dataloader(batch_size=100)



    print('==> Building model..')
    if modeltype=='resnet34':
        net = ResNet34(nc=6,num_classes=nclass)
        modelname = 'DUQ_ResNet34'
    elif modeltype=='resnet20':
        net = resnet20(nc=6,num_classes=nclass)
        modelname = 'DUQ_ResNet20'

    ''' ********* ∆UQ wrapper ********  '''

    net = deltaUQ(base_network=net,seed=seed)
    
    ''' ********* ∆UQ wrapper ********  '''
    modelname = modelname + f'_seed_{seed}'

    modelpath = f'chkpts/{datasettype}/{modelname}/{logname}/'
    if not os.path.isdir(modelpath):
        os.makedirs(modelpath)
    _self_filename = sys.argv[0]
    ## freeze a copy of the training file for the record
    copyfile(_self_filename,modelpath+'/exec_script.py')

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [60, 120, 160], gamma=0.2) #learning rate decay


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            corrupt = (1+batch_idx)%5==0
            
            outputs = net(inputs,corrupt=corrupt)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx %100==0:
                print(f'Epoch# {epoch}, Batch# {batch_idx}, Loss: {train_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.3f}')

    def test(epoch,best_acc):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs,n_anchors=1)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f'********** Epoch {epoch} of {max_epochs} -- Test Loss: {test_loss/(batch_idx+1):.3f}, Test Acc: {100.*correct/total:.3f} **********')

        # Save checkpoint.
        acc = 100.*correct/total
        if epoch == max_epochs-1:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            torch.save(state, f'{modelpath}/ckpt-{epoch}.pth')
            best_acc = acc

        return best_acc



    for epoch in range(start_epoch, start_epoch+max_epochs):
        train(epoch)
        best_acc = test(epoch,best_acc)
        scheduler.step()
    return

if __name__=='__main__':
    args = parser.parse_args()
    run_training(args=args)
