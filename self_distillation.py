from __future__ import print_function
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time
import numpy as np
start_time = time.time()


class Net(nn.Module):
    def __init__(self, dropout=0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout = dropout
        print("Droput rate : ", self.dropout)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x

def cross_entropy_soft(predicted, target):
    return -(target * torch.log(predicted)).sum(dim=1).mean()

def combined_loss(output, target, soft_target, alpha):
    hard_loss = F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    soft_target = F.softmax(soft_target, dim=1)
    soft_loss = cross_entropy_soft(output, soft_target)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def distillation_loss(target, output, teacher, data, device, alpha):
    if teacher is None:
        loss = F.cross_entropy(output, target)
        #print("Hard label loss ", loss.item())
    else:
        teacher.eval()
        teacher.to(device)

        #Loss 1 - KLDivLoss()
        #criterion = nn.KLDivLoss()  # use Kullback-Leibler divergence loss
        #output = F.log_softmax(output, dim=1)

        with torch.no_grad():
            soft_target = teacher(data)

        loss = combined_loss(output, target, soft_target, alpha)

    return loss

def self_distillation_train(model, train_loader, device, optimizer, epochs, teacher, test_loader, log_every, alpha, step):
    model.train()
    mx_test_acc = 0.0
    for epoch in range(epochs):
        epoch_loss = []
        print("\nBeginning of Epoch : {}".format(epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = distillation_loss(target, output, teacher, data, device, alpha)
            loss.backward()
            #print("Loss ", loss.item())
            optimizer.step()
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #                    epoch, batch_idx * len(data), len(train_loader.dataset),
            #                                    100. * batch_idx / len(train_loader), loss.item()))
            epoch_loss.append(loss.item())

        if epoch % log_every == 0:
            #test_acc, test_loss = test_batch(model, data, target, teacher)
            test_loss, test_acc = test(model, test_loader, device, distilled=True)
            if test_acc > mx_test_acc:
                mx_test_acc = test_acc
                torch.save(model.state_dict(), 'distilled_models/distilled' + str(step) + "-" + str(epoch) + "_" + str(mx_test_acc) + '.pth.tar')

            #print("epoch {} test  accuracy {} and  loss {:06f} (for 1 batch)".format(epoch , test_acc , test_loss))


        print('Epoch: {}, Loss : {:.6f}'.format(epoch, np.sum(epoch_loss) / len(epoch_loss)))

    return model


def train_evaluate(model, train_loader, device):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            #data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            train_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))


def test(model, test_loader, device, distilled):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        if distilled == False:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        else:
            print('Test set loss for distilled model: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    return test_loss, test_acc


def get_optimizer(optimizer_name, model, lr, momentum):
    """

    :param optimizer_name:
    :return:
    """
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)


    return optimizer



def get_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-batch-size', type=int, default=32, metavar='N', help='train batch size')
    parser.add_argument('-test-batch-size', type=int, default=32, metavar='N', help='test batch size')
    parser.add_argument('-epochs', type=int, default=20, metavar='N', help='number of epochs to train')
    parser.add_argument('-lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('-optimizer', type=str, default='sgd', metavar='O', help='optimizer name')
    parser.add_argument('-alpha', type=float, default=1.0, metavar='O', help='Hyperparameter to weight soft and hard loss')
    parser.add_argument('-dropout', type=float, default=0.0, metavar='LR', help='Dropout rate')
    parser.add_argument('-momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('-distill_iter', type=int, default=15, metavar='M', help='Number of self distillation iterations to perform.')
    parser.add_argument('-log_every', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()

    return  args

def main(args):

    model = Net()
    #model.load_state_dict(torch.load('teacher_MLP_test.pth.tar'))
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.momentum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    kwargs = {'num_workers': 1, 'pin_memory': False}

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #test(model, test_loader, device, distilled=False)

    teacher = None
    test_losses = test_accs = []
    model = Net(args.dropout)
    model.to(device)
    for step in range(args.distill_iter):
        print("\n********** Begin Self Distillation Step - {} *************\n".format(step))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
        model = self_distillation_train(model, train_loader, device, optimizer, args.epochs, teacher, test_loader, args.log_every, args.alpha, step)
        teacher = Net(args.dropout)
        teacher.load_state_dict(model.state_dict())
        #torch.save(model.state_dict(), 'distilled_models/distilled' + str(step) + '.pth.tar')

    torch.save(model.state_dict(), 'distilled.pth.tar')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    args = get_arguments()
    main(args)

