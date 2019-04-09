import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime


########################################################################
# 0. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# neural network that take 3-channel images 

class Net(nn.Module):

    imgSizeStart = 32
    totalParam = 0
    currentImgSize = 32
    epochNumb = 6
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = self.conv2DStat(3, 12, 5, paddings=5)
        self.conv2 = self.conv2DStat(12, 24, 5, paddings=5)
        self.conv3 = self.conv2DStat(24, 48, 3)

        self.pool = self.maxPool2dStat(2, 2)
        self.fc1 = self.linearStat(48 * 5 * 5, 120)
        self.fc2 = self.linearStat(120, 84)
        self.fc3 = self.linearStat(84, 10)

        #self.totalParam = self.totalParam * self.epochNumb
        self.totalParam

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print (x.shape)
        x = x.view(-1, 48 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def getParamNumber(self):
        imgSizeStart = 32

    def runNetwork(self, testloader, epoch) :
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return str(epoch) + ': Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total)

    def conv2DStat(self,inputs, outputs, size, paddings = 0):
        self.totalParam += ((inputs + paddings) * outputs * size * size) + outputs
        self.currentImgSize = self.currentImgSize - (size - 1) + paddings
        return nn.Conv2d(inputs, outputs, size, padding=paddings)

    def maxPool2dStat(self,width, heigth):
        self.currentImgSize = self.currentImgSize / 2
        return nn.MaxPool2d(width, heigth)

    def linearStat(self,inputs, outputs):
        self.totalParam += inputs * outputs + outputs
        return nn.Linear(inputs, outputs)

if __name__ == '__main__':

    start = datetime.datetime.now()
    ########################################################################
    # 1. Check Cuda availability
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ########################################################################
    # 2. Import Dataset
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Image Dataset

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ########################################################################
    # 3. Initiate neural net
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Image Dataset

    net = Net()
    net.to(device)

    ########################################################################
    # 4. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    ########################################################################
    # 5. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    # Train the network
    print ("Total Param:" + str(net.totalParam))
    print (net.runNetwork(testloader, 0))

    for epoch in range(net.epochNumb):  # loop over the dataset multiple times
       
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)           

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print (net.runNetwork(testloader, epoch+1))
    
    print('Finished Training')

    ########################################################################
    # Let us look at how the network performs on the whole dataset.

    print (net.runNetwork(testloader, 100))

    ########################################################################
    # Better than chance, which is 10% accuracy
    # Seems like the network learnt something.
    # Display classes statistic

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    end = datetime.datetime.now() - start
    print ('total time: '+ str(end) )