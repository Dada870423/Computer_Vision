import os
import sys
import torch
import torchvision
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import resnet
from matplotlib import pyplot as plt

self_transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                    torchvision.transforms.CenterCrop(200),
                                    torchvision.transforms.Resize(200), 
                                    torchvision.transforms.Pad(1),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.ImageFolder(root='../hw5_data/train',transform=self_transforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.ImageFolder(root='../hw5_data/test',transform=self_transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

print('data has been loaded')

cuda_flat = torch.cuda.is_available()
net = resnet.resnet34()
print(net)

if cuda_flat:
    torch.backends.cudnn.benchmark = True
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print("cuda can be used")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())#, lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
acc_list=[]

def plotData(plt, data):
  x = [p[0] for p in data]
  y = [p[1] for p in data]
  plt.plot(x, y, '-o')

def train(epoch):

    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if cuda_flat:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()

        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()   #return python number
        _, predicted = torch.max(outputs.data, 1)   #predicted class
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        print("epoch: %d, batch: %d, accuracy: %.3f"%(epoch+1, batch_idx+1,float(correct)/float(total)))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cuda_flat:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)   #value/idx, every row gets one element with largest value
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum() #tensor.cpu() moves it back to memory accessible to the CPU.

        print("epoch: %d, batch: %d, accuracy: %.3f" % (epoch+1, batch_idx+1,float(correct)/float(total)))
        if batch_idx == 4:
            acc_list.append((epoch+1,(float(correct.data)/float(total))))

epoch_sum=50
for epoch in range(epoch_sum):
    print('------Start training for Epoch %d'%epoch)
    train(epoch)
    print('------Start testing for Epoch %d'%epoch)
    test(epoch)



plotData(plt,acc_list)
my_x_ticks = np.arange(1, epoch_sum, 1)
plt.xticks(my_x_ticks)
plt.ylim(0,1)
plt.xlabel("training times")
plt.ylabel("accuracy")


plt.show()