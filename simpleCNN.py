import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


train_data = CIFAR10(root="../Cifar/",
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)

test_data = CIFAR10(root="../Cifar/",
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)

batchsize = 32
num_classes = 10
num_epochs = 3

trainloader = DataLoader(dataset=train_data,
                         batch_size=batchsize,
                         shuffle=True)

testloader = DataLoader(dataset=test_data,
                         batch_size=batchsize,
                         shuffle=False)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(800, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


total_step = len(trainloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(testloader):
        outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)
        correct += (predictions == labels).sum().item()
        total += batchsize
        if i % 100 == 0:
            print("Test accuracy at iter {} = {}".format(i, 100*correct/total))
    print("Final test accuracy = {}".format(100 * correct / total))