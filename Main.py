from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
import csv
from datetime import datetime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3) # padding=1
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, 1, padding=1)  
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3) # padding=1
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(32, 16, 1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 32, 3) # padding=1
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 10, 3) #1  # Changed kernel size to 1
        self.gap = nn.AdaptiveAvgPool2d(1)  # Added Global Average Pooling

        self.dropout = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.15)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.dropout(x)
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))
        x = self.dropout2(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)  # Added dim=1

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

torch.manual_seed(3) # 1
batch_size = 128 # 128 # 64

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation((-8, 8)),  # Reduced rotation further
                        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05), shear=(-5, 5)),
                        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
                    ])),
    batch_size=128, shuffle=True, **kwargs)  # Reduced batch size
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch, best_accuracy, csv_writer):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    total = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Epoch: {epoch}, Training Accuracy: {train_accuracy:.2f}%')
    
    return train_accuracy, avg_train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))
    
    return accuracy, test_loss

model = Net().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, amsgrad=True)

# Initialize best accuracy tracker
best_accuracy = [0.0]

# Before the training loop, create a CSV file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f'training_log_{timestamp}.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Epoch', 'Training Accuracy', 'Test Accuracy', 'Training Loss', 'Test Loss', 'Best Accuracy'])

# Before training loop
for epoch in range(1, 20):
    train_accuracy, train_loss = train(model, device, train_loader, optimizer, epoch, best_accuracy, csv_writer)
    test_accuracy, test_loss = test(model, device, test_loader)
    
    # Save epoch details to CSV
    csv_writer.writerow([
        epoch,
        f'{train_accuracy:.2f}',
        f'{test_accuracy:.2f}',
        f'{train_loss:.4f}',
        f'{test_loss:.4f}',
        f'{best_accuracy[0]:.2f}'
    ])
    csv_file.flush()
    
    if test_accuracy > best_accuracy[0]:
        best_accuracy[0] = test_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved!")

csv_file.close()
