import torch
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as datautil
import statistics



epochs = 10

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
traindataset = torch.load('Causal data A-0.hdf5')
train_loader = datautil.DataLoader(
        traindataset,
        batch_size=args.batch_size, shuffle = True, **kwargs)
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
print(model)

# create a stochastic gradient descent optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# create a loss function
criterion = nn.MSELoss()


# run the main training loop
for epoch in range(epochs):
    for batch_idx, (target, data, mu) in enumerate(train_loader):

        # data, target = Variable(data), Variable(target)
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        optimizer.zero_grad()
        net_out = model(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
# run a test loop
loss_alpha =[]
loss_alpha_std =[]
loss_alpha_mu=[]
for i in range(-10,11):
    test_loss = 0
    test_mu_loss=0
    correct = 0
    test_loss_lst = []
    testdataset = torch.load('Causal data A-'+str(i)+'.hdf5')
    print('A-'+str(i))
    test_loader = datautil.DataLoader(testdataset,
                                      batch_size=args.batch_size, shuffle=True, **kwargs)
    for data, target, mu in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        net_out = model(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).item()
        test_loss_lst.append(criterion(net_out, target).item())
        test_mu_loss += criterion(net_out, mu).item()
        # pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        # correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    test_loss_std = statistics.stdev(test_loss_lst)
    test_mu_loss /= len(test_loader.dataset)

    loss_alpha.append(test_loss)
    loss_alpha_mu.append(test_mu_loss)
    loss_alpha_std.append(test_loss_std)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

import matplotlib.pyplot as plt
plt.plot(list(range(-10,11)), loss_alpha, 'ro')
plt.plot(list(range(-10,11)), loss_alpha_mu)
plt.errorbar(list(range(-10,11)), loss_alpha, loss_alpha_std)
plt.xlabel('Alpha value')
plt.ylabel('Loss (MSE)')
plt.title('A: mu = 10 +alpha')
plt.show()

