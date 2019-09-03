# based on https://github.com/pytorch/examples/blob/master/mnist/main.py

import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(2*2*50,500)
        self.fc2 = nn.Linear(500,2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class KbmodDataSet(Dataset):
    def __init__(self, path, train=True):
        #fake_positives = np.load(path+'stamps_simulated.npy').astype(float)
        #false_positives = np.load(path+'stamp_training_set.npy').astype(float)
        
        #raw = np.load(path+'stamp_training_set.npy').astype(float)
        #np.random.shuffle(raw)
        #both = np.split(raw,2)
        #false_positives = both[0]
        #fake_positives = both[1]
        
        #full_true = []
        #rand_state = np.random.RandomState(444)
        #for fake in fake_positives:
        #    coadd_list = []
        #    for j in range(25):
        #        x_true = np.zeros((21, 21))
        #        x_true[10,10] = 1.
        #        x_true += rand_state.normal(0., 0.08, size=(21,21))
        #        x_gauss = gaussian_filter(x_true, np.random.random()+1)
        #        #x_gauss += rand_state.normal(0., 0.05, size=(21,21))
        #        coadd_list.append(x_gauss)
        #    fake += np.median(coadd_list,axis=0)
        
        #np.save('./fake_pos.npy', fake_positives)
        #np.save('./false_pos.npy', false_positives)
        
        fake_positives = np.load('./fake_pos.npy')
        false_positives = np.load('./false_pos.npy')
        
        full_dataset = [(torch.from_numpy(np.array([x])), 1) for x in fake_positives] + \
                       [(torch.from_numpy(np.array([x])), 0) for x in false_positives]
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        if (train):
            self.current_data = train_dataset
        else:
            self.current_data = test_dataset
        
    def __getitem__(self, index):
        return self.current_data[index] #(img, label)

    def __len__(self):
        return len(self.current_data)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, torch.float), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch kbmod Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        KbmodDataSet('./', train=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        KbmodDataSet('./', train=False),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        #torch.save(model.state_dict(),"kbmod_cnn.pt")
        torch.save(model,"kbmod_cnn_background.pt")
        
if __name__ == '__main__':
    main()