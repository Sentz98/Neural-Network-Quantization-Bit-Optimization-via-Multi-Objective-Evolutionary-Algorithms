import torch
import torch.nn.functional as F
from quantization import *
from model import *
from quantization import model_size


def train(args, model, device, train_loader, optimizer, criterion,  epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if criterion == F.nll_loss:
                test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            else:
                test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def testQuant(model, device, loaders, criterion, quant=False, num_bits=8):

    train_loader, test_loader = loaders        

    if not quant:
        num_bits = 32

    stats = gatherStats(model, train_loader, device)
    if isinstance(num_bits, list):
        assert len(num_bits) == len(stats)
        for i, (k, _) in enumerate(stats.items()):
            stats[k]['bits'] = num_bits[i]
    else:
        for k, _ in stats.items():
            stats[k]['bits'] = num_bits
    # print(stats)
    mo = model_size(stats)
    # print('Model size: ', mo)
    
    
    model.eval().to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
                output = model.Qforward(data, stats)
            else:
              output = model(data)
            if criterion == F.nll_loss:
                test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            else:
                test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    
    return test_loss, correct / len(test_loader.dataset), mo


