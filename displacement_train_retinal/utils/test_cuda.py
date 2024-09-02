from __future__ import print_function  #这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse  # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
import torch     # 以下这几行导入相关的pytorch包，有疑问的参考我写的 Pytorch打怪路（一）系列博文
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
 
# Training settings 就是在设置一些参数，每个都有默认值，输入python main.py -h可以获得相关帮助
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', # batch_size参数，如果想改，如改成128可这么写：python main.py -batch_size=128
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',# test_batch_size参数，
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, # GPU参数，默认为False
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', # 跑多少次batch进行一次日志记录
                    help='how many batches to wait before logging training status')
args = parser.parse_args()  # 这个是使用argparse模块时的必备行，将参数进行关联，详情用法请百度 argparse 即可
args.cuda = not args.no_cuda and torch.cuda.is_available()  # 这个是在确认是否使用gpu的参数,比如
 
torch.manual_seed(args.seed) # 设置一个随机数种子，相关理论请自行百度或google，并不是pytorch特有的什么设置
if args.cuda:
    torch.cuda.manual_seed(args.seed) # 这个是为GPU设置一个随机数种子
 
 
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(       # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(        # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
 
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
model = Net()      # 实例化一个网络对象
if args.cuda:
    model.cuda()   # 判断是否调用GPU模式 
 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # 初始化优化器 model.train() 
 
def train(epoch):      # 定义每个epoch的训练细节
    model.train()      # 设置为trainning模式
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:  # 如果要调用GPU模式，就把数据转存到GPU
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)   # 把数据输入网络并得到输出，即进行前向传播
        loss = F.nll_loss(output, target)               # 计算损失函数  
        loss.backward()        # 反向传播梯度
        optimizer.step()       # 结束一次前传+反传之后，更新优化器参数
        if batch_idx % args.log_interval == 0:          # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.data[0].item()))
            print(batch_idx)
 
def test():
    model.eval()      # 设置为test模式  
    test_loss = 0     # 初始化测试损失值为0
    correct = 0       # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss 把所有loss值进行累加
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
 
    test_loss /= len(test_loader.dataset)   # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 
 
for epoch in range(1, args.epochs + 1): # 以epoch为单位进行循环
    train(epoch)
    #test()