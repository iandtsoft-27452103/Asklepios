import torch
import torch.nn as nn
import torch.nn.functional as F

#���̓`���l��
#�Տ�̋�                   8  ��, ��, �j, ��, ��, �p, ��, ��
#�Տ�̐����               6  ��, ����, ���j, ����, �n, ��
#������̕�                 6
#������̍�                 4
#������̌j                 4
#������̋�                 4
#������̋�                 4
#������̊p                 2
#������̔�                 2
#���ʂ̋ߖT8�}�X�ɂ������  2
#���ɋ��₪3���ȏ゠�邩  2
#���Ɍj��2���ȏ゠�邩    2
#���ɍ���2���ȏ゠�邩    2
#���؂ꂩ                   2
#����������������Ă��邩   2
#���������n������Ă��邩   2
#���ʂɉ��肪�������Ă��邩 2
#�����6���ȏ㎝���Ă��邩  2
#�����G�w�ɂ��邩           2
#���ʂ�8�ߖT�̔n�E�p�̗���  2
#�G�ʂ�8�ߖT�̔n�E�p�̗���  2
#�v                        52
#��㍇�v                 104
#���                       1
#���v                     105

ch = 256
fcl = 256

#bias�N���X��dlshogi����̈ڐA
class bias(nn.Module):
    def __init__(self, shape):
        super(bias, self).__init__()
        self.bias=nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        return input + self.bias

class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv1 = nn.LazyConv2d(out_channels = ch, kernel_size = 3, padding = 1, bias = False)
        self.norm1 = nn.LazyBatchNorm2d(eps = 2e-05)
        self.conv2 = nn.LazyConv2d(out_channels = ch, kernel_size = 3, padding = 1, bias = False)
        self.norm2 = nn.LazyBatchNorm2d(eps = 2e-05)

    def forward(self, x):
        h1 = F.relu(self.norm1(self.conv1(x)))
        h2 = self.norm2(self.conv2(h1))
        return F.relu(x + h2)

class value_lazy(nn.Module):
    def __init__(self):
        super(value_lazy, self).__init__()
        self.l1 = nn.LazyConv2d(ch, 5, 1, 2, bias=False)
        self.n1 = nn.LazyBatchNorm2d(eps = 2e-05)
        self.b1 = block()
        self.b2 = block()
        self.b3 = block()
        self.b4 = block()
        self.b5 = block()
        self.b6 = block()
        self.b7 = block()
        self.b8 = block()
        self.b9 = block()
        self.b10 = block()
        self.l2 = nn.LazyConv2d(32, 1, 1, 0, 1, 1, True)
        self.n2 = nn.LazyBatchNorm2d(eps = 2e-05)
        self.v1 = nn.Linear(81 * 32, fcl)
        self.v2 = nn.Linear(fcl, 1)

    def forward(self, x):
        h = F.relu(self.n1(self.l1(x))) 
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        h = self.b6(h)
        h = self.b7(h)
        h = self.b8(h)
        h = self.b9(h)
        h = self.b10(h)
        h = F.relu(self.n2(self.l2(h)))
        h = h.reshape(self.batch_size, 32, 81)
        h = h.reshape(self.batch_size, 81 * 32)
        h = F.relu(self.v1(h))
        h = self.v2(h)
        h = h.reshape(-1)
        return h