import torch
import torch.nn as nn
import torch.nn.functional as F

#入力チャネル
#盤上の駒                   8  歩, 香, 桂, 銀, 金, 角, 飛, 玉
#盤上の成り駒               6  と, 成香, 成桂, 成銀, 馬, 龍
#持ち駒の歩                 6
#持ち駒の香                 4
#持ち駒の桂                 4
#持ち駒の銀                 4
#持ち駒の金                 4
#持ち駒の角                 2
#持ち駒の飛                 2
#自玉の近傍8マスにある金銀  2
#駒台に金銀が3枚以上あるか  2
#駒台に桂が2枚以上あるか    2
#駒台に香が2枚以上あるか    2
#歩切れか                   2
#自分だけ龍を作っているか   2
#自分だけ馬を作っているか   2
#自玉に王手がかかっているか 2
#金銀を6枚以上持っているか  2
#龍が敵陣にあるか           2
#自玉の8近傍の馬・角の利き  2
#敵玉の8近傍の馬・角の利き  2
#計                        52
#先後合計                 104
#手番                       1
#総計                     105

ch = 256
fcl = 256

#biasクラスはdlshogiからの移植
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