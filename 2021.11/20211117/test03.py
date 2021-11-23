import torch
from torch import jit
from net import net_v1

if __name__ == '__main__':
    model = net_v1()
    model.load_state_dict(torch.load("param/10.pt"))

    #虚拟一个输入
    input = torch.randn(1,784)
    #将模型和权重打包
    tsm = jit.trace(model,input)
    tsm.save("mnist.pt")