import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size) # 全连接层1
        self.activate = nn.ReLU() # 激活函数
        self.fc2 = nn.Linear(hidden_size, 1) # 全连接层2

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x))) # 网络的前向传播


class NeuralUCBDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).cuda() # 创建神经网络
        self.context_list = [] # 上下文列表
        self.reward = [] # 奖励列表
        self.lamdba = lamdba # 正则化系数
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad) # 网络参数的数量
        self.U = lamdba * torch.ones((self.total_param,)).cuda() # U矩阵
        self.nu = nu # 乘数因子

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda() # 将上下文转换为张量
        # mu就是通过神经网络预测得到的当前的上下文对应的各个arm能得到的UCB的值
        mu = self.func(tensor) # 神经网络的输出
        g_list = [] # 梯度列表
        sampled = [] # 样本列表
        ave_sigma = 0
        ave_rew = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()]) # 计算梯度
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U # 计算标准差
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = fx.item() + sigma.item() # 计算采样值
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled) # 选择臂
        self.U += g_list[arm] * g_list[arm] # 更新U矩阵
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew # 返回选择的臂，梯度的范数，平均标准差和平均奖励

    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float()) # 添加上下文
        self.reward.append(reward) # 添加奖励
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba) # 定义优化器
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta # 计算损失函数
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000  # 如果已经训练了1000次，则返回平均损失
            if batch_loss / length <= 1e-3:
                return batch_loss / length  # 如果批次损失小于1e-3，则返回平均损失
