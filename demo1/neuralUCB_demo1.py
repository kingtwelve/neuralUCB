###################################################################################
#
# Author: Meng Wang
# Date: 2023-06-19, 周一, 11:22
#
###################################################################################
import numpy as np
import torch
import torch.nn as nn
import subprocess


class NeuralUCB:
    def __init__(self, num_arms, num_features, alpha=1, hidden_size=64):
        """

        :param int num_arms:可以选择的动作的数量
        :param int num_features: 上下文特征的数量
        :param float alpha: 控制 利用和探索 所占比重的参数
        :param int hidden_size: 隐藏神经单元的数量（宽度）
        """
        self.num_arms = num_arms
        self.num_features = num_features
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.t = 0

        # 初始化神经网络
        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_arms)
        )

        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # 初始化数据缓存
        self.contexts = []
        self.rewards = []
        self.actions = []

    def select_action(self, context):
        # 将上下文转换为张量
        context = torch.tensor(context, dtype=torch.float32)

        # 使用神经网络预测每个臂的奖励
        scores = self.model(context)
        ucbs = scores + self.alpha * torch.sqrt(torch.log(self.t + 1) / torch.sum(self.actions, dim=0))

        # 选择UCB最大的臂
        action = torch.argmax(ucbs).item()

        # 将选择的臂加入数据缓存
        self.contexts.append(context)
        self.actions.append(np.eye(self.num_arms)[action])
        return action

    def update(self, reward):
        # 将奖励添加到数据缓存
        self.rewards.append(reward)

        # 损失函数为负的平均奖励
        loss = -torch.mean(torch.tensor(self.rewards, dtype=torch.float32))

        # 使用反向传播更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新时间步骤
        self.t += 1

    def reset(self):
        # 重置数据缓存和时间步骤
        self.contexts = []
        self.rewards = []
        self.actions = []
        self.t = 0

    def compute_reward(self, action, context, ee_model, acc_required, latency_required, alpha=0.5, beta=0.5):
        acc = ee_model.get_acc()
        latency = ee_model.get_latency()
        if acc < acc_required:
            reward = acc - 100
        else:
            if latency < latency_required:
                reward = alpha * latency + beta * acc
            else:
                reward = beta * acc
        return reward


class Context:
    def __init__(self):
        self.load = 11

    @property
    def get_gpu_load(self):
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        gpu_load = float(output.decode("utf-8").strip().split("\n")[0])
        return gpu_load

    def get_cpu_load(self):
        output = subprocess.check_output(["cat", "/proc/stat"])
        lines = output.decode("utf-8").strip().split("\n")
        cpu_times = lines[0].split()[1:]
        cpu_times = [int(time) for time in cpu_times]
        idle_time = cpu_times[3]
        total_time = sum(cpu_times)
        cpu_load = 100.0 * (1.0 - idle_time / total_time)
        return cpu_load


if __name__ == "__main__":
    # 创建一个NeuralUCB实例
    num_arms = 10
    num_features = 2
    neural_ucb = NeuralUCB(num_arms, num_features)

    # 迭代选择早退点和处理单元
    num_iterations = 1000
    for i in range(num_iterations):
        # 生成上下文
        context = np.random.rand(num_features)

        # 选择早退点和处理单元
        action = neural_ucb.select_action(context)

        # 计算奖励
        reward = neural_ucb.compute_reward(action, context)

        # 更新参数
        neural_ucb.update(reward)

    # 重置数据缓存
    neural_ucb.reset()
