from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np


class Bandit_multi:
    def __init__(self, name, is_shuffle=True, seed=None):
        # 从OpenML平台获取数据
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # 避免出现NaN值，将NaN值设置为-1
            X[np.isnan(X)] = -1
            X = normalize(X)  # 对特征进行归一化
        elif name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
            # 避免出现NaN值，将NaN值设置为-1
            X[np.isnan(X)] = -1
            X = normalize(X)  # 对特征进行归一化
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            # 避免出现NaN值，将NaN值设置为-1
            X[np.isnan(X)] = -1
            X = normalize(X)  # 对特征进行归一化
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            # 避免出现NaN值，将NaN值设置为-1
            X[np.isnan(X)] = -1
            X = normalize(X)  # 对特征进行归一化
        else:
            raise RuntimeError('Dataset does not exist') # 如果数据集不存在，则引发运行时错误
        # 打乱数据
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=seed)
        else:
            self.X, self.y = X, y
        # 生成one-hot编码：
        self.y_arm = OrdinalEncoder(dtype=np.int).fit_transform(self.y.reshape((-1, 1))) # 对标签进行one-hot编码
        # 光标和其他变量
        self.cursor = 0  # 记录当前处理到的样本下标
        self.size = self.y.shape[0]  # 数据集的大小
        self.n_arm = np.max(self.y_arm) + 1  # 此数据集中的臂数
        self.dim = self.X.shape[1] * self.n_arm  # 特征向量的维度
        self.act_dim = self.X.shape[1] # 每个臂的特征向量的维度

    def step(self):
        assert self.cursor < self.size # 确保光标仍在数据集范围内
        X = np.zeros((self.n_arm, self.dim)) # 初始化特征向量
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim + self.act_dim] = self.X[self.cursor] # 填充特征向量的对应位置
        arm = self.y_arm[self.cursor][0] # 计算当前样本的臂
        rwd = np.zeros((self.n_arm,)) # 初始化奖励向量
        rwd[arm] = 1 # 将当前臂的奖励设置为1
        self.cursor += 1 # 将光标移动到下一个样本
        return X, rwd # 返回特征向量和奖励向量

    def finish(self):
        return self.cursor == self.size # 判断是否处理完整个数据集

    def reset(self):
        self.cursor = 0 # 将光标重置为0，以重新开始处理数据集