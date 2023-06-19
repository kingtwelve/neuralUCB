import numpy as np
from data_multi import Bandit_multi # 导入多臂老虎机数据集
from learner_diag import NeuralUCBDiag # 导入NeuralUCB算法
import argparse # 导入命令行参数解析库
import pickle # 导入pickle模块，用于序列化和反序列化Python对象
import os # 导入操作系统模块，用于操作文件和目录
import time # 导入时间模块
import torch # 导入PyTorch库

if __name__ == '__main__':
    torch.set_num_threads(8) # 设置PyTorch的线程数
    torch.set_num_interop_threads(8) # 设置PyTorch的互操作线程数
    parser = argparse.ArgumentParser(description='NeuralUCB') # 创建命令行参数解析器

    # 添加命令行参数
    parser.add_argument('--size', default=15000, type=int, help='bandit size')
    parser.add_argument('--dataset', default='mnist', metavar='DATASET')
    parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not')
    parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
    parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')  #
    parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')  # 正则化系数
    parser.add_argument('--hidden', type=int, default=100, help='network hidden size')  # upn的隐藏层数目

    args = parser.parse_args()  # 解析命令行参数
    use_seed = None if args.seed == 0 else args.seed
    b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)  # 创建多臂老虎机数据集
    bandit_info = '{}'.format(args.dataset)
    l = NeuralUCBDiag(b.dim, args.lamdba, args.nu, args.hidden)  # 创建NeuralUCB算法实例
    ucb_info = '_{:.3e}_{:.3e}_{}'.format(args.lamdba, args.nu, args.hidden)

    regrets = []  # 损失列表
    summ = 0  # 总损失
    for t in range(min(args.size, b.size)):
        context, rwd = b.step()  # 获取上下文和奖励
        arm_select, nrm, sig, ave_rwd = l.select(context)  # 选择臂并计算梯度范数、平均标准差和平均奖励
        r = rwd[arm_select]  # 获取选择臂的奖励
        reg = np.max(rwd) - r  # 计算损失
        summ += reg  # 更新总损失
        if t < 2000:
            loss = l.train(context[arm_select], r)  # 在前2000步中进行训练
        else:
            if t % 100 == 0:
                loss = l.train(context[arm_select], r)  # 每100步训练一次
        regrets.append(summ)  # 将总损失添加到损失列表中
        if t % 100 == 0:
            print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, summ, loss, nrm, sig, ave_rwd))  # 打印每100步的结果

    path = '{}_{}_{}'.format(bandit_info, ucb_info, time.time())  # 创建文件路径
    fr = open(path, 'w')  # 打开文件
    for i in regrets:
        fr.write(str(i))
        fr.write("\n")
    fr.close()  # 关闭文件
