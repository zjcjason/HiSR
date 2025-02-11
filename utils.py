import torch
import logging
import random
import os
import numpy as np
from torch.optim import lr_scheduler


class Saver:
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


def slot2depth(slot):
    hiera = torch.load(slot)

    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth

    value2slot = {}
    num_class = 0
    for s in hiera:
        for v in hiera[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1

    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
    # torch.save(depth2label, 'depth2label.pt')
    return depth2label


def setup_logging(file_path):
    # 创建一个logger
    logger = logging.getLogger('my_htc')
    logger.setLevel(logging.DEBUG)  # 设置日志的全局级别

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # 创建另一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s -%(levelname)s - %(message)s')

    # 设置handler的格式
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将handler添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def seed_everything(seed=42):
    # Python RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

    # CuDNN行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 避免哈希算法的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)


class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                 (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)

        self.cur += 1

        return [lr for base_lr in self.base_lrs]