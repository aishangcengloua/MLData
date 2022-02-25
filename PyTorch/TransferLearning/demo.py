#numel()函数：返回数组中元素的个数
import numpy as np
import torch
arr = np.array(np.arange(10))
print(torch.tensor(arr, dtype = torch.float).numel())