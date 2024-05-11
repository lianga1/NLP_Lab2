from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset
import pandas as pd

import indx

word_to_idx, label_to_idx = idx.create_mappings('train.csv')

class CSVSequenceDataset(Dataset):
    def __init__(self, filename):
        # 读取CSV文件
        self.data = pd.read_csv(filename, encoding='utf-8')
        # 提取单词和标签列表
        self.words = self.data['word'].tolist()
        self.labels = self.data['expected'].tolist()
        
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        label = self.labels[idx]
        # 这里假设您有一个映射单词到索引的字典word_to_idx，
        # 以及一个映射标签到索引的字典label_to_idx
        # 如果没有，您需要先创建这两个映射
        return word_to_idx[word], label_to_idx[label]

# 示例：创建word_to_idx和label_to_idx映射（实际中需要基于数据集完整词汇和标签集来构建）

# 使用上面定义的Dataset类实例化数据集

# 创建DataLoader

def get_dataloader(train_name, dev_name, batch_size):
    train_dataset = CSVSequenceDataset(train_name)
    dev_dataset = CSVSequenceDataset(dev_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    return train_loader, dev_loader