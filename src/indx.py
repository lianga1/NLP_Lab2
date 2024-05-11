import pandas as pd
from collections import defaultdict

def create_mappings(filename):
    """
    从CSV文件中创建单词到索引（word_to_idx）和标签到索引（label_to_idx）的映射。
    
    :param filename: CSV文件路径
    :return: (word_to_idx, label_to_idx) 映射字典
    """
    word_counts = defaultdict(int)  # 用于统计单词出现次数，便于后续排序
    label_set = set()  # 用于收集所有不同的标签
    
    # 读取CSV文件
    data = pd.read_csv(filename, encoding='utf-8')
    
    # 统计单词和收集标签
    for word, label in zip(data['word'], data['expected']):
        word_counts[word] += 1
        label_set.add(label)
    
    # 构建word_to_idx映射，这里简单按出现频率排序，实际中可能还需考虑其他因素
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    word_to_idx = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    # 确保未知单词（UNK）和padding（PAD）有预留的索引，通常UNK为1，PAD为0
    word_to_idx['<UNK>'] = len(word_to_idx)  # 假定未出现的单词索引
    word_to_idx['<PAD>'] = 0  # 填充符号的索引
    
    # 构建label_to_idx映射
    label_list = sorted(list(label_set))
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    
    return word_to_idx, label_to_idx

# 使用函数创建映射
if __name__ == '__main__':  
    word_to_idx, label_to_idx = create_mappings('train.csv')

    # 打印映射查看结果（实际使用时可能非常长，这里仅作示例）
    print("Word to Index Mapping:")
    for word, idx in list(word_to_idx.items())[:10]:  # 打印前10个示例
        print(f"{word}: {idx}")

    print("\nLabel to Index Mapping:")
    for label, idx in label_to_idx.items():
        print(f"{label}: {idx}")