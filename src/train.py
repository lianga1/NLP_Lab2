import Dataloader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import MyModel
from experiment_util import *
def main():
    # define models
    modelparam = {'model_name': 'naive_BiLSTM_CRF'}
    lossparam = {'loss_name': 'CrossEntropyLoss'}
    optimizerparam = {'optimizer_name': 'Adam', 'lr': 0.001}
    model,citerion,optimizer = get_model(modelparam,lossparam,optimizerparam)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    
    # define dataloader
    train_name = '/mnt/3T_disk/liangzuning/NLP_Lab2/dataset/train.csv'
    dev_name = '/mnt/3T_disk/liangzuning/NLP_Lab2/dataset/dev.csv'
    train_loader,dev_loader = Dataloader.get_dataloader(train_name=train_name, dev_name=dev_name, batch_size=32)
    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = model(input_ids, targets)
            loss.backward()
            optimizer.step()
        # 验证和评估逻辑可以在这里添加
    model.save('/mnt/3T_disk/liangzuning/NLP_Lab2/model/model.pth')
    
if __name__ == '__main__':
    main()