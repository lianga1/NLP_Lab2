# main.py
import MyModel
import torch.optim
import torch.nn
def select_model(model_param):
    if model_param['model_name'] == 'naive_BiLSTM_CRF':
        return MyModel.BiLSTM_CRF(vocab_size=1, embedding_dim=100, hidden_dim=200, num_tags=17, dropout=0.5)
    # elif model_param == 'ModelB':
    #     return MyModel.ModelB()
    else:
        raise ValueError("Unsupported model name")

def select_loss(loss_param):
    if loss_param['loss_name'] == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss function name")

def select_optimizer(opt_param,model):
    if opt_param['optimizer_name'] == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=opt_param['lr'])
    elif opt_param['optimizer_name'] == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=opt_param['lr'])
    else:
        raise ValueError("Unsupported optimizer name")
def get_model(model_param, loss_param,opt_param):
    model = select_model(model_param)
    loss = select_loss(loss_param)
    optmizer = select_optimizer(opt_param,model)
    return model, loss, optmizer
