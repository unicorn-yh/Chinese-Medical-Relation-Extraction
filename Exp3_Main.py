"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""


from Exp3_Config import Training_Config
from Exp3_DataProc import train_data, val_data, test_data, id2relation, pre_embed
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model
import torch
import os
import json
from sklearn import metrics
import time


config = Training_Config()
device = 'cuda' if config.cuda else 'cpu'
idx2tag = id2relation()


def train(model, loader):
    correct_count = 0
    running_loss = 0.0
    for index, data in enumerate(loader):
        tensordata, label = data
        logits = model(tensordata)
        _, predict = logits.max(1)
        predict = predict.float()
        label = torch.squeeze(label)
        loss = loss_function(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct = (predict==label).sum().item()
        correct_count += correct
        acc = correct_count/(len(loader)*config.batch_size)
        loss = running_loss/(len(loader)*config.batch_size)
    print("Train acc:{:.4f}, Train loss:{:.4f}, ".format(acc,loss),end="")


def validation(model, loader, best_f1):
    correct_count = 0
    tags_true, tags_pred = [],[]
    for index, data in enumerate(loader):
        tensordata, label = data
        logits = model(tensordata)
        _, predict = logits.max(1)
        predict = predict.float()
        label = torch.squeeze(label)
        tags_true.extend(label.tolist())
        tags_pred.extend(predict.tolist())
        correct = (predict==label).sum().item()
        correct_count += correct
        valid_acc = correct_count/(len(loader)*config.batch_size)
    print("Valid acc:{:.4f}".format(valid_acc))

    f1 = metrics.f1_score(tags_true, tags_pred, average='weighted')
    if checkpoint_dict.get('epoch_f1'):
        checkpoint_dict['epoch_f1'][epoch] = f1
    else:
        checkpoint_dict['epoch_f1'] = {epoch: f1}
    if f1 > best_f1:
        best_f1 = f1
        checkpoint_dict['best_f1'] = best_f1
        checkpoint_dict['best_epoch'] = epoch
        torch.save(model.state_dict(), config.model_file)
    save_checkpoint(checkpoint_dict, config.checkpoint_file)


def predict(model, loader):
    tags_pred = []
    for index, data in enumerate(loader):
        model.eval()
        logits = model(data)
        predict = torch.argmax(logits,dim=1)
        tags_pred.extend(predict.tolist())
    with open("exp3_predict_labels_1820201040.txt","w") as file:
        for tag in tags_pred:
            file.write("%s\n" % tag)


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


def get_checkpoint(checkpoint_file, model, model_file):
    # load checkpoint if one exists
    print(checkpoint_file)
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_file))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0
    return checkpoint_dict, best_f1, epoch_offset

if __name__ == "__main__":
    print("训练模型......")

    # 训练集验证集
    train_dataset = train_data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)

    val_dataset = val_data
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, drop_last=True)

    # 测试集数据集和加载器
    test_dataset = test_data
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # 初始化模型对象
    Text_Model = TextCNN_Model(configs=config,pre_embedding=pre_embed).to(device)
    #Text_Model = SentenceRE(config).to(device)
    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    # 优化器设置
    optimizer = torch.optim.Adam(params=Text_Model.parameters(),lr=config.lr,weight_decay=1e-5)  # torch.optim中的优化器进行挑选，并进行参数设置
    
    # load checkpoint if one exists
    checkpoint_dict, best_f1, epoch_offset = get_checkpoint(config.checkpoint_file, Text_Model, config.model_file)

    # 训练和验证
    for epoch in range(config.epoch):
        print("Epoch {}".format(epoch+1),end=", ")
        train(Text_Model, loader=train_loader)
        #if epoch % config.num_val == 0:
        with torch.no_grad():
            validation(Text_Model, loader=val_loader, best_f1=best_f1)


    # 预测（测试）
    predict(Text_Model, test_loader)
