"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""


from Exp3_Config import Training_Config
from Exp3_DataProc import train_data, val_data, test_data, id2relation
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model
import torch


def train(model, loader):
    for index, data in enumerate(loader):
        model(data['text'])


def validation(model, loader):
    for index, data in enumerate(loader):
        model(data['head'])


def predict(model, loader):
    for index, data in enumerate(loader):
        model(data['tail'])


if __name__ == "__main__":
    config = Training_Config()

    # 训练集验证集
    train_dataset = train_data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)

    val_dataset = val_data
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size)

    # 测试集数据集和加载器
    test_dataset = test_data
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # 初始化模型对象
    Text_Model = TextCNN_Model(configs=config)
    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    # 优化器设置
    optimizer = torch.optim.Adam(params=Text_Model.parameters())  # torch.optim中的优化器进行挑选，并进行参数设置

    # 训练和验证
    for i in range(config.epoch):
        train(Text_Model, loader=train_loader)
        if i % config.num_val == 0:
            validation(Text_Model, loader=val_loader)

    # 预测（测试）
    predict(Text_Model, test_loader)
