import copy
import time
import torch

from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from model_net import Net
from get_data import get_data_label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=10)
data_path = 'F://second//data//data_15000.csv'
label_path = 'F://second//data//label_15000.csv'
train_batch_size = 100
test_batch_size = 100
epochs = 50


def train(model: nn.Module, train_data_loader):
    model.train()
    start_time = time.time()
    log_interval = 0
    total_loss = 0
    for tr_data, tr_label in train_data_loader:
        right_num = 0
        right_one = 0
        right_zero = 0
        log_interval += 1
        output = model(tr_data).view(-1, 2)
        y = torch.argmax(output, dim=1)
        for i in range(train_batch_size):
            if tr_label[i] == y[i]:
                right_num += 1
                if y[i] == 0:
                    right_zero += 1
                else:
                    right_one += 1
        loss = criterion(output, tr_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss

        if log_interval % 20 == 0:
            # print(right_num / train_batch_size)
            print(f'total: {right_num}, zero: {right_zero}, one: {right_one}')
    print(f'train_loss: {total_loss/len(train_loader)}, time:{time.time()-start_time}')


def evaluate(model: nn.Module, eval_loader):
    model.eval()
    total_loss = 0
    result_num = 0
    result_zero = 0
    result_one = 0
    log_interval = 0
    with torch.no_grad():
        for v_data, v_label in eval_loader:
            log_interval += 1
            output = model(v_data).view(-1, 2)
            loss = criterion(output, v_label)
            output_result = torch.argmax(output, dim=1)
            for i in range(test_batch_size):
                # print(output_result[i], v_label[i])
                if output_result[i] == v_label[i]:
                    result_num += 1
                    if v_label[i] == 0:
                        result_zero += 1
                    else:
                        result_one += 1
            total_loss += test_batch_size * loss.item()
            if log_interval % 10 == 0:
                print(f'eval_loss:{loss}')
    print(f'total: {result_num}, zero: {result_zero}, one: {result_one}')
    print(f'准确率：{result_num/(len(eval_loader) * train_batch_size)}')
    return total_loss / len(eval_loader)


train_data, label_data = get_data_label(data_path, label_path)
label_data = label_data.view(-1).to(torch.long)

for train_index, test_index in kf.split(train_batch_size):
    train_data, test_data = train_data[train_index], train_data[test_index]
    train_label, test_label = label_data[train_index], label_data[test_index]

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)

    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    my_model = Net()
    my_model = my_model.to(device)
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        # print(my_model.deep_learn.first_transformer_encoder_1.data.weight)
        # print(my_model.deep_learn.nine_transformer_encoder_1.data.weight)
        train(my_model, train_loader)
        test_loss = evaluate(my_model, test_loader)
        elapsed = time.time() - epoch_start_time
        print(f'epoch: {epoch}, elapsed: {elapsed}')
        scheduler.step()

