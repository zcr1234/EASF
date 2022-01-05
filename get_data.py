import torch
import pandas as pd
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data_label(data_path, data_label_path):
    data_info = pd.read_csv(data_path, header=None)
    label_info = pd.read_csv(data_label_path, header=None)
    data_info_arr = np.asarray(data_info)
    label_info_arr = np.asarray(label_info)
    train_data = torch.tensor(data_info_arr)
    label_data = torch.tensor(label_info_arr)
    train_data, label_data = train_data.to(device), label_data.to(device)
    return train_data, label_data




