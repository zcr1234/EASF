import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


# 获取seq注意力的输出，用第一句到第四句
def seq_soft_attention(attention_result, seq_weight):
    result_seq = attention_result[:, 0, :] * seq_weight[0] + attention_result[:, 1, :] * seq_weight[1] \
                 + attention_result[:, 2, :] * seq_weight[2] + attention_result[:, 3, :] * seq_weight[3]
    return result_seq


# 获取pair注意力的输出,只用到第一句与第三句
def pair_soft_attention(attention_result, pair_weight):
    result_pair = attention_result[:, 0, :] * pair_weight[0] + attention_result[:, 2, :] * pair_weight[1]
    return result_pair


# sentiment与emotion的乘积softmax
def cross_mul(result_s, result_e):
    # print(result_s.shape)
    # print(result_e.shape)
    result_s = result_s.reshape(-1, 2)
    result_e = result_e.reshape(-1, 8)
    batch_len = result_s.size()[0]
    z = torch.zeros(batch_len * 2).reshape(batch_len, 2).to(device)
    for k in range(batch_len):
        for h in range(2):
            for j in range(8):
                z[k, h] += result_s[k, h] * result_e[k, j]
    return z


# 第三层注意力权重分配
def cross_third(pair_result, seq_result, weight):
    result_cross = weight[0] * pair_result + weight[1] * seq_result
    return result_cross


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_first_feedforward = nn.Linear(768, 96, bias=False)
        self.s_second_feedforward = nn.Linear(96, 48, bias=False)
        self.s_third_feedforward = nn.Linear(48, 24, bias=False)
        self.s_four_feedforward = nn.Linear(24, 12, bias=False)
        self.s_out_feedforward = nn.Linear(12, 2, bias=False)
        self.e_first_feedforward = nn.Linear(768, 96, bias=False)
        self.e_second_feedforward = nn.Linear(96, 48, bias=False)
        self.e_third_feedforward = nn.Linear(48, 24, bias=False)
        self.e_four_feedforward = nn.Linear(24, 12, bias=False)
        self.e_out_feedforward = nn.Linear(12, 8, bias=False)
        self.weight = torch.tensor([0.01])

        # 获得sentiment第二层注意力的权重
        self.s_seq_rand = nn.Parameter(torch.tensor(([0.25, 0.25, 0.25, 0.25]), requires_grad=True))
        # nn.init.normal_(self.s_seq_rand, 0.25, 0.1)
        self.s_pair_rand = nn.Parameter(torch.tensor(([0.5, 0.5]), requires_grad=True))
        # nn.init.normal_(self.s_pair_rand, 0.5, 0.1)
        # 获得emotion第二层注意力的权重
        self.e_seq_rand = nn.Parameter(torch.tensor(([0.25, 0.25, 0.25, 0.25]), requires_grad=True))
        # nn.init.normal_(self.e_seq_rand, 0.25, 0.1)
        self.e_pair_rand = nn.Parameter(torch.tensor(([0.5, 0.5]), requires_grad=True))
        # nn.init.normal_(self.e_pair_rand, 0.5, 0.1)
        # 获得第三层注意力权重
        self.cross_weight = nn.Parameter(torch.tensor(([0.5, 0.5]), requires_grad=True))
        # nn.init.normal_(self.cross_weight, 0.5, 0.1)
        self.weight = torch.tensor([0.01])
        # print(id(self.cross_weight_r) == id(self.cross_weight))

    def forward(self, result):
        # pair_wise 部分的emotion
        pair_e = pair_soft_attention(result, self.e_pair_rand)
        pair_e_relu = F.relu(pair_e)
        pair_e_drop = F.dropout(pair_e_relu, 0.7)
        pair_e_f = self.e_first_feedforward(pair_e_drop)
        pair_e_f_relu = F.relu(pair_e_f)
        pair_e_f_drop = F.dropout(pair_e_f_relu, 0.7)
        pair_e_s = self.e_second_feedforward(pair_e_f_drop)
        pair_e_s_relu = F.relu(pair_e_s)
        pair_e_s_drop = F.dropout(pair_e_s_relu, 0.7)
        pair_e_t = self.e_third_feedforward(pair_e_s_drop)
        pair_e_t_relu = F.relu(pair_e_t)
        pair_e_t_drop = F.dropout(pair_e_t_relu, 0.7)
        pair_e_f = self.e_four_feedforward(pair_e_t_drop)
        pair_e_f_relu = F.relu(pair_e_f)
        pair_e_f_drop = F.dropout(pair_e_f_relu, 0.7)
        pair_e_out = self.e_out_feedforward(pair_e_f_drop)
        # pair_wise 部分的sentiment
        pair_s = pair_soft_attention(result, self.s_pair_rand)
        pair_s_relu = F.relu(pair_s)
        pair_s_drop = F.dropout(pair_s_relu, 0.7)
        pair_s_f = self.s_first_feedforward(pair_s_drop)
        pair_s_f_relu = F.relu(pair_s_f)
        pair_s_f_drop = F.dropout(pair_s_f_relu, 0.7)
        pair_s_s = self.s_second_feedforward(pair_s_f_drop)
        pair_s_s_relu = F.relu(pair_s_s)
        pair_s_s_drop = F.dropout(pair_s_s_relu, 0.7)
        pair_s_t = self.s_third_feedforward(pair_s_s_drop)
        pair_s_t_relu = F.relu(pair_s_t)
        pair_s_t_drop = F.dropout(pair_s_t_relu, 0.7)
        pair_s_f = self.s_four_feedforward(pair_s_t_drop)
        pair_s_f_relu = F.relu(pair_s_f)
        pair_s_f_drop = F.dropout(pair_s_f_relu, 0.7)
        pair_s_out = self.s_out_feedforward(pair_s_f_drop)
        # seq_wise 部分的emotion
        seq_e = seq_soft_attention(result, self.e_seq_rand)
        seq_e_relu = F.relu(seq_e)
        seq_e_drop = F.dropout(seq_e_relu, 0.7)
        seq_e_f = self.e_first_feedforward(seq_e_drop)
        seq_e_f_relu = F.relu(seq_e_f)
        seq_e_f_drop = F.dropout(seq_e_f_relu, 0.7)
        seq_e_s = self.e_second_feedforward(seq_e_f_drop)
        seq_e_s_relu = F.relu(seq_e_s)
        seq_e_s_drop = F.dropout(seq_e_s_relu, 0.7)
        seq_e_t = self.e_third_feedforward(seq_e_s_drop)
        seq_e_t_relu = F.relu(seq_e_t)
        seq_e_t_drop = F.dropout(seq_e_t_relu, 0.7)
        seq_e_f = self.e_four_feedforward(seq_e_t_drop)
        seq_e_f_relu = F.relu(seq_e_f)
        seq_e_f_drop = F.dropout(seq_e_f_relu, 0.7)
        seq_e_out = self.e_out_feedforward(seq_e_f_drop)
        # seq_wise 部分的sentiment
        seq_s = seq_soft_attention(result, self.s_seq_rand)
        seq_s_relu = F.relu(seq_s)
        seq_s_drop = F.dropout(seq_s_relu, 0.7)
        seq_s_f = self.s_first_feedforward(seq_s_drop)
        seq_s_f_relu = F.relu(seq_s_f)
        seq_s_f_drop = F.dropout(seq_s_f_relu, 0.7)
        seq_s_s = self.s_second_feedforward(seq_s_f_drop)
        seq_s_s_relu = F.relu(seq_s_s)
        seq_s_s_drop = F.dropout(seq_s_s_relu, 0.7)
        seq_s_t = self.s_third_feedforward(seq_s_s_drop)
        seq_s_t_relu = F.relu(seq_s_t)
        seq_s_t_drop = F.dropout(seq_s_t_relu, 0.7)
        seq_s_f = self.s_four_feedforward(seq_s_t_drop)
        seq_s_f_relu = F.relu(seq_s_f)
        seq_s_f_drop = F.dropout(seq_s_f_relu, 0.7)
        seq_s_out = self.s_out_feedforward(seq_s_f_drop)
        # pair_wise 部分的sentiment与emotion相乘
        pair_cross = cross_mul(pair_s_out, pair_e_out)
        # seq_wise 部分的sentiment与emotion相乘
        seq_cross = cross_mul(seq_s_out, seq_e_out)
        # 计算pair_wise与seq_wise的相乘
        final_out = cross_third(pair_cross, seq_cross, self.cross_weight)
        return final_out
