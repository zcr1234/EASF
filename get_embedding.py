import keras
from keras_bert import extract_embeddings
import csv
import torch
from torch.nn import LSTM

model_path = 'bert'  # path of bert file
fp = open('sen_data_embed.txt', 'r', encoding='gbk')
fq = open("sentence_embedding.csv", 'w', encoding='utf-8', newline='')
lstm = LSTM(input_size=768, hidden_size=768)
fq_writer = csv.writer(fq)
lst = []
for fp_line in fp.readlines():
    fp_line = fp_line.strip('\n')
    lst.append(fp_line)

h0 = torch.rand(1, 1, 768)
c0 = torch.rand(1, 1, 768)

for sentence in lst:
    wo_embed = extract_embeddings(model_path, sentence, output_layer_num=1)
    len_embed = wo_embed.size()[0]
    hidden = torch.zeros(768)
    for i in range(len_embed):
        out, hidden = lstm(wo_embed[i].view(1, 1, 768), (h0, c0))
    hidden = hidden.view(768)
    fq_writer.writerow(hidden)

fp.close()
fq.close()

