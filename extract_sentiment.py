from snownlp import SnowNLP
from snownlp import seg
from snownlp import sentiment
from snownlp import normal

'''
s1 = '谢谢'
s2 = '非常谢谢'

s1 = SnowNLP(s1)
s2 = SnowNLP(s2)

print(s1.words)
print(s2.words)

print(s1.sentiments)
print(s2.sentiments)
'''
import csv
zero_num = 0
two_num = 0
f_label = open("sen_label.csv", 'w', encoding='utf-8', newline='')
f_data = open("sen_data_embed.txt", 'w', encoding='gbk')
label_writer = csv.writer(f_label)
i = 0
with open("origin.txt", 'r', encoding='gbk') as f:
    for j in range(32000):
        line1 = f.readline()
        line2 = f.readline()
        line3 = f.readline()
        line4 = f.readline()
        line5 = f.readline()
        line = SnowNLP(line5)
        score = line.sentiments
        if score < 0.15:
            zero_num += 1
            lst = [0]
            label_writer.writerow(lst)
            f_data.write(line1)
            f_data.write(line2)
            f_data.write(line3)
            f_data.write(line4)
            # f_data.write(line5)
            # f_data.write('0')
            # f_data.write("\n")
    
        if score > 0.57:
            two_num += 1
            lst = [1]
            label_writer.writerow(lst)
            f_data.write(line1)
            f_data.write(line2)
            f_data.write(line3)
            f_data.write(line4)
            # f_data.write(line5)
            # f_data.write('1')
            # f_data.write("\n")
        i += 1
        print(i)
print(zero_num)
print(two_num)

