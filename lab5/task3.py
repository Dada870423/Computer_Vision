from libsvm.svmutil import *
from libsvm.svm import *
import random
import numpy as np
from sklearn.svm import SVC
import sys

n_cluster=100

# 讀第二次存成tr_histogram
tr_histogram = []
filename = 'train/train100.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        t_sum = 0
        for i in range(1, len(a)):
            colon = a[i].split(":")
            t_sum += float(colon[1])
        # normalize
        tt = np.zeros(len(a) - 1)
        for i in range(1, len(a)):
            colon = a[i].split(":")
            tt[i-1] = float(colon[1]) / t_sum
        tr_histogram.append((tt, int(a[0])))


# 讀第二次存成tr_histogram
te_histogram = []
filename = 'test/test100.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        t_sum = 0
        for i in range(1, len(a)):
            colon = a[i].split(":")
            t_sum += float(colon[1])
        # normalize
        tt = np.zeros(len(a) - 1)
        for i in range(1, len(a)):
            colon = a[i].split(":")
            tt[i-1] = float(colon[1]) / t_sum
        te_histogram.append((tt, int(a[0])))


# write train data
seq = np.arange(0, len(tr_histogram))
data = open("train/train100_add_1.txt",'w+')
for i in seq:
    s = str(tr_histogram[i][1])
    for j in range(n_cluster):
        s = s +" " + str(j+1) + ":" + str(tr_histogram[i][0][j])
    print(s, file=data)
data.close()



# write test data
seq = np.arange(0, len(te_histogram))
data = open("test/test100_add_1.txt",'w+')
for i in seq:
    s = str(te_histogram[i][1])
    for j in range(n_cluster):
        s = s +" " + str(j+1) + ":" + str(te_histogram[i][0][j])
    print(s, file=data)
data.close()


# use libsvm
y, x = svm_read_problem('train/train100_scale.txt')
yt, xt = svm_read_problem('test/test100_scale.txt')

#run grid
C = [0.001, 0.01, 0.1, 1, 10, 100]
gamma = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]

model = svm_train(y, x, '-t 0')
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print('test:')
print(p_label)


best_acc = 0
best_para = (0, 0)
for cidx in C:
    for gidx in gamma:
        parameter = '-t 0 ' + '-c ' + str(cidx) + ' -g ' + str(gidx)
        print(parameter)
        model = svm_train(y, x, parameter)
        p_label, p_acc, p_val = svm_predict(yt, xt, model)
        print('test:')
        print(p_label)
        if max(p_acc) > best_acc:
            best_acc = max(p_acc)
            best_para = (cidx, gidx)
print("best_acc:", best_acc)
print("best_para:", best_para)
