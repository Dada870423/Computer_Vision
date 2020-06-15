from libsvm.python.svmutil import *
from libsvm.python.svm import *
import random
import numpy as np

# train 讀第一次取得max, min
filename = 'train.txt'
data_max = 0
data_min = 0
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        t1 = float(a[1][2:])
        t2 = float(a[2][2:])
        t3 = float(a[3][2:])
        if max(t1,t2,t3) > data_max :
            data_max = max(t1, t2, t3)
        if min(t1,t2,t3) < data_min :
            data_min = min(t1, t2, t3)
# 讀第二次存成tr_histogram
tr_histogram = []
filename = 'train.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        # normalize
        t1 = (float(a[1][2:]) - data_min) / (data_max - data_min)
        t2 = (float(a[2][2:]) - data_min) / (data_max - data_min)
        t3 = (float(a[3][2:]) - data_min) / (data_max - data_min)
        tr_histogram.append(([t1, t2, t3], int(a[0])))

# test 讀第一次取得max, min
filename = 'test.txt'
data_max = 0
data_min = 0
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        t1 = float(a[1][2:])
        t2 = float(a[2][2:])
        t3 = float(a[3][2:])
        if max(t1,t2,t3) > data_max :
            data_max = max(t1, t2, t3)
        if min(t1,t2,t3) < data_min :
            data_min = min(t1, t2, t3)
            # 讀第二次存成tr_histogram
te_histogram = []
filename = 'test.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        # normalize
        t1 = (float(a[1][2:]) - data_min) / (data_max - data_min)
        t2 = (float(a[2][2:]) - data_min) / (data_max - data_min)
        t3 = (float(a[3][2:]) - data_min) / (data_max - data_min)
        te_histogram.append(([t1, t2, t3], int(a[0])))


# write train data
seq = np.arange(0, len(tr_histogram))
np.random.shuffle(seq)
data = open("tr.txt",'w+')
for i in seq:
    s = str(tr_histogram[i][1])+' 1:'+str(tr_histogram[i][0][0])+' 2:'+str(tr_histogram[i][0][1])+' 3:'+str(tr_histogram[i][0][2])
    print(s,file=data)
data.close()

# write test data
seq = np.arange(0, len(te_histogram))
np.random.shuffle(seq)
data = open("te.txt",'w+')
for i in seq:
    s = str(te_histogram[i][1])+' 1:'+str(te_histogram[i][0][0])+' 2:'+str(te_histogram[i][0][1])+' 3:'+str(te_histogram[i][0][2])
    print(s,file=data)
data.close()

# use libsvm
y, x = svm_read_problem('tr.txt')
yt, xt = svm_read_problem('te.txt')
model = svm_train(y, x)
print('test:')
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print(p_label)