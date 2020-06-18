from libsvm.svmutil import *
from libsvm.svm import *
import random
import numpy as np
from sklearn.svm import SVC

# train 讀第一次取得max, min
"""
filename = 'train20.txt'
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
filename = 'train20.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        t_sum =  float(a[1][2:]) + float(a[2][2:]) +  float(a[3][2:])
        # normalize
        t1 = float(a[1][2:]) / t_sum
        t2 = float(a[2][2:]) / t_sum
        t3 = float(a[3][2:]) / t_sum
        tr_histogram.append(([t1, t2, t3], int(a[0])))

# test 讀第一次取得max, min
filename = 'test20.txt'
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
filename = 'test20.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        # normalize
        t_sum =  float(a[1][2:]) + float(a[2][2:]) +  float(a[3][2:])
        t1 = float(a[1][2:]) / t_sum
        t2 = float(a[2][2:]) / t_sum
        t3 = float(a[3][2:]) / t_sum
        te_histogram.append(([t1, t2, t3], int(a[0])))

'''
# write train data
seq = np.arange(0, len(tr_histogram))
np.random.shuffle(seq)
data = open("tr2.txt",'w+')
for i in seq:
    s = str(tr_histogram[i][1])+' 1:'+str(tr_histogram[i][0][0])+' 2:'+str(tr_histogram[i][0][1])+' 3:'+str(tr_histogram[i][0][2])
    print(s,file=data)
data.close()

# write test data
seq = np.arange(0, len(te_histogram))
np.random.shuffle(seq)
data = open("te2.txt",'w+')
for i in seq:
    s = str(te_histogram[i][1])+' 1:'+str(te_histogram[i][0][0])+' 2:'+str(te_histogram[i][0][1])+' 3:'+str(te_histogram[i][0][2])
    print(s,file=data)
data.close()
'''
"""
# use libsvm
"""
train_x = []
train_y = []
test_x = []
test_y = []
for temp in tr_histogram:
    train_x.append(temp[0])
    train_y.append(temp[1])
    print(temp)
for temp in te_histogram:
    test_x.append(temp[0])
    test_y.append(temp[1])
"""
y, x = svm_read_problem('train/train20_scale.txt')
yt, xt = svm_read_problem('test/test20_scale.txt')

#clf = SVC(kernel = 'linear', probability = True)
#clf.fit(train_x, train_y)
#print(clf.score(test_x, test_y))
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
gamma = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

model = svm_train(y, x, '-t 0')
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print('test:')
print(p_label)

"""
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
"""