import numpy as np
import cv2
import os, sys
from matplotlib import pyplot as plt

def l2_distance (a,b):
    if a.shape != b.shape :
        print("illegal move")
    else:
        return np.linalg.norm(a-b)

def find_nn(train_list,test,k):   #return list of tuples(dist,category)
    test_feature,test_cat = test
    new_list=[] #return tuple of distance and category
    for i in train_list:
        train_feature,train_category=i
        dist = l2_distance(np.asarray(train_feature),np.asarray(test_feature))
        new_list.append((dist,train_category))
    new_list = sorted(new_list,key=lambda x : x[0]) #sort by distance
    return new_list[:k]

def get_distance_list(train_list,test,k):   #return list of tuples(dist,category)
    test_feature,test_cat = test
    new_list=[] #return tuple of distance and category
    for i in train_list:
        train_feature,train_category=i
        dist = l2_distance(np.asarray(train_feature),np.asarray(test_feature))
        new_list.append((dist,train_category))
    return new_list

def crop_and_tuple(img_list,category_list):
    result_list=[]
    for i in range(0,len(img_list)):
        img = img_list[i]
        tmp = cv2.resize(img, dsize=(16, 16))

        tmp=[item for sublist in tmp for item in sublist]
        
        #tmp = (tmp - np.mean(tmp)) / np.std(tmp)    #normalize
        #if(np.std(tmp,axis=0))==0:
        #    print(tmp)
        result_list.append((tmp,category_list[i]))
    return result_list

def crop_tuple_version(input):
    result_list=[]
    for i in input:
        img,cat = i
        
        """
        (h,w) = img.shape
        if(h>w):
            img= img[int(h/2)-int(w/2):int(h/2)+int(w/2),:]
        elif(h<w):
            img= img[:,int(w/2)-int(h/2):int(w/2)+int(h/2)]
        else:   
            img=img
        """

        tmp = cv2.resize(img, dsize=(16, 16))

        
        tmp=[item for sublist in tmp for item in sublist]
        print(np.shape(tmp))
        tmp = [float(t)/sum(tmp) for t in tmp]#normalize
        print(np.shape(tmp))
        #tmp = (tmp - np.mean(tmp)) / np.std(tmp)    #normalize
        #if(np.std(tmp,axis=0))==0:
        #    print(tmp)
        result_list.append((tmp,cat))
    return result_list

def find_most_occurence(List):
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 


def vote_function(input):
    cat_list=[]
    for i in input:
        trash,cat=i
        cat_list.append(cat)
    return find_most_occurence(cat_list)

def plotData(plt, data):
  x = [p[0] for p in data]
  y = [p[1] for p in data]
  plt.plot(x, y, '-o')
    

train_path = "../hw5_data/train/"
train_img_list = []
train_img_cat_list=[]
touple_list=[]
img_dirs = [d for d in os.listdir(train_path) if not d.startswith('.')]
img_names = [os.listdir(train_path + d) for d in img_dirs if not d.startswith('.')] #all image names, with each directory under a dimension, so 15 dimension in TA's dataset

for d_idx in range(len(img_names)): 
    for name in img_names[d_idx]:
        if not name.startswith('.'):
            img = cv2.imread(train_path + img_dirs[d_idx] + '/' + name, cv2.IMREAD_GRAYSCALE)
            touple_list.append((img,d_idx))
            train_img_list.append(img)
            train_img_cat_list.append(d_idx)
            

train_list=crop_tuple_version(touple_list)

test_path = "../hw5_data/test/"
test_img_list = []
test_img_cat_list=[]
test_list=[]
touple_test_list=[]
img_dirs = [d for d in os.listdir(test_path) if not d.startswith('.')]
img_names = [os.listdir(test_path + d) for d in img_dirs if not d.startswith('.')] #all image names, with each directory under a dimension, so 15 dimension in TA's dataset
print("=======")
for d_idx in range(len(img_names)): 
    for name in img_names[d_idx]:
        if not name.startswith('.'):
            img = cv2.imread(test_path + img_dirs[d_idx] + '/' + name,cv2.IMREAD_GRAYSCALE)
            touple_test_list.append((img,d_idx))
            test_img_list.append(img)
            test_img_cat_list.append(d_idx)
test_list = crop_tuple_version(touple_test_list)

print("======")
count=0 #counter of image
right=0 #counter of right prediction
print("=== testing ===")
result_list=[]
for k in range(1,31):  #testing on different k
    count=0 #counter of image
    right=0 #counter of right prediction
    mat = np.zeros((15,15))

    
    for i in test_list:
        count=count+1
        res = find_nn(train_list,i,k)   #return (dist,category)
        predict=vote_function(res)
        trash,cat = i
        if cat==predict:
            right = right+1
        mat[cat][predict]=mat[cat][predict]+1
    
   
    data = (k,right/(float(count)))
    result_list.append(data)
    print(k,count,right,right/float(count))

plotData(plt,result_list)
plt.ylim(0,0.5)
plt.show()
