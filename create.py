#-*-coding:GBK -*-
import os
import cv2
import numpy as np
import torch
import random
from transforms import *
from PIL import Image

data0 = []#存储背景
data1 = []#存储人脸
label = []
index1 = 0
index = 0
data0_test = []
data1_test = []
label_test = []
a = 686
b = 145
c = 50

def create_pair_train(dataset):
    global data0
    global data1
    global label
    
    dataset1 = sorted(os.listdir(dataset))     #
    subset0 = os.path.join(dataset,dataset1[0])#df的路径
    subset1 = os.path.join(dataset,dataset1[1])#real的路径
    subset2_d = os.listdir(subset0)#df随机顺序
    subset2_r = os.listdir(subset1)#real随机顺序
    print(subset2_d[0:50],subset2_r[0:50])
    
    for i in range(a):#添加df
        

        subset3 = os.path.join(subset0,subset2_d[i])#df0的路径
        subset4 = sorted(os.listdir(subset3))
        subset5 = os.path.join(subset3,subset4[0])#back的路径
        subset6 = os.path.join(subset3,subset4[1])#face的路径
        
        subset7 = sorted(os.listdir(subset5))#back的图像
        subset8 = sorted(os.listdir(subset6))#face的图像
        
        k0 = len(subset7)-1
        k1 = len(subset8)-1
        if k0<1:
            print(subset5)
        
        
        for j in range(c):
            image0 = os.path.join(subset5,subset7[random.randint(0,k0)])
            image1 = os.path.join(subset6,subset8[random.randint(0,k1)])
            data0.append(image0)#存储背景图
            data1.append(image1)#存储人脸图
            if subset2_d[i][0] == 'r':
                    label.append(1)
            else:
                    label.append(0)
            
    for i in range(a):#添加real
        
        
        subset3 = os.path.join(subset1,subset2_r[i])#real0的路径
        subset4 = sorted(os.listdir(subset3))
        subset5 = os.path.join(subset3,subset4[0])#back的路径
        subset6 = os.path.join(subset3,subset4[1])#face的路径
        
        subset7 = sorted(os.listdir(subset5))#back的图像
        subset8 = sorted(os.listdir(subset6))#face的图像
        
        k0 = len(subset7)-1
        k1 = len(subset8)-1
        
        
        
        for j in range(c):
            image0 = os.path.join(subset5,subset7[random.randint(0,k0)])
            image1 = os.path.join(subset6,subset8[random.randint(0,k1)])
            data0.append(image0)
            data1.append(image1)
            if subset2_r[i][0] == 'r':
                    label.append(1)
            else:
                    label.append(0)       
        
                
    if (len(data0)==len(label)) & (len(data0)==len(data1)):
        print('数据标签相等')                 
    zipped = zip(data0,data1,label)
    zipped = list(zipped)
    random.shuffle(zipped)
    data0,data1,label = zip(*zipped)
                    
    return len(data0)


def read_pair_train(batch_size):

    global index
    global data0
    global data1
    global label
    
    Max_couter = len(data0)
    Max_index = Max_couter // batch_size
    index = index % Max_index
    
    data_back = []
    data_face = []
    label_pair = []
    window = [x for x in range(index * batch_size, (index + 1) * batch_size)]
    
    for q in window:
        image0 = cv2.imread(data0[q]) 
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        image0 = Image.fromarray(image0)
        image0 = data_transforms(image0)#-128.0   
         
        image1 = cv2.imread(data1[q])
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = Image.fromarray(image1)
        image1 = data_transforms(image1)#-128.0
        
        data_face.append(np.array(image1))
        data_back.append(np.array(image0))
            
    
    data_back = torch.from_numpy(np.array(data_back))
    data_face = torch.from_numpy(np.array(data_face))
    
    data_back = data_back.type(torch.FloatTensor)
    data_face = data_face.type(torch.FloatTensor)
    
    label_pair = [label[q] for q in window]
    label_pair = torch.from_numpy(np.array(label_pair))
    label_pair = label_pair.float()
    
    index = (index + 1) % Max_index
    
    if index == 0:
        zipped = zip(data0,data1,label)
        zipped = list(zipped)
        random.shuffle(zipped)
        data0,data1,label = zip(*zipped)
    
    return data_back,data_face,label_pair


    
def create_pair_test(dataset):

    global data0_test
    global data1_test
    global label_test
    
    dataset1 = sorted(os.listdir(dataset))
    subset0 = os.path.join(dataset,dataset1[0])#df的路径
    subset1 = os.path.join(dataset,dataset1[1])#real的路径
    subset2_d = os.listdir(subset0)
    subset2_r = os.listdir(subset1)
    #print(subset0,subset1)
    
    for i in range(b):
        subset3 = os.path.join(subset0,subset2_d[i+a])#df0的路径
        subset4 = sorted(os.listdir(subset3))
        subset5 = os.path.join(subset3,subset4[0])#back的路径
        subset6 = os.path.join(subset3,subset4[1])#face的路径
        
        subset7 = sorted(os.listdir(subset5))#back的图像
        subset8 = sorted(os.listdir(subset6))#face的图像
        
        k0 = len(subset7)-1
        k1 = len(subset8)
        
        
        
        for j in range(c):
            image1 = os.path.join(subset6,subset8[j])
            for k in range(3):
                
                data1_test.append(image1)
            image0_1 = os.path.join(subset5,subset7[j])
            image0_2 = os.path.join(subset5,subset7[j+k1])
            image0_3 = os.path.join(subset5,subset7[j+2*k1])
            data0_test.append(image0_1)
            data0_test.append(image0_2)
            data0_test.append(image0_3)
            
            
            if subset2_d[i+a][0] == 'r':
                for k in range(3):
                    label_test.append(1)
            else:
                for k in range(3):
                    label_test.append(0)
            
    
    for i in range(b):        
        subset3 = os.path.join(subset1,subset2_r[i+a])#real0的路径
        subset4 = sorted(os.listdir(subset3))
        subset5 = os.path.join(subset3,subset4[0])#back的路径
        subset6 = os.path.join(subset3,subset4[1])#face的路径
        
        subset7 = sorted(os.listdir(subset5))#back的图像
        subset8 = sorted(os.listdir(subset6))#face的图像
        
        k0 = len(subset7)-1
        k1 = len(subset8)
        
        
        
        for j in range(c):
            image1 = os.path.join(subset6,subset8[j])
            for k in range(3):
                data1_test.append(image1)
                
            image0_1 = os.path.join(subset5,subset7[j])
            image0_2 = os.path.join(subset5,subset7[j+k1])
            image0_3 = os.path.join(subset5,subset7[j+2*k1])
            data0_test.append(image0_1)
            data0_test.append(image0_2)
            data0_test.append(image0_3)
            
            
            if subset2_r[i+a][0] == 'r':
                for k in range(3):
                    label_test.append(1)
            else:
                for k in range(3):
                    label_test.append(0)


    if (len(data0_test)==len(label_test)) & (len(data0_test)==len(data1_test)):
        print('数据标签相等')
    
    zipped = zip(data0_test,data1_test,label_test)
    zipped = list(zipped)
    random.shuffle(zipped)
    data0_test,data1_test,label_test = zip(*zipped)                 
    return len(data0_test)


def read_pair_test(batch_size):

    global index1
    global data0_test
    global data1_test
    global label_test
    
    Max_couter = len(data0_test)
    Max_index = Max_couter // batch_size
    index1 = index1 % Max_index
    
    data_back = []
    data_face = []
    
    window = [x for x in range(index1 * batch_size, (index1 + 1) * batch_size)]
    
    for q in window:
        image0 = cv2.imread(data0_test[q])
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        #b, image0, r = cv2.split(image0)
        image0 = Image.fromarray(image0)
        image0 = data_transformstest(image0)#-128.0
        data_back.append(np.array(image0))
        
        
        image1 = cv2.imread(data1_test[q])
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        #b, image1, r = cv2.split(image1)
        image1 = Image.fromarray(image1)
        image1 = data_transformstest(image1)#-128.0
        data_face.append(np.array(image1))
        
        
    data_back = torch.from_numpy(np.array(data_back))
    data_face = torch.from_numpy(np.array(data_face))
    
    data_back = data_back.type(torch.FloatTensor)
    data_face = data_face.type(torch.FloatTensor)
    
    label_pair = [label_test[q] for q in window]
    label_pair = torch.from_numpy(np.array(label_pair))
    label_pair = label_pair.float()
    index1 = (index1 + 1) % Max_index
    
    return data_back,data_face,label_pair    
