# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:00:08 2023

@author: JiWei Yu
"""

import numpy as np
import random

    
num_f=32  
def extract_v2(data):   
    data_train_1=data[:,:3]
    data_train_2=data[:,3:5].reshape((len(data_train_1),2))
    data_train_3=data[:,5:6].reshape((len(data_train_1),1))
    feature=[]
    label=[]
    # data_label=data[:,-1]
    for i in range(len(data)):
        x,y,z=data_train_1[i]
        mark=((data_train_1[:,0]-x)**2+(data_train_1[:,1]-y)**2+(data_train_1[:,2]-z)**2<20)
        if np.sum(mark)<num_f:
            mark=((data_train_1[:,0]-x)**2+(data_train_1[:,1]-y)**2+(data_train_1[:,2]-z)**2<1000)
        dt1,dt2,dt3=data_train_1[mark],data_train_2[mark],data_train_3[mark]
        a0=data_train_1[i]-dt1
        s0=np.sum(np.square(a0),axis=1).reshape((len(a0),1))
        # dt = np.dtype([('name',  np.float64),('age',  np.float64),('z',np.float64),('dis',np.float64)])
        b0=np.concatenate((s0,a0,dt3,dt2),axis=1)
        b0=list(b0)
        b0.sort(key=lambda u:(u[0]))
        c0=np.array(b0)
        d0=c0[:num_f,1:]
        label.append(d0[0,-1])
        zero=np.zeros((len(d0),2))
        zero+=d0[0,3:5]
        e0=np.concatenate((d0[:,:-1],zero),axis=1)
        feature.append(e0)
    return np.array(feature),np.array(label)

for i in range(0,100):
    # i='SF5'
    # i = 'SF' + str(i)
    data=np.load(f'noise/{i}.npy')
    prediction=np.load(f'prediction/{i}.npy')
    data=np.concatenate((data,prediction),axis=-1)
    # i='ex11_r'
    # np.save(f'noise2/{i}',data)
    feature,label=extract_v2(data)
    np.save(f'feature2/{i}.npy',feature)
    
