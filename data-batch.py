# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:02:02 2023

@author: JiWei Yu
"""

import numpy as np
import random

#%% genenrate perfect fcc 

# (x, y, z) part
# length = 4 when k0 = 5
k0 = 5
data0 = [[i, j, k] for i in range(-k0, k0+1) 
       for j in range(-k0, k0+1) for k in range(-k0, k0+1) ]  
data1 = [[i+0.5, j+0.5, k] for i in range(-k0, k0) 
       for j in range(-k0, k0) for k in range(-k0, k0+1)]
data2 = [[i+0.5, j, k+0.5] for i in range(-k0, k0) 
       for j in range(-k0, k0+1) for k in range(-k0, k0)]
data3 = [[i, j+0.5, k+0.5] for i in range(-k0, k0+1) 
       for j in range(-k0, k0) for k in range(-k0, k0)]
data_base = np.concatenate((data0, data1, data2, data3))
total_num = len(data_base)

# set atom ratio in matrix
r_Co = 0.40
r_Ni = 0.35
r_Al = 0.095
r_W = 0.043
r_others = 0.25 - r_Al - r_W

num_Al, num_W, num_others = int(total_num*r_Al),int(total_num*r_W),int(total_num*r_others)
num_Co, num_Ni =int(total_num*r_Co), int(total_num*r_Ni)
#%% functions

# add element type into perfect FCC
def fcc(data=data_base):
    # assign ratio
    np.random.shuffle(data)
    atom_others = data[0:num_others]
    num1 = num_others+num_Co
    atom_Co = data[num_others:num1]
    num2 = num1 + num_Ni
    atom_Ni = data[num1:num2]
    num3 = num2 + num_Al
    atom_Al = data[num2:num3]
    num4 = num3 + num_W
    atom_W = data[num3:num4]
    
    # add atom type
    data_others=np.concatenate((atom_others,np.zeros((num_others,1))),axis=-1)
    data_Co=np.concatenate((atom_Co,np.ones((num_Co,1))),axis=-1)
    data_Ni=np.concatenate((atom_Ni,np.ones((num_Ni,1))*2),axis=-1)
    data_Al=np.concatenate((atom_Al,np.ones((num_Al,1))*3),axis=-1)
    data_W=np.concatenate((atom_W,np.ones((num_W,1))*4),axis=-1)
    data=np.concatenate((data_others,data_Co,data_Ni,data_Al,data_W),axis=0)

    # add matrix label (0 for all atoms)
    data=np.concatenate((data,np.zeros((len(data),1))),axis=1)
    return data

# create domains
def L12(x,y,z,r,data):
    # location of domains
    x0,y0,z0=x-r,y-r,z-r
    l=int(r*2+1)
    
    # domains information
    data5=[[x0+i,y0+j,z0+k,0,1.0] for i in range(l) for j in range(l-1) for k in range(l-1) ]  
    data6=[[x0+i+0.5,y0+j+0.5,z0+k,0,1] for i in range(l-1) for j in range(l-1) for k in range(l)]
    data7=[[x0+i+0.5,y0+j,z0+k+0.5,0,1] for i in range(l-1) for j in range(l) for k in range(l-1)]
    data8=[[x0+i,y0+j+0.5,z0+k+0.5,0,1] for i in range(l) for j in range(l-1) for k in range(l-1)]  
    data5,data6,data7,data8=np.array(data5),np.array(data6),np.array(data7),np.array(data8)
    
    # add specific type of A3B (L12) 
    part_A=np.concatenate((data6,data7,data8))
    part_B=data5
    np.random.shuffle(part_A)
    np.random.shuffle(part_B)
    num_L12 = len(part_A) + len(part_B)
    num_Al,num_W,num_others=int(num_L12*r_Al),int(num_L12*r_W),int(num_L12*r_others)
    num_Co,num_Ni=int(num_L12*r_Co),int(num_L12*r_Ni)
    part_B[:num_Al, 3] = 3
    part_B[num_Al:num_Al+num_W, 3] = 4
    part_A[:num_Co, 3] = 1
    part_A[num_Co:num_Co+num_Ni, 3] = 2
    data_new=np.concatenate((part_A, part_B),axis=0)
    
    ## create fake l12
    # np.random.shuffle(data_new[:,-2])
    
    ## spherical nanoprecipitates
    mark1=((data_new[:,0]-x)**2+(data_new[:,1]-y)**2+(data_new[:,2]-z)**2)<=r**2
    ## matrix limits
    mark3=(-k0<data_new[:,0])&(data_new[:,0]<k0+1)&(-k0<data_new[:,1])&(data_new[:,1]<k0+1)&(-k0<data_new[:,2])&(data_new[:,2]<k0+1)
    data_l12=data_new[mark1&mark3]
    mark2=((data[:,0]-x)**2+(data[:,1]-y)**2+(data[:,2]-z)**2)>r**2
    data_fcc=data[mark2]
    data_f=np.concatenate((data_l12,data_fcc))
    return data_f

# insert domains into matrix
def fcc_to_l12(data):
    ## 
    x,y,z=random.randint(-k0+1,k0-1),random.randint(-k0+1,k0-1),random.randint(-k0+1,k0-1)
    # x,y,z=random.randint(-k0+2,k0-2),random.randint(-k0+2,k0-2),random.randint(-k0+2,k0-2)
    # r=random.randint(4,6)
    # r=random.randint(3,4)
    r = 4
    # r=random.uniform(2.5,5)
    data_f=L12(x,y,z,r,data)
    return data_f

# add detection efficiency and trajectory aberration
def noise(atom,noise=0,noise_z=0,missing=0,t=1):
    
    flux=np.random.normal(0,noise,((atom.shape[0],3)))
    flux_z=np.random.normal(0,noise_z,((atom.shape[0],3)))

    mark4=(flux[:,0]>t*noise)|(flux[:,0]<-t*noise)
    mark5=(flux[:,1]>t*noise)|(flux[:,1]<-t*noise)
    mark6=(flux[:,2]>t*noise_z)|(flux[:,2]<-t*noise_z)
    flux[mark4,0]=np.random.uniform(-t*noise,t*noise)
    flux[mark5,1]=np.random.uniform(-t*noise,t*noise)
    flux_z[mark6,2]=np.random.uniform(-t*noise_z,t*noise_z)
    atom[:,0]+=flux[:,0]
    atom[:,1]+=flux[:,1]
    atom[:,2]+=flux_z[:,2]
    k5=int(missing*len(atom))
    np.random.shuffle(atom)
    atom=atom[k5:]
    return atom
#%% generate batched data
for i in range(0,100):
    
    # create fcc matrix randomly (step 1)
    data=fcc()
    
    # repeat, to create several domains (step 2)
    num=random.randint(2,3)
    for k in range(num):
        data=fcc_to_l12(data)
        
    # save perfect data
    np.save(f'perfect/{i}',data)
    
    ## save noise data (step 4), step 3: rotation was in another file.
    # data=noise(data,random.uniform(0.5,2),random.uniform(0.2,0.5),random.uniform(0.2,0.6),1)
    # np.save(f'noise/{i}',data)


















