# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:14:20 2023

@author: JiWei Yu
"""


import numpy as np
import random

#%%
# some functions borrowed from Internet
def CW_rotate_X(angle, x, y, z):
    '''
    绕X轴正向旋转坐标计算
    INPUT --> 旋转角度, 原坐标
    '''
    angle = np.radians(angle) # 以弧度作为参数
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    new_x = x
    new_y = y*np.cos(angle) + z*np.sin(angle)
    new_z = -(y*np.sin(angle)) + z*np.cos(angle)
    return new_x, new_y, new_z

def CW_rotate_Y(angle, x, y, z):
    '''
    绕Y轴正向旋转坐标计算
    INPUT --> 旋转角度, 原坐标
    '''
    angle = np.radians(angle) # 以弧度作为参数
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    new_x = x*np.cos(angle) - z*np.sin(angle)
    new_y = y
    new_z = x*np.sin(angle) + z*np.cos(angle)
    return new_x, new_y, new_z

def CW_rotate_Z(angle, x, y, z):
    '''
    绕Z轴正向旋转坐标计算
    INPUT --> 旋转角度, 原坐标
    '''
    angle = np.radians(angle) # 以弧度作为参数
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    new_x = x*np.cos(angle) + y*np.sin(angle)
    new_y = -(x*np.sin(angle)) + y*np.cos(angle)
    new_z = z
    return new_x, new_y, new_z
#%% 
def Crystal_rotation(data,angel_x,angel_y,angel_z):
    x,y,z=data[:,0],data[:,1],data[:,2]
    x,y,z=CW_rotate_X(angel_x,x,y,z)
    x,y,z=CW_rotate_Y(angel_y,x,y,z)
    x,y,z=CW_rotate_Z(angel_z,x,y,z)
    data_rotate=np.vstack((x,y,z)).T
    return np.concatenate((data_rotate,data[:,3:]),axis=-1)
#%% 
import sys 
sys.exit()
#%% step 3: rotating
for i in range(0,100):
    ## something about experimental data reading
    # i='ex7'
    # data=np.load(f'noise/{i}.npy')
    # data_new=Crystal_rotation(data,angel_x=0,angel_y=90,angel_z=0)
    # i='ex8'
    # np.save(f'noise/{i}.npy',data_new)
    
    # step 3: rotating
    data=np.load(f'perfect/{i}.npy')
    data_new=Crystal_rotation(data,angel_x=random.randint(-3,3),angel_y=random.randint(-3,3),angel_z=random.randint(-90,90))
    np.save(f'rotate/{i}.npy',data_new)
    
