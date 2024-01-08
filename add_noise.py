# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:00:19 2023

@author: JiWei Yu
"""

import numpy as np
import random

# add detection efficiency and trajectory aberration
def noise(atom,noise=0,noise_z=0,missing=0,t=1):
    '''
    trajectory aberration:
        noise = noise_xy, bigger than noise z
        value of noise = std of normal distribution
    detection efficiency:
        missing is an uniform distribution to drop atoms
        t is the limit of n-std
    '''
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

# step 4: noise
# after this you can get simulated data!
for i in range(0,100):
    data=np.load(f'rotate/{i}.npy')
    data=noise(data,random.uniform(0.5,2),random.uniform(0.2,0.5),random.uniform(0.2,0.6),1)
    np.save(f'noise/{i}',data)

