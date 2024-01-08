# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:12:12 2023

@author: JiWei Yu
"""

import pandas as pd
import numpy as np

#%% 
df=pd.read_csv('Co_superalloy_0_-100.csv')
data1=np.array(df)
df=pd.read_csv('Co_superalloy_-100_-200(1).csv')
data2=np.array(df)
data = np.concatenate((data1, data2), axis=0)

x=range(0,len(data))
r_max=data.max(axis=0)
r_min=data.min(axis=0)
# import matplotlib.pyplot as plt
# y=data[:,-1]
# y=np.sort(y)
# plt.scatter(x,y)
# plt.show()

# add some constrains
# mark_z=(-80<data[:,2])&(data[:,2]<-60)
# mark_x=(-0<data[:,0])&(data[:,0]<20)
# mark_y=(-5<data[:,1])&(data[:,1]<5)

# mark_z=(-60<data[:,2])&(data[:,2]<-40)
# mark_x=(-5<data[:,0])&(data[:,0]<5)
# mark_y=(-5<data[:,1])&(data[:,1]<5)

mark_z=(-120<data[:,2])&(data[:,2]<-90)
mark_x=(-8<data[:,0])&(data[:,0]<8)
mark_y=(-3<data[:,1])&(data[:,1]<3)
data=data[mark_z & mark_x & mark_y]
# data=data[mark_y]
#%% read rrng
pcd=data[:,0:3]
mask_Co = ((29.425<data[:,-1])&(data[:,-1]<29.508))|((58.848<data[:,-1])&(data[:,-1]<59.030))
mask_Ni = ((28.897<data[:,-1])&(data[:,-1]<29.033))|((29.902<data[:,-1])&(data[:,-1]<30.024))|(
(57.761<data[:,-1])&(data[:,-1]<58.117))|((59.832<data[:,-1])&(data[:,-1]<60.014))|(
(63.865<data[:,-1])&(data[:,-1]<63.998))|((30.436<data[:,-1])&(data[:,-1]<30.501))|(
(30.909<data[:,-1])&(data[:,-1]<31.020))|((31.920<data[:,-1])&(data[:,-1]<32.016))|(
(60.883<data[:,-1])&(data[:,-1]<61.075))|((61.877<data[:,-1])&(data[:,-1]<62.097))
mask_Al = ((13.458<data[:,-1])&(data[:,-1]<13.526))|((26.920<data[:,-1])&(data[:,-1]<27.040))|(
(8.976<data[:,-1])&(data[:,-1]<9.013))
mask_W = ((60.570<data[:,-1])&(data[:,-1]<60.721))|((61.247<data[:,-1])&(data[:,-1]<61.930))|(
(90.865<data[:,-1])&(data[:,-1]<91.084))|((91.367<data[:,-1])&(data[:,-1]<91.577))|(
(91.874<data[:,-1])&(data[:,-1]<92.674))|((92.855<data[:,-1])&(data[:,-1]<93.097))|(
(185.772<data[:,-1])&(data[:,-1]<186.139))|((183.762<data[:,-1])&(data[:,-1]<184.129))|(
(182.783<data[:,-1])&(data[:,-1]<183.109))|((181.747<data[:,-1])&(data[:,-1]<182.156))
mask_others = ((60.215<data[:,-1])&(data[:,-1]<60.415))|(
(90.363<data[:,-1])&(data[:,-1]<90.582))|((23.928<data[:,-1])&(data[:,-1]<24.021))|(
(23.443<data[:,-1])&(data[:,-1]<23.511))|((22.946<data[:,-1])&(data[:,-1]<23.007))|(
(24.433<data[:,-1])&(data[:,-1]<24.514))|((25.920<data[:,-1])&(data[:,-1]<26.021))|(
(26.423<data[:,-1])&(data[:,-1]<26.518))|((24.938<data[:,-1])&(data[:,-1]<25.005))|(
(51.848<data[:,-1])&(data[:,-1]<52.039))|((52.864<data[:,-1])&(data[:,-1]<53.026))|(
(53.873<data[:,-1])&(data[:,-1]<53.993))|((49.847<data[:,-1])&(data[:,-1]<50.045))

pcd_Co=pcd[mask_Co]
pcd_Ni=pcd[mask_Ni]
pcd_Al=pcd[mask_Al]
pcd_W=pcd[mask_W]
pcd_others = pcd[mask_others]
atom_num = len(pcd_Co) + len(pcd_Ni) + len(pcd_Al) + len(pcd_W) + len(pcd_others)
remian = atom_num / len(pcd)

r_Al=len(pcd_Al)/atom_num
r_Co=len(pcd_Co)/atom_num
r_Ni=len(pcd_Ni)/atom_num
r_W=len(pcd_W)/atom_num
r_others=len(pcd_others)/atom_num
# r_Al=1-r_Li-r_Mg
#%%
import open3d as o3d
points1=np.concatenate((pcd_Co, pcd_Ni), axis=0)
point_cloud1 = o3d.geometry.PointCloud()
point_cloud1.points =o3d.utility.Vector3dVector(points1)    
point_cloud1.paint_uniform_color([0,0,1])
# point_cloud1.paint_uniform_color([1,0.8314,0])
points2=np.concatenate((pcd_W, pcd_others), axis=0)
point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points =o3d.utility.Vector3dVector(points2)    
point_cloud2.paint_uniform_color([0,1,0])
# point_cloud2.paint_uniform_color([1,0,1])
points3=pcd_Al
point_cloud3 = o3d.geometry.PointCloud()
point_cloud3.points =o3d.utility.Vector3dVector(points3)    
# point_cloud3.paint_uniform_color([1,0,0])
# point_cloud3.paint_uniform_color([0,0.39,0])
# img=o3d.pybind.visualization.draw_geometries([point_cloud1,point_cloud2,point_cloud3])
img=o3d.pybind.visualization.draw_geometries([point_cloud2, point_cloud1])
#%%
# import matplotlib.pyplot as plt
# plt.scatter(pcd_Al[:,0],pcd_Al[:,1])
#%%
import sys
sys.exit()
# %% show by plt
import matplotlib.pyplot as plt

dt=np.concatenate((points1, points2, points3),axis=0)
# dt=points3
# dt=np.concatenate((atom_Li,atom_Li),axis=0)
x, y = dt[:, 0], dt[:, 2]
# plt.figure(figsize=(4,6))
plt.hist2d(x, y,
            bins = 100, 
           # norm = colors.LogNorm(), 
           # cmap ="gray"
           )
#%%
# make lattice parameter equals 1
pcd_Co/=0.3587
pcd_Ni/=0.3587
pcd_Al/=0.3587
pcd_W/=0.3587
pcd_others/=0.3587

data_others=np.concatenate((pcd_others,np.zeros((len(pcd_others),1))),axis=-1)
data_Co=np.concatenate((pcd_Co,np.ones((len(pcd_Co),1))),axis=-1)
data_Ni=np.concatenate((pcd_Ni,np.ones((len(pcd_Ni),1))*2),axis=-1)
data_Al=np.concatenate((pcd_Al,np.ones((len(pcd_Al),1))*3),axis=-1)
data_W=np.concatenate((pcd_W,np.ones((len(pcd_W),1))*4),axis=-1)
data=np.concatenate((data_others,data_Co,data_Ni,data_Al,data_W),axis=0)

# add fake label (0 for all atoms)
data=np.concatenate((data,np.zeros((len(data),1))),axis=1)
np.save('noise/SF5',data)
