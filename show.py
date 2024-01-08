# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:28:06 2023

@author: JiWei Yu
"""
import numpy as np
import open3d as o3d
#%% pd read
# import pandas as pd
# df=pd.read_csv('pred_extract_0_190nm_above_62_win_4_stride_1_cross_validation.csv')
# # df=pd.read_csv('R18_15386-v24-clip_non_pole.csv')
# data=np.array(df)
#%%
# k='SF3'
k = 98
# data=np.load(f'perfect/{k}.npy')
# data=np.load(f'rotate/{k}.npy')
# data=np.load(f'small/{k}.npy')
data=np.load(f'noise/{k}.npy')
# data=np.load(f'simulated_l12/{k}.npy')
# mark_z=(10<data[:,2])&(data[:,2]<50)
# mark_x=(-5<data[:,0])&(data[:,0]<5)
# mark_y=(-5<data[:,1])&(data[:,1]<5)
# data=data[mark_z&mark_x&mark_y]
# mark_z=(40<data[:,2])&(data[:,2]<60)
# data=data[mark_z]
# np.random.shuffle(data)
# data=data[:int(len(data)/25)]
p_label=np.load(f'prediction2/{k}.npy')>0.5
# %% pole 
# data=np.load(f'noise/SF1.npy')
# p_label=np.load(f'prediction2/SF1.npy')>0.2
# for i in range(2,3):
#     # i = str(i) +'_r2(30-170)'
#     data_per=np.load(f'noise/SF{i}.npy')
#     label_per=np.load(f'prediction2/SF{i}.npy')>0.2
#     data=np.concatenate((data,data_per),axis=0)
#     p_label=np.concatenate((p_label,label_per),axis=0)
# mark_y=(-1<data[:,1])&(data[:,1]<1)
# data,p_label = data[mark_y], p_label[mark_y]
#%% random
# data=np.load(f'noise/Au1_2_r.npy')
# p_label=np.load(f'prediction3/Au1_2_r.npy')>0.2
# for i in range(2,4):
#     data_per=np.load(f'noise/Au{i}_2_r.npy')
#     label_per=np.load(f'prediction3/Au{i}_1_r.npy')>0.2
#     data=np.concatenate((data,data_per),axis=0)
#     p_label=np.concatenate((p_label,label_per),axis=0)
#%% 28
# data=np.load(f'noise/0_r1(30-170).npy')
# p_label=np.load(f'prediction3/0_r1(30-170).npy')>0.5
# for i in range(1,28):
#     # i = str(i) +'_r2(30-170)'
#     data_per=np.load(f'noise/{i}_r1(30-170).npy')
#     label_per=np.load(f'prediction3/{i}_r1(30-170).npy')>0.5
#     data=np.concatenate((data,data_per),axis=0)
#     p_label=np.concatenate((p_label,label_per),axis=0)
#%%
# mark_z=(50<data[:,2])&(data[:,2]<110)
# mark_x=(5<data[:,0])&(data[:,0]<20)

# data,p_label=data[(mark_x) & (mark_y) & (mark_z)],p_label[(mark_x) & (mark_y) & (mark_z)]
r_min=np.min(data,axis=0)
r_max=np.max(data,axis=0)
atom_Al,atom_Li,atom_Mg=data[((data[:,3]==1) | (data[:,3]==2))],data[((data[:,3]==3) | (data[:,3]==4))],data[data[:,3]==0]
r1=len(atom_Li)/len(data)
r2=len(atom_Mg)/len(data)
r3=1-r1-r2
Li_t=atom_Li[atom_Li[:,-1]==1]
Al_t=atom_Al[atom_Al[:,-1]==1]
Mg_t=atom_Mg[atom_Mg[:,-1]==1]
points6=Li_t[:,0:3]
point_cloud6 = o3d.geometry.PointCloud()
point_cloud6.points =o3d.utility.Vector3dVector(points6)    
point_cloud6.paint_uniform_color([1,0,1])
points4=Al_t[:,0:3]
point_cloud4 = o3d.geometry.PointCloud()
point_cloud4.points =o3d.utility.Vector3dVector(points4)    
point_cloud4.paint_uniform_color([1,0.8314,0])
points10=Mg_t[:,0:3]
point_cloud10 = o3d.geometry.PointCloud()
point_cloud10.points =o3d.utility.Vector3dVector(points10)    
point_cloud10.paint_uniform_color([0,0.39,0])
#%% show crystal structure with o3d

points1=atom_Al[:,0:3]
point_cloud1 = o3d.geometry.PointCloud()
point_cloud1.points =o3d.utility.Vector3dVector(points1)    
point_cloud1.paint_uniform_color([0,0,1])
points2=atom_Li[:,0:3]
point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points =o3d.utility.Vector3dVector(points2)    
point_cloud2.paint_uniform_color([0,1,0])
points3=atom_Mg[:,0:3]
point_cloud3 = o3d.geometry.PointCloud()
point_cloud3.points =o3d.utility.Vector3dVector(points3)    
point_cloud3.paint_uniform_color([1,0,0])


# img=o3d.pybind.visualization.draw_geometries([point_cloud1,point_cloud2,point_cloud3])
# img=o3d.pybind.visualization.draw_geometries([point_cloud2])
img=o3d.pybind.visualization.draw_geometries([point_cloud6,point_cloud2])
img=o3d.pybind.visualization.draw_geometries([point_cloud6,point_cloud4,point_cloud10])
# img=o3d.pybind.visualization.draw_geometries([point_cloud6,point_cloud4,point_cloud10,point_cloud1])
#%%
import sys
sys.exit()
# %% show by plt
import matplotlib.pyplot as plt
plt.hist(data[:,3],bins=5)
plt.hist(data[:,4],bins=2)
#%%
# k='ex4'
# p_label=p_label[mark_z&mark_x&mark_y]
# p_label=p_label[mark_z]
# p_label=np.load(f'prediction_2/np0_0.npy')
data=np.hstack((data[:,:-1],p_label.reshape(len(p_label),1)))
atom_Al,atom_Li,atom_Mg=data[((data[:,3]==1) | (data[:,3]==2))],data[((data[:,3]==3) | (data[:,3]==4))],data[data[:,3]==0]
Li_p=atom_Li[atom_Li[:,-1]==1]
Al_p=atom_Al[atom_Al[:,-1]==1]
Mg_p=atom_Mg[atom_Mg[:,-1]==1]
points5=Li_p[:,0:3]
point_cloud5 = o3d.geometry.PointCloud()
point_cloud5.points =o3d.utility.Vector3dVector(points5)    
point_cloud5.paint_uniform_color([1,0,1])
# img=o3d.pybind.visualization.draw_geometries([point_cloud5,point_cloud2])
points7=Al_p[:,0:3]
point_cloud7 = o3d.geometry.PointCloud()
point_cloud7.points =o3d.utility.Vector3dVector(points7)    
point_cloud7.paint_uniform_color([1,0.8314,0])
points8=Mg_p[:,0:3]
point_cloud8 = o3d.geometry.PointCloud()
point_cloud8.points =o3d.utility.Vector3dVector(points8)    
point_cloud8.paint_uniform_color([0,0.39,0])
img=o3d.pybind.visualization.draw_geometries([point_cloud5,point_cloud2])
img=o3d.pybind.visualization.draw_geometries([point_cloud7,point_cloud5,point_cloud8])
r_o=(len(Al_p)+len(Li_p))/(len(atom_Al)+len(atom_Li))
print(r_o)
#%%
import sys
sys.exit()
#%% save data to csv for AP
data_all = data[:,:-1]
data_order = data_all[data[:,-1]==1]
data_all[:,:-1]*=0.3587
data_order[:,:-1]*=0.3587
import pandas as pd
df_all = pd.DataFrame(data_all)
df_all.to_csv('data/all_1.csv',index=False,header=False)
df_order = pd.DataFrame(data_order)
df_order.to_csv('data/orderd_1.csv',index=False,header=False)
#%%
# dt1=data[:,:-1]
# mark=dt1[:,3]==0
# dt1[:,3][mark]=27
# dt1[:,3][dt1[:,3]==1]=7
# dt1[:,3][dt1[:,3]==2]=12
dt=np.concatenate((Al_p,Li_p),axis=0)
dt=dt[:,:-1]
dt[:,3][dt[:,3]==0]=64
dt[:,3][dt[:,3]==1]=195
dt[:,0]*=0.39
dt[:,1]*=0.39
dt[:,2]*=0.39
import pandas as pd
df=pd.DataFrame(dt)
df.to_csv('R1(140_0.5).csv',index=False,header=False)
# r_total=len(Li_p)+len(Al_p)+len(Mg_p)
# r_Li=len(Li_p)/r_total
# r_Mg=len(Mg_p)/r_total
# r_Al=1-r_Li-r_Mg
#%% analysis
import matplotlib.pyplot as plt
# Al_p, Mg_p, Li_p = atom_Al, atom_Mg, atom_Li
# pred_Co = Al_p[Al_p[:,3]==1]
# pred_Ni = Al_p[Al_p[:,3]==2]
# pred_Al = Li_p[Li_p[:,3]==3]
# pred_W = Li_p[Li_p[:,3]==4]
# pred_others = Mg_p[Mg_p[:,3]==0]
# element = 'all'
# dt = pred_W
dt=np.concatenate((Li_p, Al_p, Mg_p),axis=0)

# dt=Li_p
# dt=np.concatenate((atom_Al,atom_Li,atom_Mg),axis=0)
x, y = dt[:, 0], dt[:, 2]
x *= 0.3587
y *= 0.3587
plt.figure(figsize=(4,7.5))
plt.hist2d(x, y,
            bins = 100, 
           # norm = colors.LogNorm(), 
           # cmap ="gray"
           )
# plt.title(f'{element}')
# plt.savefig(f'plot/SF3_{element}.png')
#%%
# k='ex3'
dt=np.concatenate((atom_Al,atom_Li),axis=0)
np.random.shuffle(dt[:,3])
np.save(f'noise/{k}_r.npy',dt)
# np.save(f'simulated_l12/{k}_p.npy',dt)
# %% DBSCAN
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN

X=np.concatenate((Al_p,Li_p),axis=0)[:,0:3]

#AlLiMg
# db = DBSCAN(eps=5, min_samples=50).fit(X)
db = DBSCAN(eps=1, min_samples=5).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        # figsize=(2, 10),
        # facecolor="white",
        # tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=1, alpha=0.8)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1,1,170/12,170/12]))
    # ax.view_init(azim=-60, elev=9)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    # fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()
def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=10, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(10,5), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()
X *= 0.39
plot_3d(X, labels, 'DBSCAN')
# plot_2d(X[:,1:3], labels, 'DBSCAN')
# plot_2d(X[:,1:3], np.zeros(len(labels)), 'DBSCAN')



