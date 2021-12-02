# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:55:55 2020

@author: QXRZDRAGON
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
def euclidean_distance(x1,x2,y1,y2):
    dist = sqrt((x1-y1)**2+(x2-y2)**2)
    return dist

dataset = array([
        [5.0,5.0,6.0,7.0,7.0,9.0],
        [5.0,6.0,6.0,7.0,8.0,7.0],
     ])

cluster = array([
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]
     ])

clustertemp = array([
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]
     ])

x = array([
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]
     ])
    
    
dist = array([
        [0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0]
    ])

c1 = ([[7.0,9.0],
       [8.0,7.0]])   

k = 2 
index = 0

simpan = 0.0

sama = np.array_equal(cluster, clustertemp)

while(sama):
    print("iterasi: ", index+1)
    print('cluster sebelum: \n',cluster)        
        
    for i in range(dataset.shape[1]):
        dist[0][i] = euclidean_distance(dataset[0][i], dataset[1][i], c1[0][0], c1[1][0])
        dist[1][i] = euclidean_distance(dataset[0][i], dataset[1][i], c1[0][1], c1[1][1])
        if dist[0][i] < dist[1][i]:
            clustertemp[0][i] = 1
            clustertemp[1][i] = 0
        else:
            clustertemp[0][i] = 0
            clustertemp[1][i] = 1                           
                
    #print('cluster sesudah clustering: \n',cluster)
    print('clustertemp sesudah clustering: \n',clustertemp)

    if np.array_equal(cluster, clustertemp):
        sama = False
    else:
        cluster = clustertemp.copy()
        index +=1
        print(index)
        x = dataset[0:dataset.shape[0], 0:index]
        y = dataset[0:dataset.shape[0], index:dataset.shape[1]]
        c1[0][0] = mean(x[0, :x.shape[1]])
        c1[1][0] = mean(x[1, :x.shape[1]])
        c1[0][1] = mean(y[0, :y.shape[1]])
        c1[1][1] = mean(y[1, :y.shape[1]])

print('Data :\n', dataset)
print('Anggota Cluster 1 :\n', x)
print('Anggota Cluster 2 :\n', y)
print('centroid\n', c1)
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[1]):
        if cluster[0][j] == 1:        
            plt.scatter(dataset[0,j], dataset[1, j], c='red', s=50, alpha=0.5)
        else:
            plt.scatter(dataset[0,j], dataset[1, j], c='blue', s=50, alpha=0.5)
