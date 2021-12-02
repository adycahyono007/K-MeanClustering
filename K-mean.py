# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:46:18 2020

@author: QXRZDRAGON
"""

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
def euclidean_distance(x1,x2,y1,y2):
    dist = sqrt((x1-y1)**2+(x2-y2)**2)
    return dist

dataset = array([
        [1.0,3.0,4.0,5.0,1.0,4.0,1.0,2.0],
        [3.0,3.0,3.0,3.0,2.0,2.0,1.0,1.0],
     ])

cluster = array([
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0]
     ])

clustertemp = array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]
     ])


dist = array([
       [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    ])

centroid = ([[1.0,3.0,4.0],
             [3.0,3.0,3.0],
           ])  
    
k = 3 
iterasi = 0

simpan = 0.0

sama = np.array_equal(cluster, clustertemp)

print('Data yang dipakai: \n')
print('dataset: \n', dataset)
print('Centroid awal: \n', centroid[:][0],'\n',centroid[:][1])

while(sama):
    print("\n\niterasi: ", iterasi+1)
    print('variable cluster sebelum clustering: \n',cluster)  
    print('Centroid: \n', centroid)      
    z1 = np.array([[],[]])
    z2 = np.array([[],[]])
        
    for i in range(dataset.shape[1]):
        dist[0][i] = euclidean_distance(dataset[0][i], dataset[1][i], centroid[0][0], centroid[1][0], centroid[2][0])
        dist[1][i] = euclidean_distance(dataset[0][i], dataset[1][i], centroid[0][1], centroid[1][1], centroid[2][1])
        if dist[0][i] < dist[1][i]:
            clustertemp[0][i] = 1
            clustertemp[1][i] = 0
        else:
            clustertemp[0][i] = 0
            clustertemp[1][i] = 1                           
    
    print('dist: \n', dist)            
    print('vairable cluster setelah clustering: \n',clustertemp)
    if np.array_equal(cluster, clustertemp):
        sama = False
    else:
        cluster = clustertemp.copy()
        iterasi +=1
        #Perbaikan metode
        index = 0
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                if (cluster[0][j] == 1):
                    z1 = np.concatenate((z1,dataset[:, index:index+1]),axis=1)
                else:
                    z2 = np.concatenate((z2,dataset[:, index:index+1]),axis=1)
                index += 1

        centroid[0][0] = mean(z1[0, :z1.shape[1]])
        centroid[1][0] = mean(z1[1, :z1.shape[1]])
        centroid[0][1] = mean(z2[0, :z2.shape[1]])
        centroid[1][1] = mean(z2[1, :z2.shape[1]])
    
        
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if cluster[0][j] == 1:        
                plt.scatter(dataset[0,j], dataset[1, j], c='red', s=50, alpha=0.5)
            else:
                plt.scatter(dataset[0,j], dataset[1, j], c='blue', s=50, alpha=0.5)
                
plt.scatter(centroid[0][0], centroid[1][0], c='green', s=50, alpha=0.5)
plt.scatter(centroid[0][1], centroid[1][1], c='purple', s=50, alpha=0.5)