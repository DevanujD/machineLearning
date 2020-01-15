#17EC30013     #Devanuj Deka     #Assignment 4 (K-means Clustering)

import pandas as pd
import numpy as np
import pprint
import random
import math

delta  = np.finfo(float).eps #it is an infinitesimal number to avoid divide by zero error or log(0) error

#Import dataset
dataset = pd.read_csv("data4_19.csv",names=['S_Len', 'S_Wid', 'P_Len', 'P_Wid', 'Iris'])
m = []
m = (random.sample(range(0, 150), 3))
m.sort()
ind1,ind2,ind3 = m[0],m[1],m[2]

mean = pd.DataFrame(columns = dataset.columns)
mean = mean.append(dataset.iloc[ind1,: ])
mean = mean.append(dataset.iloc[ind2, :])
mean = mean.append(dataset.iloc[ind3, :])

mean.loc[:,'Iris'] = ['cluster1','cluster2','cluster3']

iteration = 10

for i in range(iteration):
    data_copy = pd.read_csv("data4_19.csv",names=['S_Len', 'S_Wid', 'P_Len', 'P_Wid', 'Iris'])
    for j in range(len(data_copy)):
        min = 100
        a=0
        for k in range(len(mean)):
            sum = (data_copy.iloc[j,0] - mean.iloc[k,0])**2 + (data_copy.iloc[j,1] - mean.iloc[k,1])**2 + (data_copy.iloc[j,2] - mean.iloc[k,2])**2 + (data_copy.iloc[j,3] - mean.iloc[k,3])**2
            sum = math.sqrt(sum)
            if sum<min:
                min = sum
                a = k
        data_copy.iloc[j, 4] = mean.iloc[a, 4]
    
    a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    for j in range(len(data_copy)):
        if data_copy.iloc[j, 4] == 'cluster1':
            a1 += data_copy.iloc[j, 0]
            b1 += data_copy.iloc[j, 1]
            c1 += data_copy.iloc[j, 2]
            d1 += data_copy.iloc[j, 3]
            cnt1 += 1
       
        if data_copy.iloc[j, 4] == 'cluster2':
            a2 += data_copy.iloc[j, 0]
            b2 += data_copy.iloc[j, 1]
            c2 += data_copy.iloc[j, 2]
            d2 += data_copy.iloc[j, 3]
            cnt2 += 1
        
        if data_copy.iloc[j, 4] == 'cluster3':
            a3 += data_copy.iloc[j, 0]
            b3 += data_copy.iloc[j, 1]
            c3 += data_copy.iloc[j, 2]
            d3 += data_copy.iloc[j, 3]
            cnt3 += 1
            
            
    mean.iloc[0,0] = a1/(cnt1 + delta)
    mean.iloc[0,1] = b1/(cnt1 + delta)
    mean.iloc[0,2] = c1/(cnt1 + delta)
    mean.iloc[0,3] = d1/(cnt1 + delta)
    
    mean.iloc[1,0] = a2/(cnt2 + delta)
    mean.iloc[1,1] = b2/(cnt2 + delta)
    mean.iloc[1,2] = c2/(cnt2 + delta)
    mean.iloc[1,3] = d2/(cnt2 + delta)
    
    mean.iloc[2,0] = a3/(cnt3 + delta)
    mean.iloc[2,1] = b3/(cnt3 + delta)
    mean.iloc[2,2] = c3/(cnt3 + delta)
    mean.iloc[2,3] = d3/(cnt3 + delta)


datacopy2 = pd.read_csv("data4_19.csv",names=['S_Len', 'S_Wid', 'P_Len', 'P_Wid', 'Iris'])  

for j in range(len(datacopy2)):
    min = 100
    a=0
    for k in range(len(mean)):
        sum = (datacopy2.iloc[j,0] - mean.iloc[k,0])**2 + (datacopy2.iloc[j,1] - mean.iloc[k,1])**2 + (datacopy2.iloc[j,2] - mean.iloc[k,2])**2 + (datacopy2.iloc[j,3] - mean.iloc[k,3])**2
        sum = math.sqrt(sum)
        if sum<min:
            min = sum
            a = k
    datacopy2.iloc[j, 4] = mean.iloc[a, 4]
        
cl1, cl2, cl3 = 0, 0, 0

for i in range(len(datacopy2)):
    if (datacopy2.iloc[i,-1] == 'cluster1'):
        cl1 += 1
    if (datacopy2.iloc[i,-1] == 'cluster2'):
        cl2 += 1
    if (datacopy2.iloc[i,-1] == 'cluster3'):
        cl3 += 1

        
il1, il2, il3 = 0, 0, 0

for i in range(len(dataset)):
    if (dataset.iloc[i,-1] == 'Iris-setosa'):
        il1 += 1
    if (dataset.iloc[i,-1] == 'Iris-versicolor'):
        il2 += 1
    if (dataset.iloc[i,-1] == 'Iris-virginica'):
        il3 += 1


t11, t12, t13, t21, t22, t23, t31, t32, t33 = 0, 0, 0, 0, 0, 0, 0, 0, 0     

for j in range(len(dataset)):
        if datacopy2.iloc[j,-1] == 'cluster1' and dataset.iloc[j, -1] == 'Iris-setosa':
            t11 += 1
        if datacopy2.iloc[j,-1] == 'cluster1' and dataset.iloc[j, -1] == 'Iris-versicolor':
            t12 += 1
        if datacopy2.iloc[j,-1] == 'cluster1' and dataset.iloc[j, -1] == 'Iris-virginica':
            t13 += 1
        if datacopy2.iloc[j,-1] == 'cluster2' and dataset.iloc[j, -1] == 'Iris-setosa':
            t21 += 1
        if datacopy2.iloc[j,-1] == 'cluster2' and dataset.iloc[j, -1] == 'Iris-versicolor':
            t22 += 1
        if datacopy2.iloc[j,-1] == 'cluster2' and dataset.iloc[j, -1] == 'Iris-virginica':
            t23 += 1
        if datacopy2.iloc[j,-1] == 'cluster3' and dataset.iloc[j, -1] == 'Iris-setosa':
            t31 += 1
        if datacopy2.iloc[j,-1] == 'cluster3' and dataset.iloc[j, -1] == 'Iris-versicolor':
            t32 += 1
        if datacopy2.iloc[j,-1] == 'cluster3' and dataset.iloc[j, -1] == 'Iris-virginica':
            t33 += 1


names = mean.columns
for i in range(len(mean)):
    print(mean.iloc[i,-1] + '--->')
    for j in range(len(mean.iloc[i,:])-1):
        print(str(names [j]) +' '+ str(mean.iloc[i,j]))
    print(' ')

            
#Jacquard Distance

jd11 = 1 - (t11/(il1 + cl1 - 1 + delta))
print("Jacquard Distance of cluster 1 and Iris-setosa:", jd11)
jd12 = 1 - (t12/(il2 + cl1 - 1 + delta))
print("Jacquard Distance of cluster 1 and Iris-versicolor:", jd12)
jd13 = 1 - (t13/(il3 + cl1 - 1 + delta))
print("Jacquard Distance of cluster 1 and Iris-virginica:", jd13)


jd21 = 1 - (t21/(il1 + cl2 - 1 + delta))
print("Jacquard Distance of cluster 2 and Iris-setosa:", jd21)
jd22 = 1 - (t22/(il2 + cl2 - 1 + delta))
print("Jacquard Distance of cluster 2 and Iris-versicolor:", jd22)
jd23 = 1 - (t23/(il3 + cl2 - 1 + delta))
print("Jacquard Distance of cluster 2 and Iris-virginica:", jd23)
    
    
jd31 = 1 - (t31/(il1 + cl3 - 1 + delta))
print("Jacquard Distance of cluster 3 and Iris-setosa:", jd31)
jd32 = 1 - (t32/(il2 + cl3 - 1 + delta))
print("Jacquard Distance of cluster 3 and Iris-setosa:", jd32)
jd33 = 1 - (t33/(il3 + cl3 - 1 + delta))
print("Jacquard Distance of cluster 3 and Iris-setosa:", jd33)