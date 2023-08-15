# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
dataset=pd.read_csv('Dataset.csv',header=None,names=['x','y'])
data_array=np.array(dataset)

# %%
z=np.zeros(shape=[1000,1])
error=[]
c=0
s=0
for i in range(0,1000):
    z[i]=random.randint(0,3)
mean_array=np.zeros(shape=[4,2])
def calc_clustermean(k):
    sum_z=[0,0]
    c=0
    for i in range(0,1000):
        if(z[i]==k):
            sum_z=sum_z+data_array[i]
            c=c+1
    z_mean=sum_z/c
    return z_mean
mean_array[0]=calc_clustermean(0)
mean_array[1]=calc_clustermean(1)
mean_array[2]=calc_clustermean(2)
mean_array[3]=calc_clustermean(3)

while True:
    change=False
    for i in range(0,1000):
        d0=np.linalg.norm(data_array[i]-mean_array[0])
        d1=np.linalg.norm(data_array[i]-mean_array[1])
        d2=np.linalg.norm(data_array[i]-mean_array[2])
        d3=np.linalg.norm(data_array[i]-mean_array[3])
        min_d=min(d0,d1,d2,d3)
        if(min_d==d0):
            cluster=0
        if(min_d==d1):
            cluster=1
        if(min_d==d2):
            cluster=2
        if(min_d==d3):
            cluster=3

        if(z[i]==0):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[0])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[0])
                change=True
        elif(z[i]==1):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[1])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[1])
                change=True
        elif(z[i]==2):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[2])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[2])
                change=True
        else:
            if(min_d<np.linalg.norm(data_array[i]-mean_array[3])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[3])
                change=True
    if(change==False):
        break
    error.append(s)
    s=0
import matplotlib.colors as mcolors
for i in range(0,1000):
    if(z[i]==0):
        plt.scatter(data_array[i][0],data_array[i][1],c='r')
    if(z[i]==1):
        plt.scatter(data_array[i][0],data_array[i][1],c='b')
    if(z[i]==2):
        plt.scatter(data_array[i][0],data_array[i][1],c='g')
    if(z[i]==3):
        plt.scatter(data_array[i][0],data_array[i][1],c='y')
plt.title("Clustered data")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
plt.title("Error Function w.r.t Iterations")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.plot(error)
plt.show()

# %%
z=np.zeros(shape=[1000,1])
for i in range(0,1000):
    z[i]=random.randint(0,3)
error1=[]
c=0
s=0
mean_array=np.zeros(shape=[4,2])
def calc_clustermean(k):
    sum_z=[0,0]
    c=0
    for i in range(0,1000):
        if(z[i]==k):
            sum_z=sum_z+data_array[i]
            c=c+1
    z_mean=sum_z/c
    return z_mean
mean_array[0]=calc_clustermean(0)
mean_array[1]=calc_clustermean(1)
mean_array[2]=calc_clustermean(2)
mean_array[3]=calc_clustermean(3)
while True:
    change=False
    for i in range(0,1000):
        d0=np.linalg.norm(data_array[i]-mean_array[0])
        d1=np.linalg.norm(data_array[i]-mean_array[1])
        d2=np.linalg.norm(data_array[i]-mean_array[2])
        d3=np.linalg.norm(data_array[i]-mean_array[3])
        min_d=min(d0,d1,d2,d3)
        if(min_d==d0):
            cluster=0
        if(min_d==d1):
            cluster=1
        if(min_d==d2):
            cluster=2
        if(min_d==d3):
            cluster=3

        if(z[i]==0):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[0])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[0])
                change=True
        elif(z[i]==1):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[1])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[1])
                change=True
        elif(z[i]==2):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[2])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[2])
                change=True
        else:
            if(min_d<np.linalg.norm(data_array[i]-mean_array[3])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[3])
                change=True
    if(change==False):
        break
    error1.append(s)
    s=0
import matplotlib.colors as mcolors
for i in range(0,1000):
    if(z[i]==0):
        plt.scatter(data_array[i][0],data_array[i][1],c='r')
    if(z[i]==1):
        plt.scatter(data_array[i][0],data_array[i][1],c='b')
    if(z[i]==2):
        plt.scatter(data_array[i][0],data_array[i][1],c='g')
    if(z[i]==3):
        plt.scatter(data_array[i][0],data_array[i][1],c='y')
plt.title("Clustered data")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
plt.title("Error Function w.r.t Iterations")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.plot(error1)
plt.show()

# %%
z=np.zeros(shape=[1000,1])
for i in range(0,1000):
    z[i]=random.randint(0,3)
error2=[]
c=0
s=0
mean_array=np.zeros(shape=[4,2])
def calc_clustermean(k):
    sum_z=[0,0]
    c=0
    for i in range(0,1000):
        if(z[i]==k):
            sum_z=sum_z+data_array[i]
            c=c+1
    z_mean=sum_z/c
    return z_mean
mean_array[0]=calc_clustermean(0)
mean_array[1]=calc_clustermean(1)
mean_array[2]=calc_clustermean(2)
mean_array[3]=calc_clustermean(3)
while True:
    change=False
    for i in range(0,1000):
        d0=np.linalg.norm(data_array[i]-mean_array[0])
        d1=np.linalg.norm(data_array[i]-mean_array[1])
        d2=np.linalg.norm(data_array[i]-mean_array[2])
        d3=np.linalg.norm(data_array[i]-mean_array[3])
        min_d=min(d0,d1,d2,d3)
        if(min_d==d0):
            cluster=0
        if(min_d==d1):
            cluster=1
        if(min_d==d2):
            cluster=2
        if(min_d==d3):
            cluster=3

        if(z[i]==0):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[0])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[0])
                change=True
        elif(z[i]==1):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[1])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[1])
                change=True
        elif(z[i]==2):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[2])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[2])
                change=True
        else:
            if(min_d<np.linalg.norm(data_array[i]-mean_array[3])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[3])
                change=True
    if(change==False):
        break
    error2.append(s)
    s=0
import matplotlib.colors as mcolors
for i in range(0,1000):
    if(z[i]==0):
        plt.scatter(data_array[i][0],data_array[i][1],c='r')
    if(z[i]==1):
        plt.scatter(data_array[i][0],data_array[i][1],c='b')
    if(z[i]==2):
        plt.scatter(data_array[i][0],data_array[i][1],c='g')
    if(z[i]==3):
        plt.scatter(data_array[i][0],data_array[i][1],c='y')
plt.title("Clustered data")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
plt.title("Error Function w.r.t Iterations")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.plot(error2)
plt.show()

# %%
import random
z=np.zeros(shape=[1000,1])
for i in range(0,1000):
    z[i]=random.randint(0,3)
error3=[]
c=0
s=0
mean_array=np.zeros(shape=[4,2])
def calc_clustermean(k):
    sum_z=[0,0]
    c=0
    for i in range(0,1000):
        if(z[i]==k):
            sum_z=sum_z+data_array[i]
            c=c+1
    z_mean=sum_z/c
    return z_mean
mean_array[0]=calc_clustermean(0)
mean_array[1]=calc_clustermean(1)
mean_array[2]=calc_clustermean(2)
mean_array[3]=calc_clustermean(3)
while True:
    change=False
    for i in range(0,1000):
        d0=np.linalg.norm(data_array[i]-mean_array[0])
        d1=np.linalg.norm(data_array[i]-mean_array[1])
        d2=np.linalg.norm(data_array[i]-mean_array[2])
        d3=np.linalg.norm(data_array[i]-mean_array[3])
        min_d=min(d0,d1,d2,d3)
        if(min_d==d0):
            cluster=0
        if(min_d==d1):
            cluster=1
        if(min_d==d2):
            cluster=2
        if(min_d==d3):
            cluster=3

        if(z[i]==0):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[0])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[0])
                change=True
        elif(z[i]==1):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[1])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[1])
                change=True
        elif(z[i]==2):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[2])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[2])
                change=True
        else:
            if(min_d<np.linalg.norm(data_array[i]-mean_array[3])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[3])
                change=True
    if(change==False):
        break
    error3.append(s)
    s=0
import matplotlib.colors as mcolors
for i in range(0,1000):
    if(z[i]==0):
        plt.scatter(data_array[i][0],data_array[i][1],c='r')
    if(z[i]==1):
        plt.scatter(data_array[i][0],data_array[i][1],c='b')
    if(z[i]==2):
        plt.scatter(data_array[i][0],data_array[i][1],c='g')
    if(z[i]==3):
        plt.scatter(data_array[i][0],data_array[i][1],c='y')
plt.title("Clustered data")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
plt.title("Error Function w.r.t Iterations")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.plot(error3)
plt.show()

# %%
import random
z=np.zeros(shape=[1000,1])
for i in range(0,1000):
    z[i]=random.randint(0,3)
error4=[]
c=0
s=0
mean_array=np.zeros(shape=[4,2])
def calc_clustermean(k):
    sum_z=[0,0]
    c=0
    for i in range(0,1000):
        if(z[i]==k):
            sum_z=sum_z+data_array[i]
            c=c+1
    z_mean=sum_z/c
    return z_mean
mean_array[0]=calc_clustermean(0)
mean_array[1]=calc_clustermean(1)
mean_array[2]=calc_clustermean(2)
mean_array[3]=calc_clustermean(3)
while True:
    change=False
    for i in range(0,1000):
        d0=np.linalg.norm(data_array[i]-mean_array[0])
        d1=np.linalg.norm(data_array[i]-mean_array[1])
        d2=np.linalg.norm(data_array[i]-mean_array[2])
        d3=np.linalg.norm(data_array[i]-mean_array[3])
        min_d=min(d0,d1,d2,d3)
        if(min_d==d0):
            cluster=0
        if(min_d==d1):
            cluster=1
        if(min_d==d2):
            cluster=2
        if(min_d==d3):
            cluster=3

        if(z[i]==0):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[0])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[0])
                change=True
        elif(z[i]==1):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[1])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[1])
                change=True
        elif(z[i]==2):
            if(min_d<np.linalg.norm(data_array[i]-mean_array[2])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[2])
                change=True
        else:
            if(min_d<np.linalg.norm(data_array[i]-mean_array[3])):
                z[i]=cluster
                mean_array[0]=calc_clustermean(0)
                mean_array[1]=calc_clustermean(1)
                mean_array[2]=calc_clustermean(2)
                mean_array[3]=calc_clustermean(3)
                s+=np.linalg.norm(data_array[i]-mean_array[3])
                change=True
    if(change==False):
        break
    error4.append(s)
    s=0
import matplotlib.colors as mcolors
for i in range(0,1000):
    if(z[i]==0):
        plt.scatter(data_array[i][0],data_array[i][1],c='r')
    if(z[i]==1):
        plt.scatter(data_array[i][0],data_array[i][1],c='b')
    if(z[i]==2):
        plt.scatter(data_array[i][0],data_array[i][1],c='g')
    if(z[i]==3):
        plt.scatter(data_array[i][0],data_array[i][1],c='y')
plt.title("Clustered data")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
plt.title("Error Function w.r.t Iterations")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.plot(error4)
plt.show()


