# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
warnings.filterwarnings('ignore')
dataset=pd.read_csv('Dataset.csv',header=None,names=['x','y'])
a = np.array(dataset)

# %%
def K(a,b,d):
    return(1+np.dot(a,b.transpose()))**d  

# %%
def K_exp(a,b,sigma):
    res=np.dot(-(a-b),(a-b).transpose())
    res=res/(2*(sigma**2))
    res=math.exp(res)
    return res

# %%
kernal_matrix=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix[i][j]= K(a[i],a[j],2)

# %%
def kernal_matrix_centering(km):
    mat_one=np.ones(shape=(1000,1000))
    I=np.identity(1000)
    mat_one=mat_one/1000
    diff=I-mat_one
    temp=np.dot(diff,km)
    kernal_matrix=np.dot(temp,diff)
    return kernal_matrix

# %%
def comp(x):
    return x[0]
kernal_matrix=kernal_matrix_centering(kernal_matrix)
def eigvalvec(kernal_matrix): 
    p=[]
    Beta_val,Beta_vec=np.linalg.eig(kernal_matrix)
    Beta_vec = Beta_vec.transpose()
    for i in range(0,1000):
        temp=[Beta_val[i],Beta_vec[i]]
        p.append(temp)
    p.sort(reverse=True,key=comp)
    return p

p=eigvalvec(kernal_matrix)
p[0][0]

# %%
first_component=p[0][1]
second_component=p[1][1]
alpha_1=first_component/math.sqrt(p[0][0])
alpha_2=second_component/math.sqrt(p[1][0])

# %%
x=np.matmul(kernal_matrix,alpha_1)
# print(alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Polynomial Kernal Degree 2")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix[i][j]= K(a[i],a[j],3)
kernal_matrix=kernal_matrix_centering(kernal_matrix)
p=eigvalvec(kernal_matrix)
first_component=p[0][1]
second_component=p[1][1]
alpha_1=first_component/math.sqrt(p[0][0])
alpha_2=second_component/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix,alpha_1)
# print(alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Polynomial Kernal Degree 3")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.1)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.1")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.2)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.2")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.3)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.3")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.4)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.4")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.5)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.5")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.6)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.6")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.7)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.7")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.8)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.8")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],0.9)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=0.9")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()

# %%
kernal_matrix_exp=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        kernal_matrix_exp[i][j]= K_exp(a[i],a[j],1)
kernal_matrix_exp=kernal_matrix_centering(kernal_matrix_exp)
p=eigvalvec(kernal_matrix_exp)
alpha_1=p[0][1]/math.sqrt(p[0][0])
alpha_2=p[1][1]/math.sqrt(p[1][0])
x=np.matmul(kernal_matrix_exp,alpha_1)
x = x.reshape(1000,1)
y=np.matmul(kernal_matrix_exp,alpha_2)
y = y.reshape(1000,1)
plt.plot(x,y,'o')
plt.title("Exponential Kernal sigma=1")
plt.xlabel("xvalues")
plt.ylabel("yvalues")
plt.show()


