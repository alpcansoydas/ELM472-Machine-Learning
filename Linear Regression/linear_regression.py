#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x):
    # Importing data
    data = np.load(x) # 100x2 
    x = data[:,0] # Assign column0 as x
    r = data[:,1] # Assign column1 as r

    plt.scatter(x,r) # Plotting input data as scatter
    plt.xlabel("x")
    plt.ylabel("r")

    a,b,c,d,e,f = 0,0,0,0,0,0
    N = len(x) # len(x) = len(r) = N

    # Calculating w1 from linear regression formula
    for i in range(0,N):
        a = a + x[i]*r[i]
        b = b + N*(1/N*x[i])*(1/N*r[i])
        c = c + x[i]**2
        d = d + N*((x[i]*1/N)**2)
    w1 = (a-b)/(c-d)

    # Calculating w0 from linear regression formula
    for i in range(0,N):
        e = e + (1/N)*r[i]
        f = f + (1/N*x[i])**2
    w0 = e - w1*f

    # Defining linear model from learned w1 and w0 constants
    def g(x):
        y = w1*x+w0
        return y
    x_axis = range(-3,4)

    print("w1:",w1)
    print("w0:",w0)
    plt.plot(x_axis,g(x_axis),color='red')
    
linear_regression('samples.npy')


# In[ ]:




