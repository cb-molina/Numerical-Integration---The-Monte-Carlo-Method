'''
    Christian B. Molina
    Phy 104B - Project 03_c Numerical Integration using the Monte Carlo Method
    WQ 2021
'''
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

# Defined Functions ====================================================
def func(x):
    return 1/(math.sqrt(1 - x**2))

def integral_func(numPoints): # N = number of points for integration
    intSum = 0
    n = 0
    while (n < numPoints):
        x = random.random()
        while (x == 1 or x >= b):         # This will check if x = 1 or x >= b, if it does, regenerate x
            x = random.random()
        intSum += func(x); n += 1
    integral = (1*b/numPoints) * intSum
    return integral

# Initialization =======================================================
b = 0.25 # b < 1
# (a) maximum error tolerated / exact value from textbook 
exact_value = math.asin(b)
eps = exact_value * 10**-4
# (b) Let N be the starting number of points for the integration
N = 40
# (c) Zero the iteration counter and sum
iterCt = 0
integral_results = np.zeros(3)
# Other initialized items for algorithm
prev_integrals = [0]
prev_N = [0]
seed = 0.5
random.seed(seed)

while(True):
    iterCt+=1
    i = 1
    while (i < 3):
        integral_results[i] = integral_func(N)
        i+=1
    integral_avg = (1/2)*(integral_results[1] + integral_results[2])
    diff = abs(integral_results[1] - integral_results[2])
    if (diff > eps):
        prev_integrals.append(integral_avg)
        prev_N.append(N)
        N = N * 2
    elif (diff <= eps):
        prev_integrals.append(integral_avg)
        prev_N.append(N)
        break

# Outputs ====================================================================
i = 1
while (i <= iterCt):
    print("Iteration: ", i ,"   No. of points: ",prev_N[i], "   I(b={}): ".format(b),prev_integrals[i])
    i+=1

print("Percent Error:" , (abs(exact_value - prev_integrals[iterCt])/exact_value)*100,"%")

plt.figure(1)
x = range(-(iterCt+5),(iterCt+5))
y = np.array([exact_value]*(2*(iterCt+5)))
plt.plot(x,y,color='k',linestyle='--',alpha = 0.7, label = "Exact Value = {:f}".format(exact_value))
plt.bar(range(1,iterCt+1), prev_integrals[1:])
plt.xlim(0,iterCt+1)
plt.ylim(exact_value-0.5,exact_value+0.5)
plt.xticks(range(1,iterCt+1))
plt.title("Convergence of Monte Carlo Integration for I(b={}): ".format(b), fontsize=12)
plt.xlabel('Iteration',fontsize = 13)
plt.ylabel("Integrand\nof\nI(b={})".format(b),rotation = 0, fontsize = 13, labelpad = 20)
plt.legend()
plt.show()