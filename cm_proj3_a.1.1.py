'''
    Christian B. Molina
    Phy 104B - Project 03_a Numerical Integration using the Monte Carlo Method
    WQ 2021
'''
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

# Defined Functions ====================================================
def norm_prob(t):
    return math.sqrt(2/math.pi) * math.exp((-t**2)/2)

def integral_norm_prob(numPoints): # N = number of points for integration
    intSum = 0
    n = 0
    while (n < numPoints):
        t = random.random()
        intSum += norm_prob(t); n += 1
    integral = (1/numPoints) * intSum
    return integral

# Initialization =======================================================
# (a) maximum error tolerated / exact value from textbook
eps = 10**-5 
exact_value = 0.6826895
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
        integral_results[i] = integral_norm_prob(N)
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
    print("Iteration: ", i ,"   No. of points: ",prev_N[i], "   A(x=1): ",prev_integrals[i])
    i+=1
print("Percent Error:" , (abs(exact_value - prev_integrals[iterCt])/exact_value)*100,"%")
plt.figure(1)
x = range(-(iterCt+5),(iterCt+5))
y = np.array([exact_value]*(2*(iterCt+5)))
plt.plot(x,y,color='k',linestyle='--',alpha = 0.7, label = "Exact Value = 0.6826895")
plt.bar(range(1,iterCt+1), prev_integrals[1:])
plt.xlim(0,iterCt+1)
plt.ylim(exact_value-0.01,exact_value+0.005)
plt.xticks(range(1,iterCt+1))
plt.title('Convergence of Monte Carlo Integration for A(x=1)', fontsize=12)
plt.xlabel('Iteration',fontsize = 13)
plt.ylabel('Integrand of\nA(x=1)',rotation = 0, fontsize = 13, labelpad = 20)
plt.legend()
plt.show()