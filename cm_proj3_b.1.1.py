'''
    Christian B. Molina
    Phy 104B - Project 03_b Numerical Integration using the Monte Carlo Method
    WQ 2021
'''
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

# Defined Functions ============================================
def yukawa(x,y,z):
    r = math.sqrt((x**2)+(y**2)+(z**2))
    return (1/(math.pi))*math.exp(-r)

def integral_yukawa(numPoints):
    intSum = 0
    n = 0
    while (n < numPoints):
        x = random.random(); y = random.random(); z = random.random()
        intSum += yukawa(x,y,z)
        n +=1
    integral = (1/numPoints) * intSum
    return integral

# Intitialization ==================================================
# (a) exact value / maximum error tolerated / approximated value from textbook
exact_value = 0.126758
approx_value = 0.4/math.pi
eps = exact_value*0.0002
# (b) Let N be the starting number of points for the integration
N = 10
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
        integral_results[i] = integral_yukawa(N)
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
    print("Iteration: ", i ,"   No. of points: ",prev_N[i], "   Yukawa Integral: ",prev_integrals[i])
    i+=1
print("Percent Error:" , (abs(exact_value - prev_integrals[iterCt])/exact_value)*100,"%")
plt.figure(1)
x = range(-(iterCt+5),(iterCt+5))
y = np.array([exact_value]*(2*(iterCt+5)))
y_approx = np.array([approx_value]*(2*(iterCt+5)))
plt.plot(x,y,color='k',linestyle='--',alpha = 0.7, label = "Exact Value = 0.126758")
plt.plot(x,y_approx,color='#808080',linestyle='--',alpha = 0.7, label = r'Approx Value = $0.4/\pi$')
plt.bar(range(1,iterCt+1), prev_integrals[1:])
plt.xlim(0,iterCt+1)
plt.ylim(exact_value-0.01,exact_value+0.005)
plt.xticks(range(1,iterCt+1))
plt.title('Convergence of Monte Carlo Integration for Yukawa', fontsize=12)
plt.xlabel('Iteration',fontsize = 13)
plt.ylabel('Integrand\nof\nYukawa',rotation = 0, fontsize = 13, labelpad = 20)
plt.legend()
plt.show()